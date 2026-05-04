import json
import os
import re
import contextlib
import io
import tempfile
import warnings
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from argparse import Namespace
from typing import List, Optional, Sequence, Union

CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MIDI_SMART_IMPORT_NOTE_LIMIT = 120_000      # if a MIDI has more notes than this we start dropping, shoudl not happen tho. "probably"
MIDI_DENSITY_WINDOW_BEATS = 0.125           # 1/8th beat window probably overkill to go finer than this
MIDI_MAX_NOTES_PER_WINDOW = 48              
LARGE_SONG_STREAM_NOTE_LIMIT = 100_000    
NOTE_NAME_PATTERN = re.compile(r"^\s*([A-Ga-g])([#bBsS]?)(-?\d+)\s*$")
NATURAL_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# yes this is a real function. yes it detects rush e specifically. don't ask.
# fun
def is_rush_e_name(value: str) -> bool:
    words = "".join(char.lower() if char.isalnum() else " " for char in str(value)).split()
    compact = "".join(words)
    return any(left == "rush" and right == "e" for left, right in zip(words, words[1:])) or "rushe" in compact

def note_to_midi(note_name: str) -> int:
    # e.g. "C4" -> 60, "A4" -> 69. middle C is C4 = 60, standard MIDI convention
    match = NOTE_NAME_PATTERN.match(str(note_name))
    if not match:
        raise ValueError(f"Invalid note name: {note_name!r}")
    pitch, accidental, octave_text = match.groups()
    semitone = NATURAL_NOTE_OFFSETS[pitch.upper()]
    accidental = accidental.upper()
    if accidental in {"#", "S"}:    # "S" for sharp as some notation uses it
        semitone += 1
    elif accidental == "B":         # "B" for flat, careful not to confuse with the note B
        semitone -= 1
    octave = int(octave_text)
    return (octave + 1) * 12 + semitone

def midi_to_note(midi_num: int) -> str:
    octave = (midi_num // 12) - 1
    semitone = midi_num % 12
    return f"{CHROMATIC_NOTES[semitone]}{octave}"

@dataclass
class SongNote:
    note: str
    time: float       # in beats, not seconds convert with bpm when you need real time
    duration: float   # also in beats
    hand: str = "any" # "left", "right", or "any" used by the UI to color-code notes

    @property
    def midi(self) -> int:
        return note_to_midi(self.note)

    @property
    def end_time(self) -> float:
        return self.time + self.duration  # handy shortcut, used all over the place

@dataclass
class Song:
    title: str = "Untitled"
    author: str = ""
    bpm: int = 120
    base_octave: int = 3    # lowest octave shown on the keyboard, 3 or 4 covers most songs
    octaves_used: int = 2   # how many octaves wide the song's range is
    notes: List[SongNote] = field(default_factory=list)

    @property
    def duration_beats(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end_time for n in self.notes)  # latest note end, not latest note start

    @property
    def duration_seconds(self) -> float:
        if self.bpm <= 0:
            return 0.0
        return self.duration_beats * 60.0 / self.bpm

    def beats_to_seconds(self, beats: float) -> float:
        return beats * 60.0 / self.bpm if self.bpm > 0 else 0.0

    def seconds_to_beats(self, seconds: float) -> float:
        return seconds * self.bpm / 60.0 if self.bpm > 0 else 0.0

    def note_range(self):
        if not self.notes:
            return (60, 71)
        low = high = self.notes[0].midi
        for note in self.notes[1:]:
            midi = note.midi
            low = min(low, midi)
            high = max(high, midi)
        return (low, high)


def _note_payload(note: SongNote):
    return {
        "note": note.note,
        "time": round(note.time, 4),
        "duration": round(note.duration, 4),
        "hand": note.hand,
    }


def _notes_are_sorted(notes: List[SongNote]) -> bool:
    previous = None
    for note in notes:
        key = (note.time, note.note)
        if previous is not None and key < previous:
            return False
        previous = key
    return True

def save_song(song: Song, filepath: str):
    # big songs get streamed line by line so we don't blow up RAM building one giant dict
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if len(song.notes) <= LARGE_SONG_STREAM_NOTE_LIMIT:
        data = {
            "title": song.title,
            "author": song.author,
            "bpm": song.bpm,
            "base_octave": song.base_octave,
            "octaves_used": song.octaves_used,
            "notes": [_note_payload(n) for n in song.notes],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    # large song path: write the JSON manually so we never hold all notes in memory at once
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "title": {json.dumps(song.title)},\n')
        f.write(f'  "author": {json.dumps(song.author)},\n')
        f.write(f'  "bpm": {int(song.bpm)},\n')
        f.write(f'  "base_octave": {int(song.base_octave)},\n')
        f.write(f'  "octaves_used": {int(song.octaves_used)},\n')
        f.write('  "notes": [\n')
        last_index = len(song.notes) - 1
        for index, note in enumerate(song.notes):
            suffix = "," if index < last_index else ""  # trailing comma would break JSON parsers
            payload = json.dumps(_note_payload(note), separators=(",", ":"))  # compact one note per line is already verbose enough
            f.write(f"    {payload}{suffix}\n")
        f.write("  ]\n")
        f.write("}\n")


def load_song(filepath: str) -> Song:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    notes = [
        SongNote(
            note=n["note"],
            time=float(n["time"]),
            duration=float(n["duration"]),
            hand=n.get("hand", "any"),
        )
        for n in data.get("notes", [])
    ]
    # always re-sort on load, don't trust that whoever saved this kept the order
    notes.sort(key=lambda n: (n.time, n.note))

    return Song(
        title=data.get("title", "Untitled"),
        author=data.get("author", ""),
        bpm=int(data.get("bpm", 120)),
        base_octave =int(data.get("base_octave", 3)),
        octaves_used=int(data.get("octaves_used", 2)),
        notes=notes,
    )

def list_songs(songs_dir: str) -> List[tuple]:
    if not os.path.isdir(songs_dir):
        return []
    result = []
    for filename in sorted(os.listdir(songs_dir)):
        if not filename.endswith(".json"):
            continue
        try:
            song = load_song(os.path.join(songs_dir, filename))
            result.append((filename, song))
        except Exception:
            pass  # silently skip corrupt/invalid files
    return result

def remap_song(song: Song, target_octaves: int, target_base: int) -> Song:
    # squishes a wide-range song into fewer octaves using modulo wrapping
    # sounds weird on complex pieces but makes things playable on a small keyboard
    # tried pitch folding (mirror instead of wrap), modulo sounded wayyy more natural
    target_min = note_to_midi(f"C{target_base}")
    target_max = note_to_midi(f"B{target_base + target_octaves - 1}")
    target_range = target_max - target_min + 1
    remapped = []

    for n in song.notes:
        midi = n.midi
        offset = (midi - target_min) % target_range
        new_midi = target_min + offset
        remapped.append(
            SongNote(
                note=midi_to_note(new_midi),
                time=n.time,
                duration=n.duration,
                hand=n.hand,
            )
        )
    return Song(
        title=song.title,
        author=song.author,
        bpm=song.bpm,
        base_octave=target_base,
        octaves_used=target_octaves,
        notes=remapped,
    )


def _song_range_from_notes(notes: List[SongNote], fallback_base: int = 4):
    if not notes:
        return fallback_base, 1
    midis = [n.midi for n in notes]
    low_oct = (min(midis) // 12) - 1
    high_oct = (max(midis) // 12) - 1
    return low_oct, high_oct - low_oct + 1


def _trim_leading_silence(song: Song) -> Song:
    if not song.notes:
        return song
    first_time = min(note.time for note in song.notes)
    if first_time <= 0:
        return song
    notes = [
        SongNote(note=note.note, time=max(0.0, note.time - first_time), duration=note.duration, hand=note.hand)
        for note in song.notes
    ]
    return Song(
        title=song.title,
        author=song.author,
        bpm=song.bpm,
        base_octave=song.base_octave,
        octaves_used=song.octaves_used,
        notes=notes,
    )


def smooth_sheet_scan_timing(song: Song, max_gap_beats: float = 4.0, target_gap_beats: float = 1.0) -> Song:
    # gaps bigger than max_gap_beats get squished down to target_gap_beats
    # doesn't touch gaps that are already reasonable
    if len(song.notes) < 2:
        return song
    adjusted = []
    shift = 0.0
    last_end = 0.0
    for note in sorted(song.notes, key=lambda item: (item.time, item.note)):
        start = max(0.0, note.time - shift)
        gap = start - last_end
        if gap > max_gap_beats:
            extra = gap - target_gap_beats
            shift += extra
            start -= extra
        adjusted_note = SongNote(note=note.note, time=round(start, 4), duration=note.duration, hand=note.hand)
        adjusted.append(adjusted_note)
        last_end = max(last_end, adjusted_note.end_time)
    base_octave, octaves_used = _song_range_from_notes(adjusted, song.base_octave)
    return Song(
        title=song.title,
        author=song.author,
        bpm=song.bpm,
        base_octave=base_octave,
        octaves_used=octaves_used,
        notes=adjusted,
    )


def _open_midi_file(filepath: str):
    import mido

    try:
        return mido.MidiFile(filepath, clip=True)
    except TypeError:
        return mido.MidiFile(filepath)  # older mido versions don't have clip=


def _tempo_events(mid):
    import mido

    # MIDI files can have multiple tempo changes mid-song (accel, rit, etc.)
    # we collect all of them so _tempo_converter can do accurate tick-to-second mapping
    default_tempo = mido.bpm2tempo(120)
    events = [(0, default_tempo)]
    for midi_track in mid.tracks:
        tick = 0
        for msg in midi_track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append((tick, msg.tempo))

    events.sort(key=lambda item: item[0])
    deduped = []
    for tick, tempo in events:
        if deduped and deduped[-1][0] == tick:
            deduped[-1] = (tick, tempo)  # last tempo event at a given tick wins
        else:
            deduped.append((tick, tempo))
    return deduped


def _tempo_converter(tempo_events, ticks_per_beat):
    import mido

    # builds a fast lookup: given a tick position, what's the elapsed time in seconds?
    # uses bisect to find the right tempo segment O(log n) per note, good enough
    # returns a closure so callers don't have to pass around all this state
    segment_ticks = []
    segment_seconds = []
    segment_tempos = []
    elapsed_seconds = 0.0
    last_tick = 0
    last_tempo = tempo_events[0][1]

    for tick, tempo in tempo_events:
        if tick > last_tick:
            elapsed_seconds += mido.tick2second(tick - last_tick, ticks_per_beat, last_tempo)
        segment_ticks.append(tick)
        segment_seconds.append(elapsed_seconds)
        segment_tempos.append(tempo)
        last_tick = tick
        last_tempo = tempo

    def to_seconds(tick):
        index = max(0, bisect_right(segment_ticks, tick) - 1)
        return segment_seconds[index] + mido.tick2second(tick - segment_ticks[index], ticks_per_beat, segment_tempos[index])

    return to_seconds


def _append_imported_midi_note(notes, note, density_counts, stats, smart_limit):
    # smart_limit is for "performance" MIDIs things like Rush E where the raw note count is technically correct but completely unplayable
    # density check drops notes when a single 1/8th beat window is already packed
    stats["source_notes"] += 1  # always track source count even if we skip it
    if smart_limit and len(notes) >= MIDI_SMART_IMPORT_NOTE_LIMIT:
        stats["skipped_limit"] += 1
        return

    if smart_limit:
        bucket = int(note.time / MIDI_DENSITY_WINDOW_BEATS)
        if density_counts.get(bucket, 0) >= MIDI_MAX_NOTES_PER_WINDOW:
            stats["skipped_density"] += 1
            return
        density_counts[bucket] = density_counts.get(bucket, 0) + 1

    notes.append(note)


def import_midi(filepath: str, track: Optional[int] = None, smart_limit: bool = False) -> Song:
    # if track is None we merge ALL tracks works for most MIDIs
    # some MIDIs keep melody on track 1, accompaniment on track 2, etc.
    # passing a specific track lets you grab just the melody
    try:
        import mido
    except ImportError:
        raise ImportError("Install 'mido' to import MIDI files: pip install mido")

    mid = _open_midi_file(filepath)
    ticks_per_beat = mid.ticks_per_beat
    default_tempo = mido.bpm2tempo(120)
    tempo_changes = _tempo_events(mid)
    unique_tempos = list(dict.fromkeys(tempo for _tick, tempo in tempo_changes)) or [default_tempo]
    single_tempo = len(unique_tempos) == 1
    # bpm=60 for variable-tempo songs is a lie but it makes the internal beat math consistent
    bpm = int(round(mido.tempo2bpm(unique_tempos[0]))) if single_tempo else 60
    to_seconds = None if single_tempo else _tempo_converter(tempo_changes, ticks_per_beat)
    notes = []
    density_counts = {}
    stats = {"source_notes": 0, "skipped_density": 0, "skipped_limit": 0}

    if track is not None:
        if track < 0 or track >= len(mid.tracks):
            raise ValueError(f"MIDI track {track} is out of range for {len(mid.tracks)} track(s).")
        tracks_to_process = [(track, mid.tracks[track])]
    else:
        tracks_to_process = list(enumerate(mid.tracks))

    for track_index, midi_track in tracks_to_process:
        tick = 0
        active_notes = {}
        for msg in midi_track:
            tick += msg.time
            if msg.type not in ("note_on", "note_off"):
                continue

            channel = getattr(msg, "channel", 0)
            active_key = (channel, msg.note)
            # note_on with velocity 0 is technically a note_off — a lot of MIDIs do this
            is_note_on = msg.type == "note_on" and msg.velocity > 0

            if is_note_on:
                active_notes.setdefault(active_key, deque()).append((tick, msg.velocity))
                continue

            starts = active_notes.get(active_key)
            if not starts:
                continue
            start_tick, velocity = starts.popleft()
            if not starts:
                active_notes.pop(active_key, None)

            if single_tempo:
                start = start_tick / ticks_per_beat
                # minimum 0.125 beats — without this floor, grace notes cause rendering issues
                duration = max((tick - start_tick) / ticks_per_beat, 0.125)
            else:
                start_seconds = to_seconds(start_tick)
                end_seconds = to_seconds(tick)
                start = start_seconds
                duration = max(end_seconds - start_seconds, 0.075)  # same idea, 75ms floor for time-space

            _append_imported_midi_note(
                notes,
                SongNote(note=midi_to_note(msg.note), time=start, duration=duration),
                density_counts,
                stats,
                smart_limit,
            )

    if not _notes_are_sorted(notes):
        notes.sort(key=lambda n: (n.time, n.note))

    if notes:
        midis = [n.midi for n in notes]
        low_oct = (min(midis) // 12) - 1
        high_oct = (max(midis) // 12) - 1
        base_octave = low_oct
        octaves_used = high_oct - low_oct + 1
    else:
        base_octave, octaves_used = 4, 1

    title = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()
    skipped = stats["skipped_density"] + stats["skipped_limit"]
    author = "MIDI Import"
    if skipped:
        title = f"{title} (Performance Import)"
        author = (
            "MIDI Import, smart-reduced "
            f"{stats['source_notes']:,} source notes to {len(notes):,} playable notes"
        )
    elif stats["source_notes"]:
        author = f"MIDI Import, full source notes: {stats['source_notes']:,}"

    return Song(title=title, author=author, bpm=bpm, base_octave=base_octave, octaves_used=octaves_used, notes=notes,)

def import_musicxml(filepath: str) -> Song:
    try:
        import music21
    except ImportError:
        raise ImportError("Install 'music21' to import MusicXML files: pip install music21")

    score = music21.converter.parse(filepath)
    flat = score.flatten()
    bpm = 120
    markings = flat.getElementsByClass(music21.tempo.MetronomeMark)
    if markings:
        number = markings[0].number
        if number:
            bpm = int(round(number))

    notes = []
    for element in flat.notes:
        start = float(element.offset)
        duration = max(float(element.duration.quarterLength), 0.125)
        if isinstance(element, music21.chord.Chord):
            for pitch in element.pitches:
                notes.append(SongNote(note=midi_to_note(int(pitch.midi)), time=start, duration=duration))
        elif isinstance(element, music21.note.Note):
            notes.append(SongNote(note=midi_to_note(int(element.pitch.midi)), time=start, duration=duration))

    if not _notes_are_sorted(notes):
        notes.sort(key=lambda n: (n.time, n.note))
    base_octave, octaves_used = _song_range_from_notes(notes)

    metadata_title = getattr(score.metadata, "title", None) if score.metadata else None
    title = metadata_title or os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title()
    return Song(
        title=title,
        author="MusicXML Import",
        bpm=bpm,
        base_octave=base_octave,
        octaves_used=octaves_used,
        notes=notes,
    )

def import_sheet_image(image_path: str) -> Song:
    # this will also handle MIDI/MusicXML if you somehow pass one in here just delegates down
    # actual image scanning uses oemer (optical music recognition) results vary a LOT by image quality
    # clean, high-contrast, flat scans work well photos of books don't
    ext = Path(image_path).suffix.lower()
    if ext in {".musicxml", ".xml", ".mxl"}:
        return import_musicxml(image_path)
    if ext in {".mid", ".midi"}:
        return import_midi(image_path)
    if ext == ".pdf":
        return import_sheet_files([image_path])

    numpy_stack_message = (
        "Sheet scan dependencies are incompatible. Reinstall the project requirements so NumPy stays below 2.0 "
        "and scikit-learn is rebuilt against it."
    )
    onnx_message = (
        "Sheet scan could not start ONNX Runtime. Reinstall requirements so the CPU ONNX Runtime is active: "
        "python -m pip install --force-reinstall onnxruntime>=1.17,<1.20 numpy>=1.24,<2 protobuf>=3.11,<4"
    )
    try:
        import oemer
        from oemer.ete import CHECKPOINTS_URL, clear_data, download_file, extract
    except ImportError:
        raise ImportError(
            "Install 'oemer' for sheet music scanning: pip install oemer\n"
            "NOTE: This is experimental, may not give accurate results"
        )
    except ValueError as exc:
        if "numpy.dtype size changed" in str(exc):
            raise RuntimeError(numpy_stack_message) from exc
        raise

    def force_onnx_cpu_provider():
        # oemer's default tries GPU providers and will crash on most machines(did on mine)
        # patch InferenceSession globally before oemer touches it
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(onnx_message) from exc
        if getattr(ort, "_instrumentalcv_cpu_patch", False):
            return
        original_session = ort.InferenceSession

        def cpu_session(*args, **kwargs):
            kwargs["providers"] = ["CPUExecutionProvider"]
            try:
                return original_session(*args, **kwargs)
            except Exception as exc:
                raise RuntimeError(onnx_message) from exc

        ort.InferenceSession = cpu_session
        ort._instrumentalcv_cpu_patch = True

    def ensure_oemer_checkpoints():
        # oemer needs two model files downloads them if missing
        # if there's no internet this throws a clear error instead of a cryptic one
        checkpoints = [
            (os.path.join(oemer.MODULE_PATH, "checkpoints", "unet_big", "model.onnx"), "1st_model.onnx", "unet_big"),
            (os.path.join(oemer.MODULE_PATH, "checkpoints", "unet_big", "weights.h5"), "1st_weights.h5", "unet_big"),
            (os.path.join(oemer.MODULE_PATH, "checkpoints", "seg_net", "model.onnx"), "2nd_model.onnx", "seg_net"),
            (os.path.join(oemer.MODULE_PATH, "checkpoints", "seg_net", "weights.h5"), "2nd_weights.h5", "seg_net"),
        ]
        for target_path, title, folder in checkpoints:
            if os.path.exists(target_path):
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            url = CHECKPOINTS_URL[title]
            try:
                download_file(title, url, target_path)
            except Exception as exc:
                raise RuntimeError(
                    "Oemer model files are missing and could not be downloaded. "
                    "Import MIDI or MusicXML from a scanner app, or run the sheet scan again with internet access."
                ) from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        force_onnx_cpu_provider()
        ensure_oemer_checkpoints()
        clear_data()
        args = Namespace(
            img_path=image_path,
            output_path=tmp_dir,
            use_tf=False,
            save_cache=False,
            without_deskew=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
            warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
            # redirect stdout because oemer prints a wall of progress text we don't want
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    musicxml_path = extract(args)
                except ValueError as exc:
                    if "numpy.dtype size changed" in str(exc):
                        raise RuntimeError(numpy_stack_message) from exc
                    raise
        if not musicxml_path or not os.path.exists(musicxml_path):
            musicxml_files = [f for f in os.listdir(tmp_dir) if f.endswith(".musicxml")]
            if not musicxml_files:
                raise RuntimeError("Oemer did not produce MusicXML. Try a brighter, flatter image.")
            musicxml_path = os.path.join(tmp_dir, musicxml_files[0])
        song = import_musicxml(musicxml_path)
        # post process trim silence at the start, collapse long gaps between staves
        song = smooth_sheet_scan_timing(_trim_leading_silence(song))
        song.title = os.path.splitext(os.path.basename(image_path))[0].replace("_", " ").title()
        song.author = "Sheet Scan"
        return song


def _render_pdf_pages(pdf_path: str, output_dir: str) -> List[str]:
    try:
        import fitz
    except ImportError:
        raise ImportError("Install PyMuPDF to scan PDF sheet music: pip install PyMuPDF")

    rendered = []
    document = fitz.open(pdf_path)
    if document.page_count == 0:
        raise RuntimeError("PDF has no pages to scan.")
    stem = Path(pdf_path).stem or "sheet"
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)  # 2x scale — oemer needs decent resolution
        output_path = os.path.join(output_dir, f"{stem}_page_{page_index + 1:03d}.png")
        pixmap.save(output_path)
        rendered.append(output_path)
    document.close()
    return rendered


def _merge_sheet_songs(songs: List[Song], title: str, page_gap_beats: float = 0.5) -> Song:
    # PDF imports come in page by page stitch them together with a small gap
    # 0.5 beats between pages feels natural, big enough to breathe but not awkward
    merged_notes = []
    bpm = next((song.bpm for song in songs if song.bpm > 0), 120)
    current_end = 0.0
    for song in songs:
        page_song = _trim_leading_silence(song)
        if not page_song.notes:
            continue
        offset = current_end + (page_gap_beats if merged_notes else 0.0)
        for note in page_song.notes:
            merged_notes.append(
                SongNote(
                    note=note.note,
                    time=round(note.time + offset, 4),
                    duration=note.duration,
                    hand=note.hand,
                )
            )
        current_end = max(note.end_time for note in merged_notes)

    merged_notes.sort(key=lambda note: (note.time, note.note))
    base_octave, octaves_used = _song_range_from_notes(merged_notes)
    merged = Song(
        title=title,
        author="Sheet Scan",
        bpm=bpm,
        base_octave=base_octave,
        octaves_used=octaves_used,
        notes=merged_notes,
    )
    return smooth_sheet_scan_timing(merged)


def import_sheet_files(paths: Union[str, Sequence[str]]) -> Song:
    if isinstance(paths, (str, os.PathLike)):
        source_paths = [str(paths)]
    else:
        source_paths = [str(path) for path in paths]
    if not source_paths:
        raise ValueError("Choose at least one sheet image or PDF.")

    songs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        scan_paths = []
        for source_path in source_paths:
            ext = Path(source_path).suffix.lower()
            if ext == ".pdf":
                scan_paths.extend(_render_pdf_pages(source_path, tmp_dir))
            elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                scan_paths.append(source_path)
            elif ext in {".musicxml", ".xml", ".mxl"}:
                songs.append(import_musicxml(source_path))
            elif ext in {".mid", ".midi"}:
                songs.append(import_midi(source_path))
            else:
                raise ValueError(f"Unsupported sheet scan file: {Path(source_path).name}")

        for scan_index, scan_path in enumerate(scan_paths, start=1):
            try:
                songs.append(import_sheet_image(scan_path))
            except Exception as exc:
                raise RuntimeError(f"Page/image {scan_index} failed: {exc}") from exc

    if not songs:
        raise RuntimeError("Sheet scan did not produce any songs.")
    if len(songs) == 1:
        song = songs[0]
        source_title = Path(source_paths[0]).stem.replace("_", " ").title()
        song.title = source_title or song.title
        return song

    if len(source_paths) == 1:
        title = Path(source_paths[0]).stem.replace("_", " ").title()
    else:
        title = "Sheet Scan Import"
    return _merge_sheet_songs(songs, title)


def create_example_songs(songs_dir: str):
    # these ship with the app so new users have something to play immediately
    # kept intentionally simple single octave, slow bpm, well-known melodies
    os.makedirs(songs_dir, exist_ok=True)

    save_song(
        Song(
            title="Twinkle Twinkle Little Star",
            author="Jane Taylor",
            bpm=100,
            base_octave=4,
            octaves_used=1,
            notes=[
                SongNote("C4", 0, 1), SongNote("C4", 1, 1),
                SongNote("G4", 2, 1), SongNote("G4", 3, 1),
                SongNote("A4", 4, 1), SongNote("A4", 5, 1),
                SongNote("G4", 6, 2),
                SongNote("F4", 8, 1), SongNote("F4", 9, 1),
                SongNote("E4", 10, 1), SongNote("E4", 11, 1),
                SongNote("D4", 12, 1), SongNote("D4", 13, 1),
                SongNote("C4", 14, 2),
            ],
        ),
        os.path.join(songs_dir, "twinkle_twinkle.json"),
    )

    save_song(
        Song(
            title="Happy Birthday",
            author="Patty Hill & Mildred J. Hill",
            bpm=120,
            base_octave=4,
            octaves_used=1,
            notes=[
                SongNote("C4", 0, 0.75), SongNote("C4", 0.75, 0.25),
                SongNote("D4", 1, 1), SongNote("C4", 2, 1),
                SongNote("F4", 3, 1), SongNote("E4", 4, 2),
                SongNote("C4", 6, 0.75), SongNote("C4", 6.75, 0.25),
                SongNote("D4", 7, 1), SongNote("C4", 8, 1),
                SongNote("G4", 9, 1), SongNote("F4", 10, 2),
            ],
        ),
        os.path.join(songs_dir, "happy_birthday.json"),
    )

    save_song(
        Song(
            title="C Major Scale",
            author="Practice",
            bpm=90,
            base_octave=4,
            octaves_used=2,
            notes=[
                SongNote("C4", 0, 1), SongNote("D4", 1, 1),
                SongNote("E4", 2, 1), SongNote("F4", 3, 1),
                SongNote("G4", 4, 1), SongNote("A4", 5, 1),
                SongNote("B4", 6, 1), SongNote("C5", 7, 2),
            ],
        ),
        os.path.join(songs_dir, "c_major_scale.json"),
    )

    save_song(
        Song(
            title="Ode to Joy",
            author="Beethoven",
            bpm=108,
            base_octave=4,
            octaves_used=1,
            notes=[
                SongNote("E4", 0, 1), SongNote("E4", 1, 1),
                SongNote("F4", 2, 1), SongNote("G4", 3, 1),
                SongNote("G4", 4, 1), SongNote("F4", 5, 1),
                SongNote("E4", 6, 1), SongNote("D4", 7, 1),
                SongNote("C4", 8, 1), SongNote("C4", 9, 1),
                SongNote("D4", 10, 1), SongNote("E4", 11, 1),
                SongNote("E4", 12, 1.5), SongNote("D4", 13.5, 0.5),
                SongNote("D4", 14, 2),
            ],
        ),
        os.path.join(songs_dir, "ode_to_joy.json"),
    )
    print(f"Created example songs in {songs_dir}")



if __name__ == "__main__":
    import sys
    songs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        create_example_songs(songs_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--import-midi":
        if len(sys.argv) < 3:
            print("Usage: python song.py --import-midi <file.mid>")
            sys.exit(1)
        song = import_midi(sys.argv[2], smart_limit=False)
        out_name = os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json"
        out_path = os.path.join(songs_dir, out_name)
        save_song(song, out_path)
        print(f"Imported '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM) -> {out_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--import-midi-smart":
        if len(sys.argv) < 3:
            print("Usage: python song.py --import-midi-smart <file.mid>")
            sys.exit(1)
        song = import_midi(sys.argv[2], smart_limit=True)
        out_name = os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json"
        out_path = os.path.join(songs_dir, out_name)
        save_song(song, out_path)
        print(f"Imported '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM) -> {out_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--import-sheet":
        if len(sys.argv) < 3:
            print("Usage: python song.py --import-sheet <image.png>")
            sys.exit(1)
        song = import_sheet_image(sys.argv[2])
        out_name = os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json"
        out_path = os.path.join(songs_dir, out_name)
        save_song(song, out_path)
        print(f"Imported '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM) -> {out_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--import-musicxml":
        if len(sys.argv) < 3:
            print("Usage: python song.py --import-musicxml <file.musicxml>")
            sys.exit(1)
        song = import_musicxml(sys.argv[2])
        out_name = os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json"
        out_path = os.path.join(songs_dir, out_name)
        save_song(song, out_path)
        print(f"Imported '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM) -> {out_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        for filename, song in list_songs(songs_dir):
            dur = f"{song.duration_seconds:.1f}s"
            print(f"  {filename:30s}  {song.title:30s}  {song.bpm:>3d} BPM  {len(song.notes):>3d} notes  {dur}")
    else:
        print("Usage:")
        print("  python song.py --examples          Generate example songs")
        print("  python song.py --import-midi FILE   Import a full MIDI file")
        print("  python song.py --import-midi-smart FILE Import a reduced performance MIDI")
        print("  python song.py --import-musicxml FILE Import a MusicXML file")
        print("  python song.py --import-sheet FILE  Import a scanned sheet image")
        print("  python song.py --list               List all songs")