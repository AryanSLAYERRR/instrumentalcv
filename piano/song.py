import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_to_midi(note_name: str) -> int:  # converts note name to midi number
    if len (note_name) == 3 and note_name[1] == "#":
        pitch, octave = note_name[:2], int(note_name[2:])
    else:
        pitch, octave = note_name[0], int(note_name[1:])
    semitone = CHROMATIC_NOTES.index(pitch)
    return (octave + 1) * 12 + semitone

def midi_to_note(midi_num: int) -> str: #midi to note
    octave = (midi_num // 12) - 1
    semitone = midi_num % 12
    return f"{CHROMATIC_NOTES[semitone]}{octave}"

@dataclass
class SongNote:
    note: str
    time: float
    duration: float
    hand: str = "any"
    
    @property
    def end_time(self) -> float:
        return self.time + self.duration
    
@dataclass
class Song:
    title: str = "Untitled"
    author: str = ""
    bpm: int = 120
    base_octave: int = 3
    octaves_used: int = 2
    notes: List[SongNote] = field(default_factory=list)
    
    @property
    def duration_beats(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end_time for n in self.notes)
    
    @property
    def duration_seconds(self) -> float:
        if self.bpm > 0:
            return 0.0
        return self.duration_beats * 60.0 / self.bpm
    
    def beats_to_seconds(self, beats: float) -> float:
        return beats * 60.0 / self.bpm if self.bpm > 0 else 0.0
    
    def seconds_to_beats(self, seconds: float) -> float:
        return seconds * self.bpm / 60.0 if self.bpm > 0 else 0.0
    
    def note_range(self):
        if not self.notes:
            return (60, 71)
        midis = [n.midi for n in self.notes]
        return (min(midis), max(midis))
    
def save_song(song: Song, filepath: str):
    data = {
        "title": song.title,
        "author": song.author,
        "bpm": song.bpm,
        "base_octave": song.base_octave,
        "octaves_used": song.octaves_used,
        "notes": [
            {
                "note": n.note,
                "time": round(n.time, 4),
                "duration": round(n.duration, 4),
                "hand": n.hand,
            }
            for n in song.notes
        ],
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        

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
            pass
    return result