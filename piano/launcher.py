import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# local stuff
from roast_lines import LAUNCHER_ROASTS, RUSH_E_AUTOPLAY_LABEL, RUSH_E_LOCK_MESSAGE
from song import (
    Song, SongNote,
    import_midi, import_musicxml, import_sheet_files,
    is_rush_e_name, load_song, note_to_midi, save_song
)


APP_DIR = Path(__file__).resolve().parent
HAND_TRACKING_SCRIPT = APP_DIR / "hand_tracking.py"
SETTINGS_FILE = APP_DIR / "user_settings.json"
SONGS_DIR = APP_DIR / "songs"
PREVIEW_WIDTH = 380
PREVIEW_HEIGHT = 214
CAMERA_DETECT_LIMIT = 1  # checking more than 1 takes forever on some machines
MAX_PIANO_OCTAVES = 4
EDITOR_RENDER_NOTE_LIMIT = 250  # crashes if a huge song is opened in editor window, therefore limit
PERFORMANCE_AUTOPLAY_NOTE_THRESHOLD = 80_000  # tries performance mode for songs like rushe (very dense note composition)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

BG = "#070a10"
PANEL = "#101722"
PANEL_ALT = "#172233"
PANEL_SOFT = "#0b1018"
BORDER = "#263449"
BORDER_HOT = "#36516b"
TEXT = "#f5f7fb"
TEXT_MUTED = "#a3b0c2"
ACCENT = "#00a8ff"
ACCENT_HOVER = "#047dc1"
ACCENT_WARM = "#ffb86b"
ACCENT_MINT = "#3ddc97"
SUCCESS = "#3ddc97"
WARNING = "#ffcf5c"
DANGER = "#ff5c7a"
LEARNING_AUTOPLAY_LABEL = 'Autoplay notes with sound (toggle live with P)'
MODE_OPTIONS = ["Air Piano"]
RESOLUTION_OPTIONS = ["1280 x 720", "960 x 540", "640 x 480"]
PIANO_OCTAVE_OPTIONS = ["1", "2", "3", "4"]
MAX_HAND_OPTIONS = ["1", "2"]
PLAY_TRIGGER_OPTIONS = ["Precision Tap", "Tap to Play", "Hover to Play"]
PLAY_TRIGGER_ARGS = {
    "Precision Tap": "precision",
    "Tap to Play": "tap",
    "Hover to Play": "hover",
}
# tooltips, shows when you hover over options
CONTROL_HELP = {
    "mode": "Chooses the app mode. Air Piano uses your webcam and the on-screen keybed.",
    "camera": "Chooses which webcam device the preview and piano tracker will use.",
    "resolution": "Sets the camera capture size. Higher resolutions look sharper but can reduce FPS.",
    "mirror": "Flips the camera horizontally so your hand movement feels like a mirror.",
    "preview": "Starts or stops the small camera preview inside the launcher.",
    "test_camera": "Opens a quick camera check using the selected camera device.",
    "instrument": "Chooses the sample pack used for piano note sounds.",
    "octave": "Sets the first visible octave on the on-screen piano.",
    "piano_octaves": "Controls how many octaves are visible and playable at once.",
    "volume": "Sets the maximum playback volume for note samples.",
    "fadeout_ms": "Controls how long notes fade after your finger releases them.",
    "play_trigger": "Chooses how a finger starts a note: precision tap, basic tap, or hover.",
    "max_hands": "Sets whether MediaPipe tracks one hand or both hands.",
    "detection_confidence": "Minimum confidence required before a new hand is detected.",
    "tracking_confidence": "Minimum confidence required to keep tracking an already detected hand.",
    "tracking_scale": "Runs hand tracking on a smaller or larger copy of the frame. Lower is faster, higher is more accurate.",
    "smoothing": "Averages recent fingertip positions to reduce jitter.",
    "trigger_cooldown_ms": "Minimum time before the same finger can trigger another note.",
    "dead_zone": "Shrinks the playable edge of each key to avoid accidental neighboring notes.",
    "show_landmarks": "Draws the hand skeleton over the camera feed while playing.",
    "show_fps": "Shows the live frame rate in the piano window.",
    "keybed_preset": "Applies saved keybed height and opacity settings for common camera setups.",
    "piano_top": "Moves the top edge of the playable piano keybed up or down.",
    "piano_bottom": "Moves the bottom edge of the playable piano keybed up or down.",
    "piano_opacity": "Changes how transparent the on-screen piano keys are over the camera view.",
    "show_note_labels": "Shows note names on the on-screen piano keys.",
    "show_note_trail": "Shows a short history of recently played notes in the piano window.",
    "learning_enabled": "Starts the session with falling-note learning mode enabled.",
    "learning_song": "Chooses the song used for falling-note practice.",
    "learning_autoplay": "Plays learning-mode notes automatically and can be toggled with P.",
    "performance_autoplay": "Uses the lighter autoplay renderer for very dense songs to avoid lag.",
    "editor_song": "Chooses which saved song to open in the song editor.",
    "metronome": "Starts the metronome automatically when the piano launches.",
    "metronome_bpm": "Sets the metronome tempo in beats per minute.",
    "download_all_samples": "Selects every instrument sample pack for download.",
    "download_grand": "Downloads or refreshes the Grand Piano sample pack.",
    "download_bright": "Downloads or refreshes the Bright Piano sample pack.",
    "download_electronic": "Downloads or refreshes the Electric Piano sample pack.",
    "download_organ": "Downloads or refreshes the Organ sample pack.",
    "download_reverb": "Downloads or refreshes the Reverb Piano sample pack.",
    "download_samples": "Downloads the selected instrument sample packs.",
}

# maps display name -> folder where wav samples are present
INSTRUMENTS = {
    'Grand Piano': "sounds",
    'Bright Piano': "sounds_bright",
    'Electric Piano': "sounds_electronic",
    'Organ': "sounds_organ",
    'Reverb Piano': "sounds_reverb",
}

SAMPLE_OPTIONS = [
    ('download_grand', 'Grand Piano', 'grand'),
    ('download_bright', 'Bright Piano', 'bright'),
    ('download_electronic', 'Electric Piano', 'electronic'),
    ("download_organ", "Organ", "organ"),
    ("download_reverb", "Reverb Piano", "reverb"),
]
KEYBED_PRESETS = {
    'Laptop webcam': {'piano_top': 34, 'piano_bottom': 66, 'piano_opacity': 78},
    'Phone tripod': {'piano_top': 42, 'piano_bottom': 78, 'piano_opacity': 76},
    'Wide room': {"piano_top": 28, "piano_bottom": 58, "piano_opacity": 82},
    'Close-up hands': {"piano_top": 45, "piano_bottom": 86, "piano_opacity": 74},
    'Custom': None,
}
HAND_OPTIONS = ['any', 'left', 'right']
FLAT_TO_SHARP = {  # midi lookup only knows sharps
    'DB': 'C#',
    'EB': 'D#',
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}
DEFAULT_SETTINGS = {
    "mode": "Air Piano",
    "camera": "0",
    "instrument": "Grand Piano",
    "octave": 3,
    "piano_octaves": "2",
    "volume": 70,
    "resolution": "1280 x 720",
    "tracking_scale": 55,
    "max_hands": "2",
    "detection_confidence": 50,
    "tracking_confidence": 50,
    "mirror": True,
    "show_landmarks": False,
    "show_note_labels": True,
    "show_fps": True,
    "show_note_trail": False,
    "piano_top": 30,
    "piano_bottom": 60,
    "piano_opacity": 78,
    "play_trigger": "Precision Tap",
    "trigger_cooldown_ms": 150,
    "smoothing": 3,
    "dead_zone": 3,
    "fadeout_ms": 300,
    "metronome": False,
    "metronome_bpm": 120,
    "download_all_samples": True,
    "download_grand": True,
    "download_bright": True,
    "download_electronic": True,
    "download_organ": True,
    "download_reverb": True,
    "keybed_preset": "Laptop webcam",
    "learning_enabled": False,
    "learning_song": "",
    "learning_autoplay": False,
    "performance_autoplay": True,
    "editor_song": "",
}


# gradient helpers
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*[int(c) for c in rgb])


def _blend_hex(start, end, t):
    a, b = _hex_to_rgb(start), _hex_to_rgb(end)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


class HeroVisualizer(tk.Canvas):  # falling note blocks in the header
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
            **kwargs,
        )
        self.phase = 0
        self.bind("<Configure>", lambda _event: self._draw())
        self.after(80, self._animate)

    def _animate(self):
        if not self.winfo_exists():
            return
        self.phase = (self.phase + 1) % 240
        self._draw()
        self.after(80, self._animate)

    def _draw(self):
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        self.delete("all")

        for y in range(h):
            color = _blend_hex("#07111d", "#181123", y / max(1, h - 1))
            self.create_line(0, y, w, y, fill=color)

        # offset by ~25px because the canvas border was clipping the right edge for some reason
        self.create_polygon((w // 2) + 25, 0, w, 0, w, h * 2 // 5, (w * 3 // 4) - 10, h // 3, fill="#102333", outline="")
        self.create_polygon(0, h - h // 4, w // 4, h, 0, h, fill="#231722", outline="")

        key_top = (h * 5) // 9
        key_bottom = h
        key_count = 14
        key_w = w / key_count

        colors = ['#00a8ff', '#ffb86b', '#3ddc97', '#ff5c7a']
        for i in range(11):
            lane = (i * 3 + self.phase // 16) % key_count
            x_center = int(lane * key_w + key_w / 2)
            note_h = 28 + (i % 3) * 12
            y = int((self.phase * (2 + i % 3) + i * 31) % max(1, key_bottom + note_h))
            color = colors[i % len(colors)]
            hw = int(key_w * 0.32)  # half-width of the note block
            self.create_rectangle(
                x_center - hw, y - note_h,
                x_center + hw, y,
                fill=color, outline='#f5f7fb',
            )
            self.create_line(x_center - int(key_w * 0.25), y - note_h + 4, x_center + int(key_w * 0.25), y - note_h + 4, fill='#ffffff')

        for i in range(key_count):
            x1 = int(i * key_w)
            x2 = int((i + 1) * key_w)
            fill = '#edf4f7' if i % 2 == 0 else '#dce8ef'
            self.create_rectangle(x1, key_top, x2, key_bottom, fill=fill, outline='#6f8092')
            self.create_line(x1 + 2, key_top + 3, x2 - 2, key_top + 3, fill='#ffffff')

        black_offsets = {0, 1, 3, 4, 5}
        for i in range(key_count - 1):
            if i % 7 not in black_offsets:
                continue
            cx = int((i + 1) * key_w)
            bw = max(9, int(key_w * 0.56))
            bh = key_top + (key_bottom - key_top) * 58 // 100  # 58% down
            self.create_rectangle(
                cx - bw // 2, key_top,
                cx + bw // 2, bh,
                fill='#0a0e15', outline='#354153',
            )


class HoverTooltip:
    def __init__(self, root, text, delay_ms=450, wraplength=320):
        self.root = root
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.after_id = None
        self.window = None
        self.anchor = None

    def attach(self, widget):
        self._bind(widget)
        for child in widget.winfo_children():
            self.attach(child)

    def _bind(self, widget):
        try:
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")
        except TypeError:
            widget.bind("<Enter>", self._schedule)
            widget.bind("<Leave>", self._hide)
            widget.bind("<ButtonPress>", self._hide)
        except (tk.TclError, NotImplementedError):
            pass

    def _schedule(self, event):
        self.anchor = event.widget
        self._cancel()
        self.after_id = self.root.after(self.delay_ms, self._show)

    def _cancel(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _show(self):
        self.after_id = None
        self._destroy_window()
        if self.anchor is None:
            return
        try:
            x = self.anchor.winfo_rootx() + 12
            y = self.anchor.winfo_rooty() + self.anchor.winfo_height() + 8
        except tk.TclError:
            return
        self.window = tk.Toplevel(self.root)
        self.window.wm_overrideredirect(True)
        self.window.wm_attributes("-topmost", True)
        self.window.geometry(f"+{x}+{y}")
        label = tk.Label(
            self.window,
            text=self.text,
            justify="left",
            wraplength=self.wraplength,
            bg=PANEL,
            fg=TEXT,
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=7,
            font=("Segoe UI", 9),
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        self._destroy_window()

    def _destroy_window(self):
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            self.window = None


class StrictComboBox(ctk.CTkComboBox):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("state", "readonly")
        super().__init__(*args, **kwargs)
        self._dropdown_is_open = False
        try:
            self._dropdown_menu.bind("<Unmap>", self._mark_dropdown_closed, add="+")
        except tk.TclError:
            pass

    def _mark_dropdown_closed(self, _event=None):
        self._dropdown_is_open = False

    def _clicked(self, event=None):
        if self._state == tk.DISABLED or len(self._values) == 0:
            return
        try:
            menu_is_mapped = bool(self._dropdown_menu.winfo_ismapped())
        except tk.TclError:
            menu_is_mapped = False
        if self._dropdown_is_open or menu_is_mapped:
            self._dropdown_menu.unpost()
            self._dropdown_is_open = False
            return
        self._dropdown_is_open = True
        self._open_dropdown_menu()

    def _dropdown_callback(self, value):
        self._dropdown_is_open = False
        super()._dropdown_callback(value)

    def configure(self, require_redraw=False, **kwargs):
        values = kwargs.get("values")
        super().configure(require_redraw=require_redraw, **kwargs)
        if values is not None and values and self.get() not in values:
            self.set(values[0])


def camera_backends(include_fallback=False):
    if os.name == "nt":
        backends = [cv2.CAP_MSMF]
        if include_fallback:
            backends.append(cv2.CAP_DSHOW)
        return backends
    return [cv2.CAP_ANY]


def open_camera(index, include_fallback=False):
    for backend in camera_backends(include_fallback=include_fallback):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def detect_cameras(max_check=CAMERA_DETECT_LIMIT):
    available = []
    for i in range(max_check):
        cap = open_camera(i, include_fallback=False)
        if cap is not None:
            available.append(str(i))
            cap.release()
    return available or ["0"]


def normalize_note_name(raw):
    note = raw.strip().upper()
    if len(note) >= 3 and note[1] == 'S':
        return f"{note[0]}#{note[2:]}"
    if len(note) >= 3 and note[:2] in FLAT_TO_SHARP:
        return f'{FLAT_TO_SHARP[note[:2]]}{note[2:]}'
    return note


def clamp_int(value, min_value, max_value):
    return max(min_value, min(max_value, int(value)))


def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if not SETTINGS_FILE.exists():
        return settings
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return settings
    for key in settings:
        if key in saved:
            settings[key] = saved[key]
    if settings["instrument"] not in INSTRUMENTS:
        settings["instrument"] = DEFAULT_SETTINGS["instrument"]
    if settings["mode"] not in MODE_OPTIONS:
        settings["mode"] = DEFAULT_SETTINGS["mode"]
    if settings["resolution"] not in RESOLUTION_OPTIONS:
        settings["resolution"] = DEFAULT_SETTINGS["resolution"]
    if str(settings["piano_octaves"]) not in PIANO_OCTAVE_OPTIONS:
        settings["piano_octaves"] = DEFAULT_SETTINGS["piano_octaves"]
    if str(settings["max_hands"]) not in MAX_HAND_OPTIONS:
        settings["max_hands"] = DEFAULT_SETTINGS["max_hands"]
    if settings["keybed_preset"] not in KEYBED_PRESETS:
        settings["keybed_preset"] = DEFAULT_SETTINGS["keybed_preset"]
    if settings["play_trigger"] not in PLAY_TRIGGER_ARGS:
        settings["play_trigger"] = DEFAULT_SETTINGS["play_trigger"]
    return settings


class PianoLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("InstrumentalCV Launcher")
        self.geometry("1060x820")
        self.minsize(940, 720)
        self.configure(fg_color=BG)

        self.process = None
        self.setup_process = None
        self.sample_switches = {}
        self.preview_cap = None
        self.preview_running = False
        self.preview_image = None  # holds PIL image so it doesn't get GC'd
        self.preview_photo = None  # same for tkinter PhotoImage
        self.preview_after_id = None
        self.preview_canvas_image_id = None
        self.preview_canvas_text_id = None
        self.summary_labels = {}
        self.summary_refresh_after_id = None
        self.import_buttons = []
        self.tooltips = []
        self.learning_fit_mode = "retune"
        self._rush_e_lock_guard = False  # avoids recursive trace when forcing autoplay
        self.cameras = detect_cameras()
        self.settings = load_settings()
        if self.settings["camera"] not in self.cameras:
            self.settings["camera"] = self.cameras[0]
        self.song_choices = self._load_song_choices()
        if self.song_choices:
            if self.settings["learning_song"] not in self.song_choices:
                self.settings["learning_song"] = next(iter(self.song_choices))
            if self.settings["editor_song"] not in self.song_choices:
                self.settings["editor_song"] = self.settings["learning_song"]
        else:
            self.settings["learning_enabled"] = False
            self.settings["learning_song"] = ""
            self.settings["editor_song"] = ""

        self.slider_labels = {}
        self.vars = self._create_vars()
        for variable in self.vars.values():
            variable.trace_add("write", self._schedule_summary_refresh)
        for key in ("learning_enabled", "learning_song", "learning_autoplay", "performance_autoplay", "piano_octaves", "octave"):
            self.vars[key].trace_add("write", self._sync_learning_warning)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_footer()
        self.update_labels()
        self._sync_sample_selection_status()
        self._sync_learning_state()
        self._sync_mode_state()
        self._refresh_summary()
        self._set_status(self._readiness_text(), SUCCESS)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_song_choices(self):
        choices = {}
        if not SONGS_DIR.is_dir():
            return choices
        used_labels = set()
        for path in sorted(SONGS_DIR.glob("*.json")):
            label = path.stem.replace("_", " ").title()
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                label = data.get("title") or label
            except (OSError, json.JSONDecodeError):
                pass
            if label in used_labels:
                label = f"{label} ({path.name})"
            used_labels.add(label)
            choices[label] = path
        return choices

    def _create_vars(self):
        settings = self.settings
        return {
            "mode": ctk.StringVar(value=settings["mode"]),
            "camera": ctk.StringVar(value=str(settings["camera"])),
            "instrument": ctk.StringVar(value=settings["instrument"]),
            "octave": ctk.IntVar(value=int(settings["octave"])),
            "piano_octaves": ctk.StringVar(value=str(settings["piano_octaves"])),
            "volume": ctk.IntVar(value=int(settings["volume"])),
            "resolution": ctk.StringVar(value=settings["resolution"]),
            "tracking_scale": ctk.IntVar(value=int(settings["tracking_scale"])),
            "max_hands": ctk.StringVar(value=str(settings["max_hands"])),
            "detection_confidence": ctk.IntVar(value=int(settings["detection_confidence"])),
            "tracking_confidence": ctk.IntVar(value=int(settings["tracking_confidence"])),
            "mirror": ctk.BooleanVar(value=bool(settings["mirror"])),
            "show_landmarks": ctk.BooleanVar(value=bool(settings["show_landmarks"])),
            "show_note_labels": ctk.BooleanVar(value=bool(settings["show_note_labels"])),
            "show_fps": ctk.BooleanVar(value=bool(settings["show_fps"])),
            "show_note_trail": ctk.BooleanVar(value=bool(settings["show_note_trail"])),
            "piano_top": ctk.IntVar(value=int(settings["piano_top"])),
            "piano_bottom": ctk.IntVar(value=int(settings["piano_bottom"])),
            "piano_opacity": ctk.IntVar(value=int(settings["piano_opacity"])),
            "play_trigger": ctk.StringVar(value=settings["play_trigger"]),
            "trigger_cooldown_ms": ctk.IntVar(value=int(settings["trigger_cooldown_ms"])),
            "smoothing": ctk.IntVar(value=int(settings["smoothing"])),
            "dead_zone": ctk.IntVar(value=int(settings["dead_zone"])),
            "fadeout_ms": ctk.IntVar(value=int(settings["fadeout_ms"])),
            "metronome": ctk.BooleanVar(value=bool(settings["metronome"])),
            "metronome_bpm": ctk.IntVar(value=int(settings["metronome_bpm"])),
            "download_all_samples": ctk.BooleanVar(value=bool(settings["download_all_samples"])),
            "download_grand": ctk.BooleanVar(value=bool(settings["download_grand"])),
            "download_bright": ctk.BooleanVar(value=bool(settings["download_bright"])),
            "download_electronic": ctk.BooleanVar(value=bool(settings["download_electronic"])),
            "download_organ": ctk.BooleanVar(value=bool(settings["download_organ"])),
            "download_reverb": ctk.BooleanVar(value=bool(settings["download_reverb"])),
            "keybed_preset": ctk.StringVar(value=settings["keybed_preset"]),
            "learning_enabled": ctk.BooleanVar(value=bool(settings["learning_enabled"])),
            "learning_song": ctk.StringVar(value=settings["learning_song"]),
            "learning_autoplay": ctk.BooleanVar(value=bool(settings["learning_autoplay"])),
            "performance_autoplay": ctk.BooleanVar(value=bool(settings["performance_autoplay"])),
            "editor_song": ctk.StringVar(value=settings["editor_song"]),

        }

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=18, border_width=1, border_color=BORDER_HOT)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        copy = ctk.CTkFrame(header, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="nsew", padx=(22, 12), pady=18)
        copy.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            copy,
            text="InstrumentalCV",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            copy,
            text="Use your webcam to play piano with your hands. No MIDI needed.",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_MUTED,
            anchor="w",
            wraplength=560,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 12))

        chips = ctk.CTkFrame(copy, fg_color="transparent")
        chips.grid(row=2, column=0, sticky="w")
        self.mode_pill = self._header_pill(chips, "AIR PIANO", ACCENT, 0)
        self.header_camera_pill = self._header_pill(chips, "CAM 0", PANEL_ALT, 1)
        self.header_samples_pill = self._header_pill(chips, "SAMPLES", PANEL_ALT, 2)
        self.header_song_pill = self._header_pill(chips, "FREE PLAY", PANEL_ALT, 3)

        visual = HeroVisualizer(header, width=360, height=138)
        visual.grid(row=0, column=1, sticky="e", padx=(8, 18), pady=18)

    def _header_pill(self, parent, text, color, column):
        pill = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT,
            fg_color=color,
            corner_radius=999,
            height=30,
            width=max(92, len(text) * 8 + 22),
        )
        pill.grid(row=0, column=column, sticky="w", padx=(0, 8))
        return pill

    def _build_body(self):
        body = ctk.CTkTabview(
            self,
            fg_color="transparent",
            segmented_button_fg_color=PANEL,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=PANEL_ALT,
            segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT,
            corner_radius=14,
        )
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        self.tabs = body

        summary_tab = self._launcher_tab("Summary")
        play_tab = self._launcher_tab("Play")
        camera_tab = self._launcher_tab("Camera")
        learning_tab = self._launcher_tab("Learning")
        scanner_tab = self._launcher_tab("Sheet Scanner")
        editor_tab = self._launcher_tab("Song Editor")
        settings_tab = self._launcher_tab("Settings")

        summary_tab.grid_columnconfigure(0, weight=1)
        self._build_summary_tab(summary_tab)

        play_tab.grid_columnconfigure((0, 1), weight=1, uniform="play_columns")
        play_left = ctk.CTkFrame(play_tab, fg_color="transparent")
        play_right = ctk.CTkFrame(play_tab, fg_color="transparent")
        play_left.grid(row=0, column=0, sticky="new", padx=(0, 8), pady=8)
        play_right.grid(row=0, column=1, sticky="new", padx=(8, 0), pady=8)
        play_left.grid_columnconfigure(0, weight=1)
        play_right.grid_columnconfigure(0, weight=1)
        self._build_mode_section(play_left, 0)
        self._build_sound_section(play_left, 1)
        self._build_piano_layout_section(play_right, 0)
        self._build_metronome_section(play_right, 1)

        camera_tab.grid_columnconfigure((0, 1), weight=1, uniform="camera_columns")
        camera_left = ctk.CTkFrame(camera_tab, fg_color="transparent")
        camera_right = ctk.CTkFrame(camera_tab, fg_color="transparent")
        camera_left.grid(row=0, column=0, sticky="new", padx=(0, 8), pady=8)
        camera_right.grid(row=0, column=1, sticky="new", padx=(8, 0), pady=8)
        camera_left.grid_columnconfigure(0, weight=1)
        camera_right.grid_columnconfigure(0, weight=1)
        self._build_camera_section(camera_left, 0)
        self._build_camera_preview_section(camera_right, 0)

        learning_tab.grid_columnconfigure(0, weight=1)
        self._build_learning_section(learning_tab, 0)

        scanner_tab.grid_columnconfigure(0, weight=1)
        self._build_sheet_scanner_section(scanner_tab, 0)

        editor_tab.grid_columnconfigure(0, weight=1)
        self._build_song_editor_section(editor_tab, 0)

        settings_tab.grid_columnconfigure((0, 1), weight=1, uniform="settings_columns")
        settings_left = ctk.CTkFrame(settings_tab, fg_color="transparent")
        settings_right = ctk.CTkFrame(settings_tab, fg_color="transparent")
        settings_left.grid(row=0, column=0, sticky="new", padx=(0, 8), pady=8)
        settings_right.grid(row=0, column=1, sticky="new", padx=(8, 0), pady=8)
        settings_left.grid_columnconfigure(0, weight=1)
        settings_right.grid_columnconfigure(0, weight=1)
        self._build_tracking_section(settings_left, 0)
        self._build_controls_section(settings_left, 1)
        self._build_readiness_section(settings_right, 0)
        self._build_sample_setup_section(settings_right, 1)

    def _launcher_tab(self, name):
        self.tabs.add(name)
        tab = self.tabs.tab(name)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        content = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        content.grid(row=0, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        return content

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER_HOT)
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        footer.grid_columnconfigure(0, weight=1)

        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=16, pady=14)

        self.reset_btn = ctk.CTkButton(
            footer,
            text="Reset",
            width=86,
            fg_color=PANEL_ALT,
            hover_color=BORDER_HOT,
            command=self._reset_defaults,
        )
        self.reset_btn.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=12)
        self._add_tooltip(self.reset_btn, "Restores every launcher setting to the project defaults.")

        self.stop_btn = ctk.CTkButton(
            footer,
            text="Stop",
            width=86,
            fg_color=DANGER,
            hover_color="#c0392b",  # slightly darker red on hover
            state="disabled",
            command=self._stop_session,
        )
        self.stop_btn.grid(row=0, column=2, sticky="e", padx=(0, 8), pady=12)
        self._add_tooltip(self.stop_btn, "Stops the currently running piano session.")

        self.launchButton = ctk.CTkButton(
            footer,
            text="Launch Air Piano",
            width=190,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            corner_radius=10,
            command=self._launch,
        )
        self.launchButton.grid(row=0, column=3, sticky="e", padx=(0, 12), pady=12)
        self._add_tooltip(self.launchButton, "Starts the piano session with the current launcher settings.")

    def _section(self, parent, title, row):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_WARM,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
        return frame

    def _tooltip_text(self, key_or_text):
        if not key_or_text:
            return ""
        return CONTROL_HELP.get(key_or_text, key_or_text)

    def _add_tooltip(self, widgets, key_or_text):
        text = self._tooltip_text(key_or_text)
        if not text:
            return None
        if not isinstance(widgets, (list, tuple, set)):
            widgets = [widgets]
        tooltip = HoverTooltip(self, text)
        self.tooltips.append(tooltip)
        for widget in widgets:
            if widget is not None:
                tooltip.attach(widget)
        return tooltip

    def _build_summary_tab(self, parent):
        parent.grid_columnconfigure((0, 1, 2), weight=1, uniform='summary_cards')

        overview = ctk.CTkFrame(parent, fg_color='#101b2a', corner_radius=16, border_width=1, border_color=BORDER_HOT)
        overview.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(8, 14))
        overview.grid_columnconfigure(0, weight=1)
        self.summary_title = ctk.CTkLabel(overview, text='Air Piano',
            font=ctk.CTkFont(size=22, weight='bold'), text_color=TEXT, anchor='w')
        self.summary_title.grid(row=0, column=0, sticky='ew', padx=18, pady=(16, 2))
        self.summary_subtitle = ctk.CTkLabel(overview, text='',
            font=ctk.CTkFont(size=13), text_color=TEXT_MUTED, anchor='w', justify='left')
        self.summary_subtitle.grid(row=1, column=0, sticky='ew', padx=18, pady=(0, 16))
        launch_button = ctk.CTkButton(overview, text='Launch Session', width=120, height=38,
            corner_radius=10, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._launch)
        launch_button.grid(row=0, column=1, rowspan=2, sticky='e', padx=18, pady=16)
        self._add_tooltip(launch_button, "Starts the piano session with the current launcher settings.")

        cards = [
            ('Session', ['Mode', 'Song', 'Instrument']),
            ('Camera', ['Camera', 'Resolution', 'Tracking Load', 'Trigger']),
            ('Enabled', ['Learning', 'Autoplay', 'Metronome']),
            ('Piano', ['Octaves', 'Keybed', 'Labels']),
            ('Performance', ['Landmarks', 'Trail', 'Samples']),
        ]
        border_colors = [BORDER_HOT, '#4d3d62', '#345943', '#5d4935', '#533949', BORDER]
        for i, (title, labels) in enumerate(cards, start=1):
            card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14,
                border_width=1, border_color=border_colors[(i - 1) % 6])
            card.grid(row=1 + (i - 1) // 3, column=(i - 1) % 3, sticky='nsew', padx=6, pady=6)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=title.upper(), font=ctk.CTkFont(size=11, weight='bold'),
                text_color=ACCENT_WARM if i % 2 else ACCENT_MINT, anchor='w',
            ).grid(row=0, column=0, sticky='ew', padx=14, pady=(12, 4))
            for j, lbl in enumerate(labels, start=1):
                v = ctk.CTkLabel(card, text='', font=ctk.CTkFont(size=13),
                    text_color=TEXT, anchor='w', justify='left', wraplength=240)
                v.grid(row=j, column=0, sticky='ew', padx=14, pady=3)
                self.summary_labels[lbl] = v
            ctk.CTkFrame(card, height=8, fg_color='transparent').grid(row=len(labels) + 1, column=0)

        # not writing this out 4 times
        nav = ctk.CTkFrame(parent, fg_color='transparent')
        nav.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(12, 0))
        nav.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform='summary_nav')
        for col, (lbl, tab) in enumerate([('Camera', 'Camera'), ('Learning', 'Learning'),
                                           ('Sheet Scanner', 'Sheet Scanner'), ('Settings', 'Settings')]):
            nav_button = ctk.CTkButton(nav, text=lbl, fg_color=PANEL_ALT, hover_color=BORDER_HOT,
                corner_radius=10, command=lambda t=tab: self.tabs.set(t),
            )
            nav_button.grid(row=0, column=col, sticky='ew', padx=6)
            self._add_tooltip(nav_button, f"Opens the {tab} tab.")

    def _song_summary(self, label):
        path = self.song_choices.get(label)
        if path is None:
            return label or "No song selected"
        try:
            song = load_song(str(path))
        except Exception:
            return label
        mins = int(song.duration_seconds // 60)
        secs = int(song.duration_seconds % 60)
        return f"{song.title} ({len(song.notes)} notes, {mins}:{secs:02d})"

    def _selected_learning_song_is_rush_e(self, song=None):
        if not hasattr(self, "vars"):
            return False
        label = self.vars["learning_song"].get()
        path = self.song_choices.get(label)
        return (
            is_rush_e_name(label)
            or (path is not None and (is_rush_e_name(path.stem) or is_rush_e_name(path.name)))
            or (song is not None and is_rush_e_name(song.title))
        )

    def _rush_e_autoplay_locked(self, song=None):
        # rush e must autoplay, no exceptions
        return bool(self.vars["learning_enabled"].get()) and self._selected_learning_song_is_rush_e(song)

    def _apply_rush_e_autoplay_lock(self, song=None):
        locked = self._rush_e_autoplay_locked(song)
        if locked and not self.vars["learning_autoplay"].get() and not self._rush_e_lock_guard:
            self._rush_e_lock_guard = True
            try:
                self.vars["learning_autoplay"].set(True)
            finally:
                self._rush_e_lock_guard = False
        return locked

    def _cheeky_line(self, rush_e_locked=False):
        seed_text = "|".join(
            [
                self.vars["learning_song"].get() if hasattr(self, "vars") else "",
                self.vars["instrument"].get() if hasattr(self, "vars") else "",
                self.vars["mode"].get() if hasattr(self, "vars") else "",
            ]
        )
        seed = sum(ord(char) for char in seed_text)
        return LAUNCHER_ROASTS[seed % len(LAUNCHER_ROASTS)]

    def _sync_mischief_label(self, rush_e_locked=None, song=None):
        if not hasattr(self, "learning_mischief_label"):
            return
        if rush_e_locked is None:
            rush_e_locked = self._rush_e_autoplay_locked(song)
        prefix = "Autopilot" if rush_e_locked else "{label}"
        self.learning_mischief_label.configure(text=f"{prefix}: {self._cheeky_line(rush_e_locked)}")

    def _onoff(self, enabled, on_text="On", off_text="Off"):
        return on_text if enabled else off_text

    def _schedule_summary_refresh(self, *_args):
        if not hasattr(self, "summary_labels") or not self.summary_labels:
            return
        if self.summary_refresh_after_id is not None:
            return
        self.summary_refresh_after_id = self.after_idle(self._refresh_summary)

    def _refresh_summary(self):
        self.summary_refresh_after_id = None
        if not self.summary_labels:
            return
        learning_on = bool(self.vars["learning_enabled"].get())
        rush_e_locked = self._apply_rush_e_autoplay_lock() if learning_on else False  # rush e forces autoplay on, no escape
        song_label = self.vars["learning_song"].get() if learning_on else ""
        song_text = self._song_summary(song_label) if learning_on else "Free play"
        mode_text = "Learning Mode" if learning_on else self.vars["mode"].get()
        instrument = self.vars["instrument"].get()
        resolution = self.vars["resolution"].get()
        top = int(float(self.vars["piano_top"].get()))
        bottom = int(float(self.vars["piano_bottom"].get()))
        octaves = int(float(self.vars["piano_octaves"].get()))
        start_octave = int(float(self.vars["octave"].get()))
        missing_samples = [
            label for label, folder in INSTRUMENTS.items()
            if not list((APP_DIR / folder).glob("*.wav"))
        ]

        summary_title = f"{mode_text} with {instrument}"
        if rush_e_locked:
            summary_title = f"Rush E Autopilot with {instrument}"
        self.summary_title.configure(text=summary_title)
        summary_subtitle = f"Camera {self.vars['camera'].get()} at {resolution}. Playing: {song_text}."
        if rush_e_locked:
            summary_subtitle = f"Rush E selected. Autoplay is locked on; your fingers lost custody."
        self.summary_subtitle.configure(
            text=summary_subtitle
        )
        if hasattr(self, "header_camera_pill"):
            self.header_camera_pill.configure(text=f"CAM {self.vars['camera'].get()}", width=92)
        if hasattr(self, "header_samples_pill"):
            ready_count = len(INSTRUMENTS) - len(missing_samples)
            self.header_samples_pill.configure(
                text=f"SAMPLES {ready_count}/{len(INSTRUMENTS)}",
                fg_color=ACCENT_MINT if not missing_samples else PANEL_ALT,
                text_color="#07100b" if not missing_samples else TEXT,
                width=118,
            )
        if hasattr(self, "header_song_pill"):
            header_song = "FREE PLAY" if not learning_on else song_text.split(" (", 1)[0][:20].upper()
            self.header_song_pill.configure(text=header_song, width=max(104, len(header_song) * 8 + 20))
        values = {
            "Mode": f"Mode: {mode_text}",
            "Song": f"Song: {song_text}",
            "Instrument": f"Sound: {instrument}, {int(float(self.vars['volume'].get()))}% volume",
            "Camera": f"Camera: {self.vars['camera'].get()}",
            "Resolution": f"Resolution: {resolution}",
            "Tracking Load": f"Tracking load: {int(float(self.vars['tracking_scale'].get()))}% frame",
            "Trigger": f"Trigger: {self.vars['play_trigger'].get()}",
            "Learning": f"Learning: {self._onoff(learning_on)}",
            "Autoplay": (
                "Autoplay: Rush E locked"
                if rush_e_locked
                else
                "Autoplay: performance"
                if self.vars["learning_autoplay"].get() and self.vars["performance_autoplay"].get()
                else f"Autoplay: {self._onoff(self.vars['learning_autoplay'].get())}"
            ),
            "Metronome": f"Metronome: {self._onoff(self.vars['metronome'].get(), str(int(float(self.vars['metronome_bpm'].get()))) + ' BPM', 'Off')}",
            "Octaves": f"Keys: C{start_octave}-B{start_octave + octaves - 1}",
            "Keybed": f"Keybed: {top}% to {bottom}% vertical",
            "Labels": f"Labels: {self._onoff(self.vars['show_note_labels'].get())}",
            "Landmarks": f"Hand overlay: {self._onoff(self.vars['show_landmarks'].get())}",
            "Trail": f"Note trail: {self._onoff(self.vars['show_note_trail'].get())}",
            "Samples": "Samples: ready" if not missing_samples else f"Samples missing: {', '.join(missing_samples[:2])}",
        }
        for key, value in values.items():
            if key in self.summary_labels:
                self.summary_labels[key].configure(text=value)

    def _row(self, parent, row, label, tooltip=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=5)
        frame.grid_columnconfigure(1, weight=1)
        label_widget = ctk.CTkLabel(frame, text=label, text_color=TEXT_MUTED, font=ctk.CTkFont(size=13), width=150, anchor="w")
        label_widget.grid(row=0, column=0, sticky="w")
        if tooltip:
            self._add_tooltip([frame, label_widget], tooltip)
        return frame

    def _combo(self, parent, row, label, variable, values, command=None, tooltip=None, width=210):
        frame = self._row(parent, row, label, tooltip)
        combo = StrictComboBox(
            frame,
            values=values,
            variable=variable,
            command=command,
            fg_color=PANEL_ALT,
            border_color=BORDER_HOT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            width=width,
            corner_radius=9,
        )
        combo.grid(row=0, column=1, sticky="e")
        if tooltip:
            self._add_tooltip(combo, tooltip)
        return combo

    def _slider(self, parent, row, label, variable, from_, to, steps, key, suffix="", formatter=None, command=None, tooltip=None):
        frame = self._row(parent, row, label, tooltip or key)
        value_label = ctk.CTkLabel(frame, text="", text_color=TEXT_MUTED, width=58, anchor="e")
        value_label.grid(row=0, column=2, sticky="e", padx=(8, 0))
        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            number_of_steps=steps,
            variable=variable,
            fg_color=PANEL_SOFT,
            progress_color=ACCENT,
            button_color=ACCENT_WARM,
            button_hover_color=WARNING,
            command=lambda _value: self._refresh_slider(key, command),
        )
        slider.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.slider_labels[key] = (value_label, variable, suffix, formatter)
        self._add_tooltip([slider, value_label], tooltip or key)
        return slider

    def _switch(self, parent, row, label, variable, command=None, tooltip=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=5)
        switch = ctk.CTkSwitch(
            frame,
            text=label,
            variable=variable,
            command=command,
            progress_color=ACCENT_MINT,
            button_color=TEXT,
            button_hover_color=ACCENT_WARM,
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        switch.pack(anchor="w")
        self._add_tooltip([frame, switch], tooltip or label)
        return switch

    def _build_mode_section(self, parent, row):
        section = self._section(parent, "Mode", row)
        # TODO: desk mode goes here eventually
        mode = ctk.CTkSegmentedButton(
            section,
            values=MODE_OPTIONS,
            variable=self.vars["mode"],
            command=lambda _v: self._sync_mode_state(),
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_color=PANEL_ALT,
            unselected_hover_color=BORDER,
        )
        mode.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.mode_control = mode
        self._add_tooltip(mode, "mode")

        ctk.CTkLabel(
            section,
            text="Point the camera at your keyboard and play. Works on any flat surface.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

    def _build_camera_section(self, parent, row):
        section = self._section(parent, "Camera", row)
        camera_row = self._row(section, 1, "Camera device", "camera")
        self.camera_combo = StrictComboBox(
            camera_row,
            values=self.cameras,
            variable=self.vars["camera"],
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            width=116,
        )
        self.camera_combo.grid(row=0, column=1, sticky="e")
        self._add_tooltip(self.camera_combo, "camera")
        refresh_button = ctk.CTkButton(
            camera_row,
            text="Refresh",
            width=86,
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=self._refresh_cameras,
        )
        refresh_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self._add_tooltip(refresh_button, "Refreshes the list of detected camera devices.")
        self._combo(section, 2, "Resolution", self.vars["resolution"], RESOLUTION_OPTIONS, tooltip="resolution")
        self._switch(section, 3, "Mirror camera preview", self.vars["mirror"], tooltip="mirror")
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=4, column=0)

    def _build_camera_preview_section(self, parent, row):
        section = self._section(parent, "Camera Preview", row)

        self.preview_canvas = tk.Canvas(
            section,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            bg=PANEL_SOFT,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.preview_canvas.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        self._draw_preview_placeholder("Preview stopped")

        controls = ctk.CTkFrame(section, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        controls.grid_columnconfigure((0, 1), weight=1)

        self.previewBtn = ctk.CTkButton(
            controls,
            text="Start Preview",
            fg_color=PANEL_ALT,
            hover_color=BORDER_HOT,
            corner_radius=10,
            command=self._toggle_preview,
        )
        self.previewBtn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._add_tooltip(self.previewBtn, "preview")

        test_button = ctk.CTkButton(
            controls,
            text="Test Camera",
            fg_color=PANEL_ALT,
            hover_color=BORDER_HOT,
            corner_radius=10,
            command=self._test_camera,
        )
        test_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._add_tooltip(test_button, "test_camera")



    def _build_sound_section(self, parent, row):
        section = self._section(parent, "Sound", row)

        # instrument picker — done manually so I can store the ref
        inst_row = ctk.CTkFrame(section, fg_color="transparent")
        inst_row.grid(row=1, column=0, sticky="ew", padx=16, pady=5)
        inst_row.grid_columnconfigure(1, weight=1)
        inst_label = ctk.CTkLabel(inst_row, text="Instrument", text_color=TEXT_MUTED, font=ctk.CTkFont(size=13), width=150, anchor="w")
        inst_label.grid(row=0, column=0, sticky="w")
        self.instrument_menu = StrictComboBox(
            inst_row,
            values=list(INSTRUMENTS.keys()),
            variable=self.vars["instrument"],
            fg_color=PANEL_ALT,
            border_color=BORDER_HOT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            width=210,
            corner_radius=9,
        )
        self.instrument_menu.grid(row=0, column=1, sticky="e")
        self._add_tooltip([inst_row, inst_label, self.instrument_menu], "instrument")

        self._slider(section, 2, "Base octave", self.vars["octave"], 1, 6, 5, "octave", formatter=lambda v: f"C{v}")
        self._combo(section, 3, "Octaves shown", self.vars["piano_octaves"], PIANO_OCTAVE_OPTIONS, command=lambda _v: self._normalize_octave(), tooltip="piano_octaves")
        self._slider(section, 4, "Volume", self.vars["volume"], 0, 100, 100, "volume", "%")
        self._slider(section, 5, "Note fadeout", self.vars["fadeout_ms"], 30, 1200, 117, "fadeout_ms", " ms")
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=6, column=0)

    def _build_tracking_section(self, parent, row):
        section = self._section(parent, "Tracking", row)
        self._combo(section, 1, "Play trigger", self.vars["play_trigger"], PLAY_TRIGGER_OPTIONS, tooltip="play_trigger")
        self._combo(section, 2, "Hands tracked", self.vars["max_hands"], MAX_HAND_OPTIONS, tooltip="max_hands")
        self._slider(section, 3, "Detection", self.vars["detection_confidence"], 0, 100, 100, "detection_confidence", "%")
        self._slider(section, 4, "Tracking", self.vars["tracking_confidence"], 0, 100, 100, "tracking_confidence", "%")
        self._slider(section, 5, "Tracking load", self.vars["tracking_scale"], 0, 100, 100, "tracking_scale", "%")
        self._slider(section, 6, "Smoothing", self.vars["smoothing"], 1, 10, 9, "smoothing")
        self._slider(section, 7, "Trigger cooldown", self.vars["trigger_cooldown_ms"], 50, 500, 45, "trigger_cooldown_ms", " ms")
        self._slider(section, 8, "Key edge guard", self.vars["dead_zone"], 0, 16, 16, "dead_zone", " px")

        # these two I just slap in as switches rather than going through the helper
        show_lm = ctk.CTkFrame(section, fg_color="transparent")
        show_lm.grid(row=9, column=0, sticky="ew", padx=16, pady=5)
        self.landmarks_switch = ctk.CTkSwitch(
            show_lm, text="Show hand landmarks",
            variable=self.vars["show_landmarks"],
            progress_color=ACCENT_MINT, button_color=TEXT,
            button_hover_color=ACCENT_WARM, text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        self.landmarks_switch.pack(anchor="w")
        self._add_tooltip([show_lm, self.landmarks_switch], "show_landmarks")

        show_fps_frame = ctk.CTkFrame(section, fg_color="transparent")
        show_fps_frame.grid(row=10, column=0, sticky="ew", padx=16, pady=5)
        fps_switch = ctk.CTkSwitch(
            show_fps_frame, text="Show FPS",
            variable=self.vars["show_fps"],
            progress_color=ACCENT_MINT, button_color=TEXT,
            button_hover_color=ACCENT_WARM, text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        fps_switch.pack(anchor="w")
        self._add_tooltip([show_fps_frame, fps_switch], "show_fps")

        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=11, column=0)

    def _build_piano_layout_section(self, parent, row):
        section = self._section(parent, "Piano Layout", row)
        self._combo(section, 1, "Preset", self.vars["keybed_preset"], list(KEYBED_PRESETS.keys()), command=self._apply_keybed_preset, tooltip="keybed_preset")
        self._slider(section, 2, "Keybed top", self.vars["piano_top"], 10, 70, 60, "piano_top", "%", command=lambda: self._mark_keybed_custom(sync=True))
        self._slider(section, 3, "Keybed bottom", self.vars["piano_bottom"], 30, 92, 62, "piano_bottom", "%", command=lambda: self._mark_keybed_custom(sync=True))
        self._slider(section, 4, "Key opacity", self.vars["piano_opacity"], 25, 95, 70, "piano_opacity", "%", command=self._mark_keybed_custom)
        self._switch(section, 5, "Show note labels", self.vars["show_note_labels"], tooltip="show_note_labels")
        self._switch(section, 6, "Show note trail", self.vars["show_note_trail"], tooltip="show_note_trail")
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=6, column=0)

    def _build_learning_section(self, parent, row):
        section = self._section(parent, "Learning Mode", row)
        self.learning_switch = self._switch(
            section,
            1,
            "Start with falling-note practice",
            self.vars["learning_enabled"],
            command=self._on_learning_change,
            tooltip="learning_enabled",
        )

        song_row = self._row(section, 2, "Song", "learning_song")
        song_values = list(self.song_choices) or ["No songs found"]
        self.learning_song_combo = StrictComboBox(
            song_row,
            values=song_values,
            variable=self.vars["learning_song"],
            command=lambda _value: self._on_learning_change(),
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            width=210,
        )
        self.learning_song_combo.grid(row=0, column=1, sticky="e")
        self._add_tooltip(self.learning_song_combo, "learning_song")
        refresh_button = ctk.CTkButton(
            song_row,
            text="Refresh",
            width=76,
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=self._refresh_songs,
        )
        refresh_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self._add_tooltip(refresh_button, "Refreshes the saved song list from the songs folder.")

        self.learning_autoplay_switch = self._switch(
            section,
            3,
            LEARNING_AUTOPLAY_LABEL,
            self.vars["learning_autoplay"],
            command=self._on_learning_change,
            tooltip="learning_autoplay",
        )

        self.performance_autoplay_switch = self._switch(
            section,
            4,
            "Autoplayer performance mode for full black MIDI (Reduces lag for autoplay on note heavy songs)",
            self.vars["performance_autoplay"],
            command=self._on_learning_change,
            tooltip="performance_autoplay",
        )

        self.learning_warning_label = ctk.CTkLabel(
            section,
            text="",
            text_color=WARNING,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
            wraplength=720,
        )
        self.learning_warning_label.grid(row=5, column=0, sticky="ew", padx=16, pady=(2, 4))

        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=6, column=0)
        self._sync_learning_state()

    def _build_song_editor_section(self, parent, row):
        section = self._section(parent, "Song Editor", row)
        song_row = self._row(section, 1, "Song", "editor_song")
        song_values = list(self.song_choices) or ["No songs found"]
        self.editor_song_combo = StrictComboBox(
            song_row,
            values=song_values,
            variable=self.vars["editor_song"],
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            width=210,
        )
        self.editor_song_combo.grid(row=0, column=1, sticky="e")
        self._add_tooltip(self.editor_song_combo, "editor_song")

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 4))
        actions.grid_columnconfigure((0, 1, 2), weight=1, uniform="editor_actions")
        new_button = ctk.CTkButton(
            actions,
            text="New Song",
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=lambda: self._open_song_editor(new_song=True),
        )
        new_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._add_tooltip(new_button, "Creates a new editable practice song.")
        edit_button = ctk.CTkButton(
            actions,
            text="Edit Song",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._open_song_editor,
        )
        edit_button.grid(row=0, column=1, sticky="ew", padx=6)
        self._add_tooltip(edit_button, "Opens the selected song for manual note editing.")
        refresh_button = ctk.CTkButton(
            actions,
            text="Refresh",
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=self._refresh_songs,
        )
        refresh_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))
        self._add_tooltip(refresh_button, "Refreshes the saved song list from the songs folder.")

        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=3, column=0)

    def _build_sheet_scanner_section(self, parent, row):
        section = self._section(parent, "Sheet Scanner", row)
        ctk.CTkLabel(
            section,
            text="Import MIDI or MusicXML directly, or throw sheet images/PDFs at the scanner and hope for the best.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
            wraplength=720,
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 8))
        actions.grid_columnconfigure((0, 1, 2), weight=1, uniform="scanner_actions")
        buttons = [
            ("Import MIDI", self._import_midi_song, "Imports a MIDI file and saves it as a playable learning song."),
            ("Import MusicXML", self._import_musicxml_song, "Imports a MusicXML score and saves it as a playable learning song."),
            ("Scan Sheets", self._import_sheet_song, "Runs sheet-image or PDF scanning and turns the result into a learning song."),
        ]
        for column, (label, command, tooltip) in enumerate(buttons):
            button = ctk.CTkButton(
                actions,
                text=label,
                fg_color=ACCENT if column == 0 else PANEL_ALT,
                hover_color=ACCENT_HOVER if column == 0 else BORDER,
                command=command,
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0 if column == 2 else 6))
            self._add_tooltip(button, tooltip)
            self.import_buttons.append(button)

        self.import_status_label = ctk.CTkLabel(
            section,
            text="MIDI imports are tuned for real-time play. Really dense black MIDI gets thinned automatically — the renderer has limits, your fingers have more.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left",
            wraplength=720,
        )
        self.import_status_label.grid(row=3, column=0, sticky="ew", padx=16, pady=(2, 14))

    def _build_metronome_section(self, parent, row):
        section = self._section(parent, "Metronome", row)

        sw_frame = ctk.CTkFrame(section, fg_color="transparent")
        sw_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=5)
        metronome_switch = ctk.CTkSwitch(
            sw_frame,
            text="Start metronome on launch",
            variable=self.vars["metronome"],
            progress_color=ACCENT_MINT,
            button_color=TEXT,
            button_hover_color=ACCENT_WARM,
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        metronome_switch.pack(anchor="w")
        self._add_tooltip([sw_frame, metronome_switch], "metronome")

        bpm_row = ctk.CTkFrame(section, fg_color="transparent")
        bpm_row.grid(row=2, column=0, sticky="ew", padx=16, pady=5)
        bpm_row.grid_columnconfigure(1, weight=1)
        bpm_label = ctk.CTkLabel(bpm_row, text="Tempo", text_color=TEXT_MUTED, font=ctk.CTkFont(size=13), width=150, anchor="w")
        bpm_label.grid(row=0, column=0, sticky="w")
        bpm_val = ctk.CTkLabel(bpm_row, text="", text_color=TEXT_MUTED, width=58, anchor="e")
        bpm_val.grid(row=0, column=2, sticky="e", padx=(8, 0))
        bpm_slider = ctk.CTkSlider(
            bpm_row,
            from_=40, to=220, number_of_steps=180,
            variable=self.vars["metronome_bpm"],
            fg_color=PANEL_SOFT,
            progress_color=ACCENT,
            button_color=ACCENT_WARM,
            button_hover_color=WARNING,
            command=lambda _v: self._refresh_slider("metronome_bpm"),
        )
        bpm_slider.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.slider_labels["metronome_bpm"] = (bpm_val, self.vars["metronome_bpm"], " BPM", None)
        self._add_tooltip([bpm_row, bpm_label, bpm_val, bpm_slider], "metronome_bpm")

        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=3, column=0)

    def _build_controls_section(self, parent, row):
        section = self._section(parent, "Runtime Controls", row)
        controls = [
            ("Q / Esc", "Quit the piano window"),
            ("P", "Toggle learning autoplay, unless Rush E steals the switch"),
            ("M", "Toggle metronome"),
            ("+ / -", "Adjust metronome tempo"),
            ("Left / Right", "Shift octave range"),
            ("1 - 5", "Switch instruments"),
        ]
        for i, (key, description) in enumerate(controls, start=1):
            line = ctk.CTkFrame(section, fg_color="transparent")
            line.grid(row=i, column=0, sticky="ew", padx=16, pady=4)
            line.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                line,
                text=key,
                font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                text_color=TEXT,
                fg_color=PANEL_ALT,
                corner_radius=6,
                width=90,
                height=26,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                line,
                text=description,
                text_color=TEXT_MUTED,
                font=ctk.CTkFont(size=12),
            ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=len(controls) + 1, column=0)

    def _build_readiness_section(self, parent, row):
        section = self._section(parent, "Readiness", row)
        self.samples_label = ctk.CTkLabel(section, text="", text_color=TEXT_MUTED, anchor="w", justify="left")
        self.samples_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.camera_label = ctk.CTkLabel(section, text="", text_color=TEXT_MUTED, anchor="w", justify="left")
        self.camera_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        self._refresh_readiness_labels()

    def _build_sample_setup_section(self, parent, row):
        section = self._section(parent, "Sample Setup", row)
        self.sample_switches["download_all_samples"] = self._switch(
            section,
            1,
            "Download all instruments",
            self.vars["download_all_samples"],
            command=self._on_download_all_toggle,
            tooltip="download_all_samples",
        )
        for i, (var_name, label, _arg_name) in enumerate(SAMPLE_OPTIONS, start=2):
            self.sample_switches[var_name] = self._switch(
                section,
                i,
                label,
                self.vars[var_name],
                command=self._on_sample_choice_toggle,
                tooltip=var_name,
            )

        self.sample_status_label = ctk.CTkLabel(section, text="", text_color=TEXT_MUTED, anchor="w", justify="left")
        self.sample_status_label.grid(row=7, column=0, sticky="ew", padx=16, pady=(6, 0))
        self.download_btn = ctk.CTkButton(
            section,
            text="Download Selected Samples",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._download_selected_samples,
        )
        self.download_btn.grid(row=8, column=0, sticky="ew", padx=16, pady=(10, 14))
        self._add_tooltip(self.download_btn, "download_samples")

    def _refresh_slider(self, key, extra=None):
        if extra:
            extra()
        label, variable, suffix, fmt = self.slider_labels[key]
        val = int(float(variable.get()))  # variable sometimes comes back as a float string
        # print(key, val)
        label.configure(text=fmt(val) if fmt else f"{val}{suffix}")

    def update_labels(self):
        for key in self.slider_labels:
            label, variable, suffix, fmt = self.slider_labels[key]
            val = int(float(variable.get()))
            label.configure(text=fmt(val) if fmt else f"{val}{suffix}")
        self._normalize_octave()
        self._sync_piano_bounds()

    def _on_download_all_toggle(self):
        all_selected = bool(self.vars["download_all_samples"].get())
        for var_name, _label, _arg_name in SAMPLE_OPTIONS:
            self.vars[var_name].set(all_selected)
        self._sync_sample_selection_status()

    def _on_sample_choice_toggle(self):
        all_selected = all(bool(self.vars[var_name].get()) for var_name, _label, _arg_name in SAMPLE_OPTIONS)
        self.vars["download_all_samples"].set(all_selected)
        self._sync_sample_selection_status()

    def _selected_sample_args(self):
        if self.vars["download_all_samples"].get():
            return ["all"]
        return [
            arg_name
            for var_name, _label, arg_name in SAMPLE_OPTIONS
            if self.vars[var_name].get()
        ]

    def _sample_args_label(self, selected):
        if selected == ["all"]:
            return "all instrument packs"
        labels_by_arg = {arg_name: label for _var_name, label, arg_name in SAMPLE_OPTIONS}
        return ", ".join(labels_by_arg.get(arg_name, arg_name) for arg_name in selected)

    def _sync_sample_selection_status(self):
        if not hasattr(self, "sample_status_label"):
            return

        if self.vars["download_all_samples"].get():
            for var_name, _label, _arg_name in SAMPLE_OPTIONS:
                self.vars[var_name].set(True)

        selected = self._selected_sample_args()
        if selected == ["all"]:
            text = "Selected: all instrument packs"
            color = TEXT_MUTED
        elif selected:
            labels = [label for var_name, label, _arg_name in SAMPLE_OPTIONS if self.vars[var_name].get()]
            text = "Selected: " + ", ".join(labels)
            color = TEXT_MUTED
        else:
            text = "Select at least one sample pack."
            color = WARNING

        if self.setup_process is not None and self.setup_process.poll() is None:
            text = "Downloading: " + self._sample_args_label(selected)
            color = WARNING

        self.sample_status_label.configure(text=text, text_color=color)

    def _set_setup_controls(self, is_running):
        if hasattr(self, "download_btn"):
            self.download_btn.configure(
                state="disabled" if is_running else "normal",
                text="Downloading Samples..." if is_running else "Download Selected Samples",
            )
        if hasattr(self, "launch_btn") and self.process is None and self.vars["mode"].get() == "Air Piano":
            self.launchButton.configure(state="disabled" if is_running else "normal", text=self._launch_button_text())

    def _apply_keybed_preset(self, preset_name=None):
        preset_name = preset_name or self.vars["keybed_preset"].get()
        preset = KEYBED_PRESETS.get(preset_name)

        if preset is None:
            return
        self.vars["piano_top"].set(preset["piano_top"])
        self.vars["piano_bottom"].set(preset["piano_bottom"])
        self.vars["piano_opacity"].set(preset["piano_opacity"])
        self._sync_piano_bounds()
        self._refresh_slider("piano_opacity")
        self._set_status(f"Applied {preset_name} keybed preset.", SUCCESS)

    def _mark_keybed_custom(self, sync=False):
        self.vars["keybed_preset"].set("Custom")
        if sync:
            self._sync_piano_bounds()


    def _sync_piano_bounds(self):
        top = int(float(self.vars["piano_top"].get()))
        bottom = int(float(self.vars["piano_bottom"].get()))
        # idk why but lower than this makes the keybed render weird on some machines
        if bottom - top < 12:
            if top <= 80:
                self.vars["piano_bottom"].set(min(92, top + 12))
            else:
                self.vars["piano_top"].set(max(10, bottom - 12))
        for key in ("piano_top", "piano_bottom"):
            if key in self.slider_labels:
                self._refresh_slider(key)

    def _normalize_octave(self):
        octaves = int(self.vars["piano_octaves"].get())
        max_octave = 8 - octaves
        octave = int(float(self.vars["octave"].get()))
        if octave > max_octave:
            self.vars["octave"].set(max_octave)
        if "octave" in self.slider_labels:
            self._refresh_slider("octave")

    def _sync_mode_state(self):
        mode = self.vars["mode"].get()
        if mode == "Desk Mode":
            self.mode_pill.configure(text="DESK MODE", fg_color=WARNING, text_color="#111111", width=104)
            self.launchButton.configure(text="Desk Mode Coming Soon", state="disabled")
            self._set_status("Desk Mode is planned, Coming Soon.", WARNING)
        else:
            self.mode_pill.configure(text="AIR PIANO", fg_color=ACCENT, text_color=TEXT, width=104)
            setup_running = self.setup_process is not None and self.setup_process.poll() is None
            if self.process is None:
                self.launchButton.configure(text=self._launch_button_text(), state="disabled" if setup_running else "normal")

    def _launch_button_text(self):
        if self._rush_e_autoplay_locked():
            return "Launch Rush E Autopilot"
        if self.vars["learning_enabled"].get() and self.vars["learning_autoplay"].get() and self.vars["performance_autoplay"].get():
            return "Launch Autoplayer"
        return "Launch Learning Mode" if self.vars["learning_enabled"].get() else "Launch Air Piano"

    def _on_learning_change(self):
        self._sync_learning_state()
        if self._rush_e_autoplay_locked():
            self._set_status(RUSH_E_LOCK_MESSAGE, WARNING)
        # update the button text directly here too, don't want to go through sync_mode_state
        # because that resets other stuff
        if self.vars["mode"].get() == "Air Piano" and self.process is None:
            setup_running = self.setup_process is not None and self.setup_process.poll() is None
            txt = self._launch_button_text()
            self.launchButton.configure(text=txt, state="disabled" if setup_running else "normal")

    def _selected_learning_song(self):
        path = self.song_choices.get(self.vars["learning_song"].get())
        if path is None:
            return None
        try:
            return load_song(str(path))
        except Exception:
            return None

    def _learning_range_status(self, song=None):
        if song is None:
            song = self._selected_learning_song()
        if song is None or not song.notes:
            return None

        shown_octaves = int(self.vars["piano_octaves"].get())
        start_octave = int(float(self.vars["octave"].get()))
        display_min = note_to_midi(f"C{start_octave}")
        display_max = note_to_midi(f"B{start_octave + shown_octaves - 1}")
        song_min, song_max = song.note_range()
        song_low_octave = (song_min // 12) - 1
        song_high_octave = (song_max // 12) - 1
        needed_octaves = song_high_octave - song_low_octave + 1
        too_wide = needed_octaves > shown_octaves
        outside_range = song_min < display_min or song_max > display_max
        if not too_wide and not outside_range:
            return None
        # clamp suggested to what the piano actually supports
        suggested_octaves = clamp_int(max(shown_octaves, needed_octaves), 1, MAX_PIANO_OCTAVES)
        suggested_start = clamp_int(song_low_octave, 1, 8 - suggested_octaves)
        return {
            "song": song,
            "song_low_octave": song_low_octave,
            "song_high_octave": song_high_octave,
            "needed_octaves": needed_octaves,
            "shown_octaves": shown_octaves,
            "start_octave": start_octave,
            "suggested_octaves": suggested_octaves,
            "suggested_start": suggested_start,
            "too_wide": too_wide,
            "outside_range": outside_range,
        }

    def _sync_learning_warning(self, *_args):
        # this fires too early on startup before the vars exist, hence the hasattr
        if not hasattr(self, "learning_warning_label"):
            return
        if not self.vars["learning_enabled"].get():
            self.learning_warning_label.configure(text="")
            self._sync_mischief_label(False)
            return
        song = self._selected_learning_song()
        rush_e_locked = self._apply_rush_e_autoplay_lock(song)
        status = self._learning_range_status(song)
        messages = []
        if rush_e_locked:
            messages.append(RUSH_E_LOCK_MESSAGE)
        if status is not None:
            messages.append(
                f"Range warning: {status['song'].title} uses "
                f"C{status['song_low_octave']}-B{status['song_high_octave']} "
                f"({status['needed_octaves']} octaves), while the piano shows "
                f"{status['shown_octaves']} octave(s). Launch will ask how to handle it."
            )
        elif (
            song is not None
            and self.vars["learning_autoplay"].get()
            and len(song.notes) >= PERFORMANCE_AUTOPLAY_NOTE_THRESHOLD
            and not self.vars["performance_autoplay"].get()
        ):
            messages.append(
                f"{song.title} has {len(song.notes):,} notes. "
                "Performance autoplay will be forced at launch to avoid scoring/visual lag."
            )
        self.learning_warning_label.configure(text="\n".join(messages))
        self._sync_mischief_label(rush_e_locked, song)

    def _sync_learning_state(self):
        if not hasattr(self, "learning_song_combo"):
            return  # called before UI is built sometimes
        enabled = bool(self.vars["learning_enabled"].get()) and bool(self.song_choices)
        rush_e_locked = self._apply_rush_e_autoplay_lock()
        combo_state = "readonly" if enabled else "disabled"
        switch_state = "normal" if enabled else "disabled"
        self.learning_song_combo.configure(state=combo_state)
        if hasattr(self, "learning_autoplay_switch"):
            self.learning_autoplay_switch.configure(
                state="disabled" if rush_e_locked else switch_state,
                text=RUSH_E_AUTOPLAY_LABEL if rush_e_locked else LEARNING_AUTOPLAY_LABEL,
            )
        if hasattr(self, "performance_autoplay_switch"):
            performance_state = switch_state if bool(self.vars["learning_autoplay"].get()) else "disabled"
            self.performance_autoplay_switch.configure(state=performance_state)
        if self.vars["learning_enabled"].get() and not self.song_choices:
            self._set_status("No learning songs found in the songs folder.", WARNING)
        self._sync_learning_warning()

    def _refresh_songs(self):
        self.song_choices = self._load_song_choices()
        values = list(self.song_choices) or ["No songs found"]
        if hasattr(self, "learning_song_combo"):
            self.learning_song_combo.configure(values=values)
        if hasattr(self, "editor_song_combo"):
            self.editor_song_combo.configure(values=values)
        if self.song_choices:
            if self.vars["learning_song"].get() not in self.song_choices:
                self.vars["learning_song"].set(next(iter(self.song_choices)))
            if self.vars["editor_song"].get() not in self.song_choices:
                self.vars["editor_song"].set(self.vars["learning_song"].get())
            self._set_status(f"Found {len(self.song_choices)} learning song(s).", SUCCESS)
        else:
            self.vars["learning_song"].set("")
            self.vars["editor_song"].set("")
            self.vars["learning_enabled"].set(False)
            self._set_status("No learning songs found in the songs folder.", WARNING)
        self._sync_learning_state()
        self._save_settings()

    def _open_song_editor(self, new_song=False):
        song_path = None
        if not new_song:
            song_path = self.song_choices.get(self.vars["editor_song"].get())
            if song_path is None:
                self._set_status("Choose a song to edit, or create a new one.", WARNING)
                return
        SongEditorWindow(self, song_path)

    def _song_path_for_import(self, title):
        # similar to the editor window version but kept separate
        SONGS_DIR.mkdir(parents=True, exist_ok=True)
        slug = "".join(c.lower() if c.isalnum() else "_" for c in title).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug or "imported_song"
        candidate = SONGS_DIR / f"{slug}.json"
        n = 2
        while candidate.exists():
            candidate = SONGS_DIR / f"{slug}_{n}.json"
            n += 1
        return candidate

    def _finish_song_import(self, song):
        out_path = self._song_path_for_import(song.title)
        save_song(song, str(out_path))
        self._refresh_songs()
        for label, path in self.song_choices.items():
            if path == out_path:
                self.vars["learning_song"].set(label)
                self.vars["editor_song"].set(label)
                break
        detail = f"Imported {song.title} with {len(song.notes):,} notes."
        if song.author.startswith("MIDI Import, smart-reduced"):
            detail += " Dense MIDI was thinned for real-time playback."
        elif song.author.startswith("MIDI Import, full source notes"):
            detail += " Full MIDI kept intact; enable Autoplayer performance mode for dense files."
        if is_rush_e_name(song.title):
            detail += " Rush E detected. Autoplay locks on in Learning Mode."
        self._set_status(detail, SUCCESS)
        if hasattr(self, "import_status_label"):
            self.import_status_label.configure(
                text=f"{detail} Ready in Learning Mode.",
                text_color=SUCCESS,
            )
        self._refresh_summary()

    def _set_import_controls(self, running):
        for button in self.import_buttons:
            button.configure(state="disabled" if running else "normal")

    def _run_song_import(self, label, importer, path):
        self._set_import_controls(True)
        if hasattr(self, "import_status_label"):
            self.import_status_label.configure(text=f"{label} import is running...", text_color=WARNING)
        self._set_status(f"{label} import is running...", WARNING)

        def worker():
            try:
                song = importer(path)
            except Exception as exc:
                self.after(0, lambda exc=exc: self._finish_song_import_error(label, exc))
                return
            self.after(0, lambda song=song: self._finish_song_import_success(song))

        threading.Thread(target=worker, daemon=True).start()

    def _choose_midi_import_mode(self, path):
        choice = tk.StringVar(value="")
        dialog = ctk.CTkToplevel(self)
        dialog.title("MIDI import mode")
        dialog.geometry("560x310")
        dialog.resizable(False, False)
        dialog.configure(fg_color=BG)
        dialog.transient(self)
        dialog.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            dialog,
            text="How should this MIDI be imported?",
            text_color=TEXT,
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=22, pady=(22, 6))
        ctk.CTkLabel(
            dialog,
            text=(
                f"{Path(path).name}\n\n"
                "Full keeps every note from the file. Smart-reduced strips the density down to something a human can follow. "
                "Black MIDI files work either way; Full just means the renderer earns its keep."
            ),
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
            justify="left",
            wraplength=510,
        ).grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 16))

        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 18))
        buttons.grid_columnconfigure(0, weight=1)

        def choose(value):
            choice.set(value)
            dialog.destroy()

        ctk.CTkButton(
            buttons,
            text="Keep Full MIDI",
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            height=38,
            command=lambda: choose("full"),
        ).grid(row=0, column=0, sticky="ew", pady=5)
        ctk.CTkButton(
            buttons,
            text="Smart-Reduce For Practice",
            fg_color=PANEL_ALT,
            hover_color=BORDER_HOT,
            height=38,
            command=lambda: choose("smart"),
        ).grid(row=1, column=0, sticky="ew", pady=5)
        ctk.CTkButton(
            buttons,
            text="Cancel",
            fg_color=DANGER,
            hover_color="#c0392b",
            height=34,
            command=lambda: choose("cancel"),
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)
        return choice.get() or "cancel"

    def _finish_song_import_success(self, song):
        self._set_import_controls(False)
        self._finish_song_import(song)

    def _finish_song_import_error(self, label, exc):
        self._set_import_controls(False)
        message = f"{label} import failed: {exc}"
        if hasattr(self, "import_status_label"):
            self.import_status_label.configure(text=message, text_color=DANGER)
        self._set_status(message, DANGER)

    def _import_midi_song(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Import MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not path:
            return
        mode = self._choose_midi_import_mode(path)
        if mode == "cancel":
            self._set_status("MIDI import cancelled.", TEXT_MUTED)
            return
        if mode == "smart":
            self._run_song_import("MIDI smart-reduced", lambda midi_path: import_midi(midi_path, smart_limit=True), path)
            return
        self._run_song_import("MIDI full", lambda midi_path: import_midi(midi_path, smart_limit=False), path)

    def _import_musicxml_song(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Import MusicXML file",
            filetypes=[("MusicXML files", "*.musicxml *.xml *.mxl"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run_song_import("MusicXML", import_musicxml, path)

    def _import_sheet_song(self):
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Scan sheet music images or PDF",
            filetypes=[
                ("Sheet images and PDFs", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        paths = list(paths)
        image_paths = [path for path in paths if Path(path).suffix.lower() != ".pdf"]
        if len(image_paths) > 1 and len(image_paths) == len(paths):
            SheetScanStageWindow(self, image_paths)
            return
        self._start_sheet_scan(paths)

    def _confirm_sheet_scan(self, paths):
        pdf_count = sum(1 for path in paths if Path(path).suffix.lower() == ".pdf")
        image_count = len(paths) - pdf_count
        source_text = []
        if pdf_count:
            source_text.append(f"{pdf_count} PDF(s), all pages")
        if image_count:
            source_text.append(f"{image_count} image(s)")
        detail = " and ".join(source_text) or "selected files"
        return messagebox.askokcancel(
            "Start sheet scan",
            (
                f"Scan {detail}?\n\n"
                "This can take several minutes and may use a lot of CPU and memory. "
                "The result is experimental; MIDI or MusicXML imports are usually cleaner."
            ),
            parent=self,
        )

    def _start_sheet_scan(self, paths):
        paths = list(paths)
        if not paths:
            self._set_status("Choose at least one sheet image or PDF.", WARNING)
            return
        if not self._confirm_sheet_scan(paths):
            self._set_status("Sheet scan cancelled.", TEXT_MUTED)
            return
        self._run_song_import("Sheet scan", import_sheet_files, paths)

    def _refresh_cameras(self):
        self._stop_preview()
        self.cameras = detect_cameras()
        self.camera_combo.configure(values=self.cameras)
        if self.vars["camera"].get() not in self.cameras:
            self.vars["camera"].set(self.cameras[0])
        self._refresh_readiness_labels()
        self._set_status(f"Found {len(self.cameras)} camera option(s).", SUCCESS)

    def _draw_preview_placeholder(self, text):
        if not hasattr(self, "preview_canvas"):
            return
        self.preview_canvas.delete("all")
        for y in range(PREVIEW_HEIGHT):
            color = _blend_hex("#090d14", "#101b2a", y / max(1, PREVIEW_HEIGHT - 1))
            self.preview_canvas.create_line(0, y, PREVIEW_WIDTH, y, fill=color)
        self.preview_canvas.create_rectangle(8, 8, PREVIEW_WIDTH - 8, PREVIEW_HEIGHT - 8, outline=BORDER_HOT)
        self.preview_canvas.create_line(12, 12, PREVIEW_WIDTH - 12, 12, fill=ACCENT)
        self.preview_canvas_image_id = None
        self.preview_canvas_text_id = self.preview_canvas.create_text(
            PREVIEW_WIDTH // 2,
            PREVIEW_HEIGHT // 2,
            text=text,
            fill=TEXT_MUTED,
            font=("Segoe UI", 13),
        )

    def _toggle_preview(self):
        if self.preview_running:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        self._stop_preview()

        try:
            camera_index = int(self.vars["camera"].get())
        except ValueError:
            self._set_status("Camera must be a numeric device index.", DANGER)
            return

        self.preview_cap = open_camera(camera_index, include_fallback=True)
        if self.preview_cap is None:
            self._set_status(f"Camera {camera_index} could not be opened.", DANGER)
            return

        self.preview_running = True
        self.previewBtn.configure(text="Stop Preview")
        self._set_status(f"Previewing camera {camera_index}.", SUCCESS)
        self._update_preview_frame()

    def _stop_preview(self):
        self.preview_running = False

        if self.preview_after_id is not None:
            try:
                self.after_cancel(self.preview_after_id)
            except Exception:
                pass
            self.preview_after_id = None

        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None

        if hasattr(self, "preview_btn"):
            self.previewBtn.configure(text="Start Preview")
        self._draw_preview_placeholder("Preview stopped")
        self.preview_photo = None
        self.preview_image = None
    
    def _update_preview_frame(self):
        if not self.preview_running or self.preview_cap is None:
            return

        ok, frame = self.preview_cap.read()
        if not ok:
            self._stop_preview()
            self._set_status("Camera preview stopped because frames were not available.", DANGER)
            return

        if self.vars["mirror"].get():
            frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        # draw a fake header bar to match the tracking window look
        cv2.rectangle(frame, (0, 0), (PREVIEW_WIDTH, 34), (14, 20, 30), -1)
        cv2.line(frame, (0, 34), (PREVIEW_WIDTH, 34), (255, 184, 107), 1)
        cv2.putText(frame, "LIVE PREVIEW", (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 245, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, f"CAM {self.vars['camera'].get()}", (PREVIEW_WIDTH - 74, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (107, 184, 255), 1, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.preview_image = Image.fromarray(frame)
        self.preview_photo = ImageTk.PhotoImage(self.preview_image, master=self)
        if self.preview_canvas_image_id is None:
            self.preview_canvas.delete("all")
            self.preview_canvas_text_id = None
            self.preview_canvas_image_id = self.preview_canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.preview_photo,
            )
        else:
            self.preview_canvas.itemconfig(self.preview_canvas_image_id, image=self.preview_photo)
        self.preview_after_id = self.after(50, self._update_preview_frame)
    
    def _test_camera(self):
        try:
            camera_index = int(self.vars["camera"].get())
        except ValueError:
            self._set_status("Camera must be a numeric device index.", DANGER)
            return

        cap = open_camera(camera_index, include_fallback=True)
        ok = cap is not None
        if ok:
            ok, _frame = cap.read()
            cap.release()

        if ok:
            self._set_status(f"Camera {camera_index} is working.", SUCCESS)
        else:
            self._set_status(f"Camera {camera_index} is not returning frames.", DANGER)


    def _refresh_readiness_labels(self):
        ready = []
        missing = []
        for label, folder in INSTRUMENTS.items():
            wav_count = len(list((APP_DIR / folder).glob("*.wav"))) if (APP_DIR / folder).is_dir() else 0
            if wav_count:
                ready.append(f"{label}: {wav_count} samples")
            else:
                missing.append(label)

        if missing:
            samples_text = "Samples need setup: " + ", ".join(missing)
            self.samples_label.configure(text=samples_text, text_color=WARNING)
        else:
            self.samples_label.configure(text=f"Samples ready: {len(ready)}/{len(INSTRUMENTS)} instrument packs", text_color=TEXT_MUTED)
        self.camera_label.configure(text=f"Cameras detected: {', '.join(self.cameras)}")
        self._sync_sample_selection_status()
        self._refresh_summary()

    def _readiness_text(self):
        missing = [label for label, folder in INSTRUMENTS.items() if not list((APP_DIR / folder).glob("*.wav"))]
        if missing:
            return "Some sample packs are missing. Choose the ones you want and download them"
        return "Ready. Settings are saved automatically on launch."

    def _collect_settings(self):
        self._apply_rush_e_autoplay_lock()
        return {
            "mode": self.vars["mode"].get(),
            "camera": self.vars["camera"].get(),
            "instrument": self.vars["instrument"].get(),
            "octave": int(float(self.vars["octave"].get())),
            "piano_octaves": self.vars["piano_octaves"].get(),
            "volume": int(float(self.vars["volume"].get())),
            "resolution": self.vars["resolution"].get(),
            "tracking_scale": int(float(self.vars["tracking_scale"].get())),
            "max_hands": self.vars["max_hands"].get(),
            "detection_confidence": int(float(self.vars["detection_confidence"].get())),
            "tracking_confidence": int(float(self.vars["tracking_confidence"].get())),
            "mirror": bool(self.vars["mirror"].get()),
            "show_landmarks": bool(self.vars["show_landmarks"].get()),
            "show_note_labels": bool(self.vars["show_note_labels"].get()),
            "show_fps": bool(self.vars["show_fps"].get()),
            "show_note_trail": bool(self.vars["show_note_trail"].get()),
            "piano_top": int(float(self.vars["piano_top"].get())),
            "piano_bottom": int(float(self.vars["piano_bottom"].get())),
            "piano_opacity": int(float(self.vars["piano_opacity"].get())),
            "play_trigger": self.vars["play_trigger"].get(),
            "trigger_cooldown_ms": int(float(self.vars["trigger_cooldown_ms"].get())),
            "smoothing": int(float(self.vars["smoothing"].get())),
            "dead_zone": int(float(self.vars["dead_zone"].get())),
            "fadeout_ms": int(float(self.vars["fadeout_ms"].get())),
            "metronome": bool(self.vars["metronome"].get()),
            "metronome_bpm": int(float(self.vars["metronome_bpm"].get())),
            "download_all_samples": bool(self.vars["download_all_samples"].get()),
            "download_grand": bool(self.vars["download_grand"].get()),
            "download_bright": bool(self.vars["download_bright"].get()),
            "download_electronic": bool(self.vars["download_electronic"].get()),
            "download_organ": bool(self.vars["download_organ"].get()),
            "download_reverb": bool(self.vars["download_reverb"].get()),
            "keybed_preset": self.vars["keybed_preset"].get(),
            "learning_enabled": bool(self.vars["learning_enabled"].get()),
            "learning_song": self.vars["learning_song"].get(),
            "learning_autoplay": bool(self.vars["learning_autoplay"].get()),
            "performance_autoplay": bool(self.vars["performance_autoplay"].get()),
            "editor_song": self.vars["editor_song"].get(),
        }

    def _save_settings(self):
        try:
            with SETTINGS_FILE.open("w", encoding="utf-8") as f:
                json.dump(self._collect_settings(), f, indent=2)
        except OSError as exc:
            self._set_status(f"Could not save settings: {exc}", DANGER)

    def _reset_defaults(self):
        for key, value in DEFAULT_SETTINGS.items():
            self.vars[key].set(value)
        self.update_labels()
        self._sync_sample_selection_status()
        self._sync_learning_state()
        self._sync_mode_state()
        self._save_settings()
        self._set_status("Defaults restored.", SUCCESS)

    def _validate(self):
        if self.vars["mode"].get() != "Air Piano":
            return False, "Desk Mode is not available yet."
        if not HAND_TRACKING_SCRIPT.exists():
            return False, f"Missing runtime file: {HAND_TRACKING_SCRIPT}"

        try:
            int(self.vars["camera"].get())
        except ValueError:
            return False, "Camera must be a numeric device index."
        if self.vars["camera"].get() not in self.cameras:
            return False, "Choose a camera from the detected camera list."
        if self.vars["instrument"].get() not in INSTRUMENTS:
            return False, "Choose an available instrument."
        if self.vars["resolution"].get() not in RESOLUTION_OPTIONS:
            return False, "Choose one of the supported camera resolutions."
        if self.vars["piano_octaves"].get() not in PIANO_OCTAVE_OPTIONS:
            return False, "Choose how many octaves to show from the dropdown."
        if self.vars["max_hands"].get() not in MAX_HAND_OPTIONS:
            return False, "Choose whether to track one hand or two hands."
        if self.vars["play_trigger"].get() not in PLAY_TRIGGER_ARGS:
            return False, "Choose a valid air piano play trigger."
        if self.vars["keybed_preset"].get() not in KEYBED_PRESETS:
            return False, "Choose a valid keybed preset."
        instrument_folder = INSTRUMENTS[self.vars["instrument"].get()]
        if not (APP_DIR / instrument_folder).is_dir():
            return False, f"Missing sample folder: {instrument_folder}. Download them from settings tab"
        if not list((APP_DIR / instrument_folder).glob("*.wav")):
            return False, f"No WAV samples found in {instrument_folder}. Download them from settings tab"

        top = int(float(self.vars["piano_top"].get()))
        bottom = int(float(self.vars["piano_bottom"].get()))
        if bottom - top < 12:  # need at least a bit of room
            return False, "The keybed needs at least 12% vertical height."
        if self.vars["learning_enabled"].get():
            song_path = self.song_choices.get(self.vars["learning_song"].get())
            if song_path is None or not song_path.exists():
                return False, "Choose an available learning song or refresh the song list."
        return True, ""

    def _show_learning_range_dialog(self, status):
        choice = tk.StringVar(value="")
        dialog = ctk.CTkToplevel(self)
        dialog.title("Song range warning")
        dialog.geometry("520x300")
        dialog.resizable(False, False)
        dialog.configure(fg_color=BG)
        dialog.transient(self)

        dialog.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            dialog,
            text="This song does not fit the visible piano range.",
            text_color=TEXT,
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        ctk.CTkLabel(
            dialog,
            text=(
                f"{status['song'].title} uses C{status['song_low_octave']}-"
                f"B{status['song_high_octave']} ({status['needed_octaves']} octaves). "
                f"Your piano shows {status['shown_octaves']} octave(s), starting at C{status['start_octave']}."
            ),
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
            justify="left",
            wraplength=470,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 14))

        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=2, column=0, sticky="ew", padx=20, pady=(2, 16))
        buttons.grid_columnconfigure(0, weight=1)

        def choose(value):
            choice.set(value)
            dialog.destroy()

        row = 0
        if status["needed_octaves"] <= MAX_PIANO_OCTAVES:
            label = (
                f"Use {status['suggested_octaves']} octaves "
                f"(C{status['suggested_start']}-B{status['suggested_start'] + status['suggested_octaves'] - 1})"
            )
            ctk.CTkButton(
                buttons,
                text=label,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
                command=lambda: choose("adjust"),
            ).grid(row=row, column=0, sticky="ew", pady=4)
            row += 1

        ctk.CTkButton(
            buttons,
            text=f"Retune to fit current {status['shown_octaves']} octave(s)",
            fg_color=PANEL_ALT if status["needed_octaves"] <= MAX_PIANO_OCTAVES else ACCENT,
            hover_color=BORDER if status["needed_octaves"] <= MAX_PIANO_OCTAVES else ACCENT_HOVER,
            command=lambda: choose("retune"),
        ).grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        ctk.CTkButton(
            buttons,
            text="Continue unchanged",
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=lambda: choose("continue"),
        ).grid(row=row, column=0, sticky="ew", pady=4)
        row += 1
        ctk.CTkButton(
            buttons,
            text="Cancel launch",
            fg_color=DANGER,
            hover_color="#c0392b",
            command=lambda: choose("cancel"),
        ).grid(row=row, column=0, sticky="ew", pady=4)

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)
        return choice.get() or "cancel"

    def _resolve_learning_range_before_launch(self):
        self.learning_fit_mode = "retune"
        if not self.vars["learning_enabled"].get():
            self.learning_fit_mode = "original"
            return True
        status = self._learning_range_status()
        if status is None:
            self.learning_fit_mode = "original"
            return True

        choice = self._show_learning_range_dialog(status)
        if choice == "cancel":
            self._set_status("Launch cancelled.", TEXT_MUTED)
            return False
        if choice == "adjust":
            self.vars["piano_octaves"].set(str(status["suggested_octaves"]))
            self.vars["octave"].set(status["suggested_start"])
            self.learning_fit_mode = "original"
            self.update_labels()
            self._set_status("Adjusted visible octaves for the selected song.", SUCCESS)
            return True
        if choice == "retune":
            self.learning_fit_mode = "retune"
            self._set_status("Learning mode will retune the song to the visible keys.", WARNING)
            return True

        self.learning_fit_mode = "original"
        self._set_status("Continuing with the original song range.", WARNING)
        return True

    def _build_command(self):
        self._normalize_octave()
        self._sync_piano_bounds()
        resolution = self.vars["resolution"].get().replace(" ", "")
        width, height = resolution.split("x", maxsplit=1)
        instrument_folder = INSTRUMENTS[self.vars["instrument"].get()]

        command = [
            sys.executable,
            str(HAND_TRACKING_SCRIPT),
            "--mode",
            "air",
            "--camera",
            self.vars["camera"].get(),
            "--instrument",
            instrument_folder,
            "--octave",
            str(int(float(self.vars["octave"].get()))),
            "--piano-octaves",
            self.vars["piano_octaves"].get(),
            "--volume",
            str(int(float(self.vars["volume"].get()))),
            "--play-trigger",
            PLAY_TRIGGER_ARGS.get(self.vars["play_trigger"].get(), "precision"),
            "--camera-width",
            width,
            "--camera-height",
            height,
            "--tracking-scale",
            f"{int(float(self.vars['tracking_scale'].get())) / 100:.2f}",
            "--max-hands",
            self.vars["max_hands"].get(),
            "--min-detection-confidence",
            f"{int(float(self.vars['detection_confidence'].get())) / 100:.2f}",
            "--min-tracking-confidence",
            f"{int(float(self.vars['tracking_confidence'].get())) / 100:.2f}",
            "--piano-top-ratio",
            f"{int(float(self.vars['piano_top'].get())) / 100:.2f}",
            "--piano-bottom-ratio",
            f"{int(float(self.vars['piano_bottom'].get())) / 100:.2f}",
            "--piano-alpha",
            f"{int(float(self.vars['piano_opacity'].get())) / 100:.2f}",
            "--hover-cooldown",
            f"{int(float(self.vars['trigger_cooldown_ms'].get())) / 1000:.3f}",
            "--smoothing",
            str(int(float(self.vars["smoothing"].get()))),
            "--dead-zone",
            str(int(float(self.vars["dead_zone"].get()))),
            "--fadeout-ms",
            str(int(float(self.vars["fadeout_ms"].get()))),
            "--metronome-bpm",
            str(int(float(self.vars["metronome_bpm"].get()))),
        ]

        command.append("--mirror" if self.vars["mirror"].get() else "--no-mirror")
        command.append("--landmarks" if self.vars["show_landmarks"].get() else "--no-landmarks")
        command.append("--note-labels" if self.vars["show_note_labels"].get() else "--no-note-labels")
        command.append("--fps" if self.vars["show_fps"].get() else "--no-fps")
        command.append("--note-trail" if self.vars["show_note_trail"].get() else "--no-note-trail")
        if self.vars["metronome"].get():
            command.append("--metronome")
        if self.vars["learning_enabled"].get():
            rush_e_locked = self._apply_rush_e_autoplay_lock()
            song_path = self.song_choices.get(self.vars["learning_song"].get())
            if song_path is not None:
                command.extend(
                    [
                        "--learning-song",
                        str(song_path),
                        "--learning-fit",
                        self.learning_fit_mode,
                    ]
                )
            if self.vars["learning_autoplay"].get() or rush_e_locked:
                command.append("--learning-autoplay")
                if self.vars["performance_autoplay"].get():
                    command.append("--performance-autoplay")
        return command

    def _download_selected_samples(self):
        if self.setup_process is not None and self.setup_process.poll() is None:
            self._set_status("Sample setup is already running.", WARNING)
            return
        if self.process is not None and self.process.poll() is None:
            self._set_status("Close the piano session before changing sample packs.", WARNING)
            return

        selected = self._selected_sample_args()
        if not selected:
            self._set_status("Select at least one instrument to download.", DANGER)
            return

        self._save_settings()
        command = [
            sys.executable,
            str(APP_DIR / "setup_sounds.py"),
            "--instruments",
            *selected,
        ]
        try:
            self.setup_process = subprocess.Popen(command, cwd=str(APP_DIR))
        except OSError as exc:
            self.setup_process = None
            self._set_status(f"Sample setup failed to start: {exc}", DANGER)
            return

        self._set_setup_controls(True)
        self._sync_sample_selection_status()
        target = self._sample_args_label(selected)
        self._set_status(f"Downloading {target}. This can take a few minutes. Check Terminal for more", WARNING)
        self.after(1000, self._poll_setup_process)

    def _poll_setup_process(self):
        if self.setup_process is None:
            return

        exit_code = self.setup_process.poll()
        if exit_code is None:
            self.after(1000, self._poll_setup_process)
            return

        self.setup_process = None
        self._set_setup_controls(False)
        self._refresh_readiness_labels()

        ready_count = sum(
            1
            for folder in INSTRUMENTS.values()
            if (APP_DIR / folder).is_dir() and list((APP_DIR / folder).glob("*.wav"))
        )
        if exit_code == 0:
            self._set_status(f"Sample setup complete. {ready_count}/{len(INSTRUMENTS)} packs are ready.", SUCCESS)
        else:
            self._set_status(f"Sample setup exited with code {exit_code}. Check the terminal output.", DANGER)


    def _launch(self):
        self._stop_preview()
        if self.setup_process is not None and self.setup_process.poll() is None:
            self._set_status("Wait for sample setup to finish before launching.", WARNING)
            return

        valid, message = self._validate()
        if not valid:
            self._set_status(message, DANGER)
            return

        if not self._resolve_learning_range_before_launch():
            return

        self._save_settings()
        command = self._build_command()
        try:
            self.process = subprocess.Popen(command, cwd=str(APP_DIR))
        except OSError as exc:
            self.process = None
            self._set_status(f"Launch failed: {exc}", DANGER)
            return

        self.launchButton.configure(state="disabled", text="Running")
        self.stop_btn.configure(state="normal")
        if self.vars["learning_enabled"].get():
            self._set_status("Learning mode is running. Use the camera window for live play.", SUCCESS)
        else:
            self._set_status("Air Piano is running. Use the camera window for live play.", SUCCESS)
        self.after(700, self._poll_process)

    def _poll_process(self):
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            self.after(700, self._poll_process)
            return

        self.process = None
        self.stop_btn.configure(state="disabled")
        if self.vars["mode"].get() == "Air Piano":
            setup_running = self.setup_process is not None and self.setup_process.poll() is None
            self.launchButton.configure(state="disabled" if setup_running else "normal", text=self._launch_button_text())
        if exit_code == 0:
            self._set_status("Piano session closed.", TEXT_MUTED)
        else:
            # non-zero = probably crashed
            self._set_status(f"Piano exited with code {exit_code}. Check the terminal output.", DANGER)

    def _stop_session(self):
        if self.process is None or self.process.poll() is not None:
            self.process = None
            self.stop_btn.configure(state="disabled")
            setup_running = self.setup_process is not None and self.setup_process.poll() is None
            self.launchButton.configure(state="disabled" if setup_running else "normal", text=self._launch_button_text())
            return
        self.process.terminate()
        self._set_status("Stopping piano session...", WARNING)
        self.after(700, self._poll_process)

    def _set_status(self, message, color):
        self.status_var.set(message)
        if hasattr(self, "status_label"):
            self.status_label.configure(text_color=color)

    def _on_close(self):
        self._save_settings()
        self._stop_preview()
        # kill child processes on exit
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self.setup_process is not None and self.setup_process.poll() is None:
            self.setup_process.terminate()
        self.destroy()


class SheetScanStageWindow(ctk.CTkToplevel):
    def __init__(self, parent, paths):
        super().__init__(parent)
        self.parent = parent
        self.paths = list(paths)
        self.preview_photo = None
        self.title("Stage Sheet Images")
        self.geometry("860x560")
        self.minsize(780, 500)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="Stage images before scanning",
            text_color=TEXT,
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            self,
            text="Put pages in reading order, preview each image, and remove anything accidental.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="new", padx=20, pady=(0, 12))

        list_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        list_panel.grid(row=2, column=0, sticky="nsew", padx=(20, 10), pady=(0, 16))
        list_panel.grid_rowconfigure(1, weight=1)
        list_panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(list_panel, text="ORDER", text_color=TEXT_MUTED, font=ctk.CTkFont(size=11, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )
        self.listbox = tk.Listbox(
            list_panel,
            width=34,
            height=18,
            bg=PANEL_ALT,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground=TEXT,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self.listbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.listbox.bind("<<ListboxSelect>>", lambda _event: self._update_preview())

        order_controls = ctk.CTkFrame(list_panel, fg_color="transparent")
        order_controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        order_controls.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(order_controls, text="Up", fg_color=PANEL_ALT, hover_color=BORDER, command=self._move_up).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6)
        )
        ctk.CTkButton(order_controls, text="Down", fg_color=PANEL_ALT, hover_color=BORDER, command=self._move_down).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6)
        )
        ctk.CTkButton(order_controls, text="Delete", fg_color=DANGER, hover_color="#c0392b", command=self._delete_selected).grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )

        preview_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        preview_panel.grid(row=2, column=1, sticky="nsew", padx=(10, 20), pady=(0, 16))
        preview_panel.grid_columnconfigure(0, weight=1)
        preview_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(preview_panel, text="PREVIEW", text_color=TEXT_MUTED, font=ctk.CTkFont(size=11, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )
        self.preview_canvas = tk.Canvas(
            preview_panel,
            width=480,
            height=330,
            bg="#0a0d12",
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))

        footer = ctk.CTkFrame(preview_panel, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(footer, text="Cancel", fg_color=PANEL_ALT, hover_color=BORDER, width=100, command=self.destroy).grid(
            row=0, column=1, sticky="e", padx=(0, 8)
        )
        ctk.CTkButton(footer, text="Scan In This Order", fg_color=ACCENT, hover_color=ACCENT_HOVER, width=170, command=self._scan).grid(
            row=0, column=2, sticky="e"
        )

        self._refresh_list(0)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _selected_index(self):
        selection = self.listbox.curselection()
        return selection[0] if selection else None

    def _refresh_list(self, selected_index=None):
        self.listbox.delete(0, tk.END)
        for i, path in enumerate(self.paths, start=1):
            self.listbox.insert(tk.END, f"{i:02d}. {Path(path).name}")
        if self.paths:
            idx = 0 if selected_index is None else max(0, min(selected_index, len(self.paths) - 1))
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)
        self._update_preview()

    def _update_preview(self):
        self.preview_canvas.delete("all")
        index = self._selected_index()
        if index is None or index >= len(self.paths):
            self.preview_canvas.create_text(240, 165, text="No image selected", fill=TEXT_MUTED, font=("Segoe UI", 13))
            self.preview_photo = None
            return
        path = self.paths[index]
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((500, 350), Image.LANCZOS)
        except Exception as exc:
            self.preview_canvas.create_text(240, 165, text=f"Preview failed: {exc}", fill=DANGER, font=("Segoe UI", 11), width=430)
            self.preview_photo = None
            return
        self.preview_photo = ImageTk.PhotoImage(image, master=self)
        self.preview_canvas.create_image(250, 175, anchor="center", image=self.preview_photo)
        self.preview_canvas.create_text(12, 18, anchor="w", text=Path(path).name, fill=TEXT_MUTED, font=("Segoe UI", 10))

    def _move_up(self):
        index = self._selected_index()
        if index is None or index <= 0:
            return
        self.paths[index - 1], self.paths[index] = self.paths[index], self.paths[index - 1]
        self._refresh_list(index - 1)

    def _move_down(self):
        index = self._selected_index()
        if index is None or index >= len(self.paths) - 1:
            return
        self.paths[index + 1], self.paths[index] = self.paths[index], self.paths[index + 1]
        self._refresh_list(index + 1)

    def _delete_selected(self):
        index = self._selected_index()
        if index is None:
            return
        del self.paths[index]
        self._refresh_list(index)

    def _scan(self):
        if not self.paths:
            self.parent._set_status("No sheet images left to scan.", WARNING)
            return
        paths = list(self.paths)
        self.destroy()
        self.parent.after(0, lambda: self.parent._start_sheet_scan(paths))


class SongEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, song_path=None):
        super().__init__(parent)
        self.parent = parent
        self.song_path = Path(song_path) if song_path else None
        self.note_rows = []

        if self.song_path is not None:
            try:
                self.song = load_song(str(self.song_path))
            except Exception as exc:
                self.song = Song(title="Untitled", author="", bpm=120, base_octave=4, octaves_used=1, notes=[])
                parent._set_status(f"Could not load song: {exc}", DANGER)
        else:
            self.song = Song(title="Untitled", author="", bpm=120, base_octave=4, octaves_used=1, notes=[])

        self.large_song_read_only = len(self.song.notes) > EDITOR_RENDER_NOTE_LIMIT  # too many rows = freezes
        self.title("Song Editor")
        self.geometry("760x640")
        self.minsize(700, 540)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.grab_set()

        self.title_var = ctk.StringVar(value=self.song.title)
        self.author_var = ctk.StringVar(value=self.song.author)
        self.bpm_var = ctk.StringVar(value=str(self.song.bpm))
        self.base_octave_var = ctk.StringVar(value=str(self.song.base_octave))
        self.octaves_used_var = ctk.StringVar(value=str(self.song.octaves_used))
        self.status_var = ctk.StringVar(value="")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_editor()
        self._rebuild_note_rows()
        if self.large_song_read_only:
            self.status_var.set(
                f"Large song: showing first {EDITOR_RENDER_NOTE_LIMIT} of {len(self.song.notes)} notes. Editing is disabled."
            )
            self.status_label.configure(text_color=WARNING)
        self.focus_force()

    def _build_editor(self):
        meta = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        meta.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        meta.grid_columnconfigure((1, 3), weight=1)

        self._metadata_entry(meta, 0, 0, "Title", self.title_var)
        self._metadata_entry(meta, 0, 2, "Author", self.author_var)
        self._metadata_entry(meta, 1, 0, "BPM", self.bpm_var, width=100)
        self._metadata_entry(meta, 1, 2, "Base Octave", self.base_octave_var, width=100)
        self._metadata_entry(meta, 2, 0, "Octaves", self.octaves_used_var, width=100)

        actions = ctk.CTkFrame(meta, fg_color="transparent")
        actions.grid(row=2, column=2, columnspan=2, sticky="e", padx=14, pady=(6, 14))
        self.add_note_btn = ctk.CTkButton(
            actions,
            text="Add Note",
            width=110,
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            state="disabled" if self.large_song_read_only else "normal",
            command=self._add_note,
        )
        self.add_note_btn.pack(side="left", padx=(0, 8))
        self.save_btn = ctk.CTkButton(
            actions,
            text="Save",
            width=110,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            state="disabled" if self.large_song_read_only else "normal",
            command=self._save,
        )
        self.save_btn.pack(side="left")

        notes_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        notes_panel.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        notes_panel.grid_columnconfigure(0, weight=1)
        notes_panel.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(notes_panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        for column, (text, width) in enumerate((("Note", 120), ("Start", 120), ("Duration", 120), ("Hand", 120), ("", 80))):
            ctk.CTkLabel(header, text=text, text_color=TEXT_MUTED, width=width, anchor="w").grid(row=0, column=column, sticky="w", padx=4)

        self.notes_frame = ctk.CTkScrollableFrame(notes_panel, fg_color="transparent")
        self.notes_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(footer, textvariable=self.status_var, text_color=TEXT_MUTED, anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(footer, text="Close", width=100, fg_color=PANEL_ALT, hover_color=BORDER, command=self.destroy).grid(row=0, column=1, sticky="e")

    def _metadata_entry(self, parent, row, column, label, variable, width=None):
        ctk.CTkLabel(parent, text=label, text_color=TEXT_MUTED, anchor="w").grid(row=row, column=column, sticky="w", padx=(14, 8), pady=8)
        entry = ctk.CTkEntry(parent, textvariable=variable, fg_color=PANEL_ALT, border_color=BORDER, width=width or 210)
        entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=8)
        return entry

    def _rebuild_note_rows(self):
        for child in self.notes_frame.winfo_children():
            child.destroy()
        self.note_rows = []
        visible = self.song.notes[:EDITOR_RENDER_NOTE_LIMIT] if self.large_song_read_only else self.song.notes
        for i, note in enumerate(visible):
            self._create_note_row(i, note)

    def _create_note_row(self, i, note):
        row = ctk.CTkFrame(self.notes_frame, fg_color="transparent")
        row.grid(row=i, column=0, sticky="ew", pady=3)
        row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        note_var = ctk.StringVar(value=note.note)
        time_var = ctk.StringVar(value=f"{note.time:g}")
        duration_var = ctk.StringVar(value=f"{note.duration:g}")
        hand_var = ctk.StringVar(value=note.hand if note.hand in HAND_OPTIONS else "any")

        entry_state = "disabled" if self.large_song_read_only else "normal"
        hand_combo_state = "disabled" if self.large_song_read_only else "readonly"
        ctk.CTkEntry(row, textvariable=note_var, width=120, fg_color=PANEL_ALT, border_color=BORDER, state=entry_state).grid(row=0, column=0, sticky="ew", padx=4)
        ctk.CTkEntry(row, textvariable=time_var, width=120, fg_color=PANEL_ALT, border_color=BORDER, state=entry_state).grid(row=0, column=1, sticky="ew", padx=4)
        ctk.CTkEntry(row, textvariable=duration_var, width=120, fg_color=PANEL_ALT, border_color=BORDER, state=entry_state).grid(row=0, column=2, sticky="ew", padx=4)
        StrictComboBox(
            row,
            values=HAND_OPTIONS,
            variable=hand_var,
            width=120,
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            state=hand_combo_state,
        ).grid(row=0, column=3, sticky="ew", padx=4)
        ctk.CTkButton(
            row,
            text="Delete",
            width=80,
            fg_color=PANEL_ALT,
            hover_color=DANGER,
            state="disabled" if self.large_song_read_only else "normal",
            command=lambda note_index=i: self._delete_note(note_index),
        ).grid(row=0, column=4, sticky="e", padx=4)

        self.note_rows.append(
            {
                "note": note_var,
                "time": time_var,
                "duration": duration_var,
                "hand": hand_var,
            }
        )

    def _snapshot_notes(self, sort_notes=True):
        if self.large_song_read_only:
            raise ValueError("Large songs are read only in the built-in editor.")
        notes = []
        for row in self.note_rows:
            note_name = normalize_note_name(row["note"].get())
            note_to_midi(note_name)
            start = float(row["time"].get())
            duration = float(row["duration"].get())
            if start < 0:
                raise ValueError("Start time cannot be negative.")
            if duration <= 0:
                raise ValueError("Duration must be greater than zero.")
            hand = row["hand"].get().strip().lower() or "any"
            if hand not in HAND_OPTIONS:
                hand = "any"
            notes.append(SongNote(note=note_name, time=start, duration=duration, hand=hand))
        if sort_notes:
            notes.sort(key=lambda item: (item.time, item.note))
        return notes

    def _add_note(self):
        try:
            current_notes = self._snapshot_notes()
        except Exception:
            current_notes = list(self.song.notes)
        next_start = 0.0
        if current_notes:
            next_start = max(note.end_time for note in current_notes)
        self.song.notes = current_notes + [SongNote("C4", next_start, 1.0, "any")]
        self._rebuild_note_rows()
        self.status_var.set("Added note.")
        self.status_label.configure(text_color=TEXT_MUTED)

    def _delete_note(self, index):
        try:
            notes = self._snapshot_notes(sort_notes=False)
        except Exception:
            notes = list(self.song.notes)
        if 0 <= index < len(notes):
            del notes[index]
        self.song.notes = notes
        self._rebuild_note_rows()
        self.status_var.set("Deleted note.")
        self.status_label.configure(text_color=TEXT_MUTED)

    def _read_song(self):
        title = self.title_var.get().strip() or "Untitled"
        author = self.author_var.get().strip()
        bpm = int(float(self.bpm_var.get()))
        base_octave = int(float(self.base_octave_var.get()))
        octaves_used = int(float(self.octaves_used_var.get()))
        if bpm <= 0:
            raise ValueError("BPM must be greater than zero.")
        if base_octave < 0 or base_octave > 8:
            raise ValueError("Base octave must be between 0 and 8.")
        if octaves_used < 1 or octaves_used > 4:
            raise ValueError("Octaves must be between 1 and 4.")
        return Song(
            title=title,
            author=author,
            bpm=bpm,
            base_octave=base_octave,
            octaves_used=octaves_used,
            notes=self._snapshot_notes(),
        )

    def _song_path_for_title(self, title):
        SONGS_DIR.mkdir(parents=True, exist_ok=True)
        slug = "".join(char.lower() if char.isalnum() else "_" for char in title).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        slug = slug or "untitled"
        candidate = SONGS_DIR / f"{slug}.json"
        counter = 2
        while candidate.exists():
            candidate = SONGS_DIR / f"{slug}_{counter}.json"
            counter += 1
        return candidate

    def _save(self):
        try:
            song = self._read_song()
            if self.song_path is None:
                self.song_path = self._song_path_for_title(song.title)
            save_song(song, str(self.song_path))
        except Exception as exc:
            self.status_var.set(f"Save failed: {exc}")
            self.status_label.configure(text_color=DANGER)
            return

        self.song = song
        self.status_var.set(f"Saved {self.song_path.name}.")
        self.status_label.configure(text_color=SUCCESS)
        self.parent._refresh_songs()
        for label, path in self.parent.song_choices.items():
            if path == self.song_path:
                self.parent.vars["editor_song"].set(label)
                break
        self.parent._save_settings()


if __name__ == "__main__":
    app = PianoLauncher()
    app.mainloop()
