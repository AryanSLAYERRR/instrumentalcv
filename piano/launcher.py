import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import customtkinter as ctk


APP_DIR = Path(__file__).resolve().parent
HAND_TRACKING_SCRIPT = APP_DIR / "hand_tracking.py"
SETTINGS_FILE = APP_DIR / "user_settings.json"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BG = "#0f1217"
PANEL = "#171c24"
PANEL_ALT = "#1f2630"
BORDER = "#2d3644"
TEXT = "#eef3f8"
TEXT_MUTED = "#95a3b5"
ACCENT = "#2f80ed"
ACCENT_HOVER = "#2567c5"
SUCCESS = "#27ae60"
WARNING = "#f2c94c"
DANGER = "#eb5757"

INSTRUMENTS = {
    "Grand Piano": "sounds",
    "Bright Piano": "sounds_bright",
    "Electric Piano": "sounds_electronic",
    "Organ": "sounds_organ",
    "Reverb Piano": "sounds_reverb",
}

DEFAULT_SETTINGS = {
    "mode": "Air Piano",
    "camera": "0",
    "instrument": "Grand Piano",
    "octave": 3,
    "piano_octaves": "2",
    "volume": 70,
    "resolution": "1280 x 720",
    "max_hands": "2",
    "detection_confidence": 50,
    "tracking_confidence": 50,
    "mirror": True,
    "show_landmarks": True,
    "show_note_labels": True,
    "show_fps": True,
    "show_note_trail": True,
    "piano_top": 30,
    "piano_bottom": 60,
    "piano_opacity": 60,
    "trigger_cooldown_ms": 150,
    "smoothing": 3,
    "dead_zone": 3,
    "fadeout_ms": 300,
    "metronome": False,
    "metronome_bpm": 120,
}


def get_camera_backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def detect_cameras(max_check=8):
    available = []
    for index in range(max_check):
        cap = cv2.VideoCapture(index, get_camera_backend())
        if cap.isOpened():
            available.append(str(index))
        cap.release()
    return available or ["0"]


def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if not SETTINGS_FILE.exists():
        return settings

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return settings

    for key in settings:
        if key in saved:
            settings[key] = saved[key]
    if settings["instrument"] not in INSTRUMENTS:
        settings["instrument"] = DEFAULT_SETTINGS["instrument"]
    if settings["mode"] not in {"Air Piano", "Desk Mode"}:
        settings["mode"] = DEFAULT_SETTINGS["mode"]
    return settings


class PianoLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("InstrumentalCV Launcher")
        self.geometry("940x780")
        self.minsize(880, 680)
        self.configure(fg_color=BG)

        self.process = None
        self.cameras = detect_cameras()
        self.settings = load_settings()
        if self.settings["camera"] not in self.cameras:
            self.settings["camera"] = self.cameras[0]

        self.slider_labels = {}
        self.vars = self._create_vars()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_footer()
        self._sync_all_labels()
        self._sync_mode_state()
        self._set_status(self._readiness_text(), SUCCESS)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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
            "trigger_cooldown_ms": ctk.IntVar(value=int(settings["trigger_cooldown_ms"])),
            "smoothing": ctk.IntVar(value=int(settings["smoothing"])),
            "dead_zone": ctk.IntVar(value=int(settings["dead_zone"])),
            "fadeout_ms": ctk.IntVar(value=int(settings["fadeout_ms"])),
            "metronome": ctk.BooleanVar(value=bool(settings["metronome"])),
            "metronome_bpm": ctk.IntVar(value=int(settings["metronome_bpm"])),
        }

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="InstrumentalCV",
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Gesture-controlled piano setup",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.mode_pill = ctk.CTkLabel(
            header,
            text="AIR PIANO",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT,
            fg_color=ACCENT,
            corner_radius=999,
            width=120,
            height=30,
        )
        self.mode_pill.grid(row=0, column=1, rowspan=2, sticky="e")

    def _build_body(self):
        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        body.grid_columnconfigure((0, 1), weight=1, uniform="columns")

        left = ctk.CTkFrame(body, fg_color="transparent")
        right = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="new", padx=(0, 8))
        right.grid(row=0, column=1, sticky="new", padx=(8, 0))
        left.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_mode_section(left, 0)
        self._build_camera_section(left, 1)
        self._build_sound_section(left, 2)
        self._build_tracking_section(left, 3)
        self._build_piano_layout_section(right, 0)
        self._build_metronome_section(right, 1)
        self._build_controls_section(right, 2)
        self._build_readiness_section(right, 3)

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
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
            hover_color=BORDER,
            command=self._reset_defaults,
        )
        self.reset_btn.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=12)

        self.stop_btn = ctk.CTkButton(
            footer,
            text="Stop",
            width=86,
            fg_color=DANGER,
            hover_color="#c0392b",
            state="disabled",
            command=self._stop_session,
        )
        self.stop_btn.grid(row=0, column=2, sticky="e", padx=(0, 8), pady=12)

        self.launch_btn = ctk.CTkButton(
            footer,
            text="Launch Air Piano",
            width=170,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._launch,
        )
        self.launch_btn.grid(row=0, column=3, sticky="e", padx=(0, 12), pady=12)

    def _section(self, parent, title, row):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
        return frame

    def _row(self, parent, row, label):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=5)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label, text_color=TEXT, font=ctk.CTkFont(size=13), width=150, anchor="w").grid(row=0, column=0, sticky="w")
        return frame

    def _combo(self, parent, row, label, variable, values, command=None):
        frame = self._row(parent, row, label)
        combo = ctk.CTkComboBox(
            frame,
            values=values,
            variable=variable,
            command=command,
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            dropdown_fg_color=PANEL_ALT,
            dropdown_hover_color=BORDER,
            width=210,
        )
        combo.grid(row=0, column=1, sticky="e")
        return combo

    def _slider(self, parent, row, label, variable, from_, to, steps, key, suffix="", formatter=None, command=None):
        frame = self._row(parent, row, label)
        value_label = ctk.CTkLabel(frame, text="", text_color=TEXT_MUTED, width=58, anchor="e")
        value_label.grid(row=0, column=2, sticky="e", padx=(8, 0))
        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            number_of_steps=steps,
            variable=variable,
            command=lambda _value: self._on_slider_change(key, command),
        )
        slider.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.slider_labels[key] = (value_label, variable, suffix, formatter)
        return slider

    def _switch(self, parent, row, label, variable):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=16, pady=5)
        switch = ctk.CTkSwitch(
            frame,
            text=label,
            variable=variable,
            progress_color=ACCENT,
            button_color=TEXT,
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        switch.pack(anchor="w")
        return switch

    def _build_mode_section(self, parent, row):
        section = self._section(parent, "Mode", row)
        mode = ctk.CTkSegmentedButton(
            section,
            values=["Air Piano", "Desk Mode"],
            variable=self.vars["mode"],
            command=self._on_mode_change,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_color=PANEL_ALT,
            unselected_hover_color=BORDER,
        )
        mode.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.mode_control = mode

        ctk.CTkLabel(
            section,
            text="Desk Mode is reserved for the top-down camera workflow.",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

    def _build_camera_section(self, parent, row):
        section = self._section(parent, "Camera", row)
        camera_row = self._row(section, 1, "Camera device")
        self.camera_combo = ctk.CTkComboBox(
            camera_row,
            values=self.cameras,
            variable=self.vars["camera"],
            fg_color=PANEL_ALT,
            border_color=BORDER,
            button_color=ACCENT,
            width=116,
        )
        self.camera_combo.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(
            camera_row,
            text="Refresh",
            width=86,
            fg_color=PANEL_ALT,
            hover_color=BORDER,
            command=self._refresh_cameras,
        ).grid(row=0, column=2, sticky="e", padx=(8, 0))
        self._combo(section, 2, "Resolution", self.vars["resolution"], ["1280 x 720", "960 x 540", "640 x 480"])
        self._switch(section, 3, "Mirror camera preview", self.vars["mirror"])
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=4, column=0)

    def _build_sound_section(self, parent, row):
        section = self._section(parent, "Sound", row)
        self._combo(section, 1, "Instrument", self.vars["instrument"], list(INSTRUMENTS.keys()))
        self._slider(section, 2, "Starting octave", self.vars["octave"], 1, 6, 5, "octave", formatter=lambda value: f"C{value}")
        self._combo(section, 3, "Octaves shown", self.vars["piano_octaves"], ["1", "2", "3", "4"], command=lambda _value: self._normalize_octave())
        self._slider(section, 4, "Volume", self.vars["volume"], 0, 100, 100, "volume", "%")
        self._slider(section, 5, "Note fadeout", self.vars["fadeout_ms"], 30, 1200, 117, "fadeout_ms", " ms")
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=6, column=0)

    def _build_tracking_section(self, parent, row):
        section = self._section(parent, "Tracking", row)
        self._combo(section, 1, "Hands tracked", self.vars["max_hands"], ["1", "2"])
        self._slider(section, 2, "Detection", self.vars["detection_confidence"], 30, 90, 60, "detection_confidence", "%")
        self._slider(section, 3, "Tracking", self.vars["tracking_confidence"], 30, 90, 60, "tracking_confidence", "%")
        self._slider(section, 4, "Smoothing", self.vars["smoothing"], 1, 10, 9, "smoothing")
        self._slider(section, 5, "Trigger cooldown", self.vars["trigger_cooldown_ms"], 50, 500, 45, "trigger_cooldown_ms", " ms")
        self._slider(section, 6, "Key edge guard", self.vars["dead_zone"], 0, 16, 16, "dead_zone", " px")
        self._switch(section, 7, "Show hand landmarks", self.vars["show_landmarks"])
        self._switch(section, 8, "Show FPS", self.vars["show_fps"])
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=9, column=0)

    def _build_piano_layout_section(self, parent, row):
        section = self._section(parent, "Piano Layout", row)
        self._slider(section, 1, "Keybed top", self.vars["piano_top"], 10, 70, 60, "piano_top", "%", command=self._sync_piano_bounds)
        self._slider(section, 2, "Keybed bottom", self.vars["piano_bottom"], 30, 92, 62, "piano_bottom", "%", command=self._sync_piano_bounds)
        self._slider(section, 3, "Key opacity", self.vars["piano_opacity"], 25, 95, 70, "piano_opacity", "%")
        self._switch(section, 4, "Show note labels", self.vars["show_note_labels"])
        self._switch(section, 5, "Show note trail", self.vars["show_note_trail"])
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=6, column=0)

    def _build_metronome_section(self, parent, row):
        section = self._section(parent, "Metronome", row)
        self._switch(section, 1, "Start metronome on launch", self.vars["metronome"])
        self._slider(section, 2, "Tempo", self.vars["metronome_bpm"], 40, 220, 180, "metronome_bpm", " BPM")
        ctk.CTkFrame(section, height=8, fg_color="transparent").grid(row=3, column=0)

    def _build_controls_section(self, parent, row):
        section = self._section(parent, "Runtime Controls", row)
        controls = [
            ("Q / Esc", "Quit the piano window"),
            ("M", "Toggle metronome"),
            ("+ / -", "Adjust metronome tempo"),
            ("Left / Right", "Shift octave range"),
            ("1 - 5", "Switch instruments"),
        ]
        for index, (key, description) in enumerate(controls, start=1):
            line = ctk.CTkFrame(section, fg_color="transparent")
            line.grid(row=index, column=0, sticky="ew", padx=16, pady=4)
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
        section = self._section(parent, "Publish Readiness", row)
        self.samples_label = ctk.CTkLabel(section, text="", text_color=TEXT_MUTED, anchor="w", justify="left")
        self.samples_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.camera_label = ctk.CTkLabel(section, text="", text_color=TEXT_MUTED, anchor="w", justify="left")
        self.camera_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        self._refresh_readiness_labels()

    def _on_slider_change(self, key, extra_command=None):
        if extra_command:
            extra_command()
        self._sync_slider_label(key)

    def _sync_slider_label(self, key):
        label, variable, suffix, formatter = self.slider_labels[key]
        value = int(float(variable.get()))
        text = formatter(value) if formatter else f"{value}{suffix}"
        label.configure(text=text)

    def _sync_all_labels(self):
        for key in self.slider_labels:
            self._sync_slider_label(key)
        self._normalize_octave()
        self._sync_piano_bounds()

    def _sync_piano_bounds(self):
        top = int(float(self.vars["piano_top"].get()))
        bottom = int(float(self.vars["piano_bottom"].get()))
        if bottom - top < 12:
            if top <= 80:
                self.vars["piano_bottom"].set(min(92, top + 12))
            else:
                self.vars["piano_top"].set(max(10, bottom - 12))
        for key in ("piano_top", "piano_bottom"):
            if key in self.slider_labels:
                self._sync_slider_label(key)

    def _normalize_octave(self):
        octaves = int(self.vars["piano_octaves"].get())
        max_octave = 8 - octaves
        octave = int(float(self.vars["octave"].get()))
        if octave > max_octave:
            self.vars["octave"].set(max_octave)
        if "octave" in self.slider_labels:
            self._sync_slider_label("octave")

    def _on_mode_change(self, _value=None):
        self._sync_mode_state()

    def _sync_mode_state(self):
        mode = self.vars["mode"].get()
        if mode == "Desk Mode":
            self.mode_pill.configure(text="DESK MODE", fg_color=WARNING, text_color="#111111")
            self.launch_btn.configure(text="Desk Mode Coming Soon", state="disabled")
            self._set_status("Desk Mode is planned; switch back to Air Piano to launch today.", WARNING)
        else:
            self.mode_pill.configure(text="AIR PIANO", fg_color=ACCENT, text_color=TEXT)
            if self.process is None:
                self.launch_btn.configure(text="Launch Air Piano", state="normal")

    def _refresh_cameras(self):
        self.cameras = detect_cameras()
        self.camera_combo.configure(values=self.cameras)
        if self.vars["camera"].get() not in self.cameras:
            self.vars["camera"].set(self.cameras[0])
        self._refresh_readiness_labels()
        self._set_status(f"Found {len(self.cameras)} camera option(s).", SUCCESS)

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

    def _readiness_text(self):
        missing = [label for label, folder in INSTRUMENTS.items() if not list((APP_DIR / folder).glob("*.wav"))]
        if missing:
            return "Some sample packs are missing. Run setup_sounds.py before publishing."
        return "Ready. Settings are saved automatically on launch."

    def _collect_settings(self):
        return {
            "mode": self.vars["mode"].get(),
            "camera": self.vars["camera"].get(),
            "instrument": self.vars["instrument"].get(),
            "octave": int(float(self.vars["octave"].get())),
            "piano_octaves": self.vars["piano_octaves"].get(),
            "volume": int(float(self.vars["volume"].get())),
            "resolution": self.vars["resolution"].get(),
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
            "trigger_cooldown_ms": int(float(self.vars["trigger_cooldown_ms"].get())),
            "smoothing": int(float(self.vars["smoothing"].get())),
            "dead_zone": int(float(self.vars["dead_zone"].get())),
            "fadeout_ms": int(float(self.vars["fadeout_ms"].get())),
            "metronome": bool(self.vars["metronome"].get()),
            "metronome_bpm": int(float(self.vars["metronome_bpm"].get())),
        }

    def _save_settings(self):
        try:
            with SETTINGS_FILE.open("w", encoding="utf-8") as handle:
                json.dump(self._collect_settings(), handle, indent=2)
        except OSError as exc:
            self._set_status(f"Could not save settings: {exc}", DANGER)

    def _reset_defaults(self):
        for key, value in DEFAULT_SETTINGS.items():
            self.vars[key].set(value)
        self._sync_all_labels()
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
        instrument_folder = INSTRUMENTS[self.vars["instrument"].get()]
        if not (APP_DIR / instrument_folder).is_dir():
            return False, f"Missing sample folder: {instrument_folder}. Run setup_sounds.py."
        if not list((APP_DIR / instrument_folder).glob("*.wav")):
            return False, f"No WAV samples found in {instrument_folder}. Run setup_sounds.py."
        top = int(float(self.vars["piano_top"].get()))
        bottom = int(float(self.vars["piano_bottom"].get()))
        if bottom - top < 12:
            return False, "The keybed needs at least 12% vertical height."
        return True, ""

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
            "--camera-width",
            width,
            "--camera-height",
            height,
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
        return command

    def _launch(self):
        valid, message = self._validate()
        if not valid:
            self._set_status(message, DANGER)
            return

        self._save_settings()
        command = self._build_command()
        try:
            self.process = subprocess.Popen(command, cwd=str(APP_DIR))
        except OSError as exc:
            self.process = None
            self._set_status(f"Launch failed: {exc}", DANGER)
            return

        self.launch_btn.configure(state="disabled", text="Running")
        self.stop_btn.configure(state="normal")
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
            self.launch_btn.configure(state="normal", text="Launch Air Piano")
        if exit_code == 0:
            self._set_status("Piano session closed.", TEXT_MUTED)
        else:
            self._set_status(f"Piano exited with code {exit_code}. Check the terminal output.", DANGER)

    def _stop_session(self):
        if self.process is None or self.process.poll() is not None:
            self.process = None
            self.stop_btn.configure(state="disabled")
            self.launch_btn.configure(state="normal", text="Launch Air Piano")
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
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.destroy()


if __name__ == "__main__":
    app = PianoLauncher()
    app.mainloop()
