import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pygame


WINDOW_TITLE = "InstrumentalCV Piano"

FINGERTIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_JOINTS = {
    "thumb": {"tip": 4, "pip": 2},
    "index": {"tip": 8, "pip": 6},
    "middle": {"tip": 12, "pip": 10},
    "ring": {"tip": 16, "pip": 14},
    "pinky": {"tip": 20, "pip": 18},
}

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_MAGENTA = (0, 80, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PANEL = (28, 32, 38)
COLOR_PANEL_ALT = (40, 48, 58)
COLOR_TEXT_DIM = (190, 198, 205)

FINGER_COLORS = {
    "thumb": COLOR_YELLOW,
    "index": COLOR_GREEN,
    "middle": COLOR_BLUE,
    "ring": COLOR_ORANGE,
    "pinky": COLOR_MAGENTA,
}

CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHITE_NOTES = ["C", "D", "E", "F", "G", "A", "B"]
BLACK_KEY_AFTER = {0, 1, 3, 4, 5}
BLACK_NOTES = {0: "C#", 1: "D#", 3: "F#", 4: "G#", 5: "A#"}

WHITE_KEY_HOVER_COLOR = (255, 222, 185)
BLACK_KEY_HOVER_COLOR = (120, 95, 210)

INSTRUMENTS = {
    "sounds": "Grand Piano",
    "sounds_bright": "Bright Piano",
    "sounds_electronic": "Electric Piano",
    "sounds_organ": "Organ",
    "sounds_reverb": "Reverb Piano",
}
INSTRUMENT_HOTKEYS = {ord(str(index)): folder for index, folder in enumerate(INSTRUMENTS, start=1)}

LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}
PLUS_KEYS = {ord("+"), ord("="), 171, 43}
MINUS_KEYS = {ord("-"), ord("_"), 173, 45}


@dataclass
class RuntimeConfig:
    mode: str
    camera: int
    instrument: str
    start_octave: int
    piano_octaves: int
    volume: int
    camera_width: int
    camera_height: int
    max_hands: int
    min_detection_confidence: float
    min_tracking_confidence: float
    mirror: bool
    show_landmarks: bool
    show_note_labels: bool
    show_fps: bool
    show_note_trail: bool
    piano_top_ratio: float
    piano_bottom_ratio: float
    piano_alpha: float
    hover_cooldown: float
    smoothing_window: int
    dead_zone_px: int
    fadeout_ms: int
    metronome: bool
    metronome_bpm: int


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run InstrumentalCV Air Piano.")
    parser.add_argument("--mode", choices=["air", "desk"], default="air")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--instrument", type=str, default="sounds")
    parser.add_argument("--octave", type=int, default=3)
    parser.add_argument("--piano-octaves", type=int, default=2)
    parser.add_argument("--volume", type=int, default=70)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)

    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--landmarks", dest="show_landmarks", action="store_true", default=True)
    parser.add_argument("--no-landmarks", dest="show_landmarks", action="store_false")
    parser.add_argument("--note-labels", dest="show_note_labels", action="store_true", default=True)
    parser.add_argument("--no-note-labels", dest="show_note_labels", action="store_false")
    parser.add_argument("--fps", dest="show_fps", action="store_true", default=True)
    parser.add_argument("--no-fps", dest="show_fps", action="store_false")
    parser.add_argument("--note-trail", dest="show_note_trail", action="store_true", default=True)
    parser.add_argument("--no-note-trail", dest="show_note_trail", action="store_false")

    parser.add_argument("--piano-top-ratio", type=float, default=0.30)
    parser.add_argument("--piano-bottom-ratio", type=float, default=0.60)
    parser.add_argument("--piano-alpha", type=float, default=0.60)
    parser.add_argument("--hover-cooldown", type=float, default=0.15)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--dead-zone", type=int, default=3)
    parser.add_argument("--fadeout-ms", type=int, default=300)
    parser.add_argument("--metronome", action="store_true")
    parser.add_argument("--metronome-bpm", type=int, default=120)

    args = parser.parse_args(argv)
    piano_octaves = clamp(args.piano_octaves, 1, 4)
    max_start_octave = 8 - piano_octaves
    start_octave = clamp(args.octave, 1, max_start_octave)
    top_ratio = clamp(args.piano_top_ratio, 0.05, 0.85)
    bottom_ratio = clamp(args.piano_bottom_ratio, top_ratio + 0.10, 0.95)

    return RuntimeConfig(
        mode=args.mode,
        camera=args.camera,
        instrument=args.instrument,
        start_octave=start_octave,
        piano_octaves=piano_octaves,
        volume=clamp(args.volume, 0, 100),
        camera_width=clamp(args.camera_width, 320, 3840),
        camera_height=clamp(args.camera_height, 240, 2160),
        max_hands=clamp(args.max_hands, 1, 2),
        min_detection_confidence=clamp(args.min_detection_confidence, 0.1, 0.95),
        min_tracking_confidence=clamp(args.min_tracking_confidence, 0.1, 0.95),
        mirror=args.mirror,
        show_landmarks=args.show_landmarks,
        show_note_labels=args.show_note_labels,
        show_fps=args.show_fps,
        show_note_trail=args.show_note_trail,
        piano_top_ratio=top_ratio,
        piano_bottom_ratio=bottom_ratio,
        piano_alpha=clamp(args.piano_alpha, 0.2, 0.95),
        hover_cooldown=clamp(args.hover_cooldown, 0.03, 1.0),
        smoothing_window=clamp(args.smoothing, 1, 12),
        dead_zone_px=clamp(args.dead_zone, 0, 20),
        fadeout_ms=clamp(args.fadeout_ms, 30, 1500),
        metronome=args.metronome,
        metronome_bpm=clamp(args.metronome_bpm, 40, 240),
    )


class PositionSmoother: # reduces jitter of finger accross frames
    def __init__(self, window=3):
        self.history = {}
        self.window = window

    def smooth(self, finger_key, x, y):
        if finger_key not in self.history:
            self.history[finger_key] = deque(maxlen=self.window)
        self.history[finger_key].append((x, y))
        points = self.history[finger_key]
        avg_x = int(sum(point[0] for point in points) / len(points))
        avg_y = int(sum(point[1] for point in points) / len(points))
        return avg_x, avg_y


class HoverTracker:
    def __init__(self, finger_name, cooldown_seconds):
        self.finger_name = finger_name
        self.cooldown_seconds = cooldown_seconds
        self.prev_key_note = None
        self.current_key_note = None
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.last_trigger_time = 0.0

    def update(self, current_key, current_time):
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.current_key_note = current_key["note"] if current_key else None

        if self.current_key_note == self.prev_key_note:
            if self.current_key_note is not None:
                self.is_held = True
        else:
            if self.prev_key_note is not None:
                self.note_off = True
                self.released_note = self.prev_key_note
            if self.current_key_note is not None:
                elapsed = current_time - self.last_trigger_time
                if elapsed >= self.cooldown_seconds:
                    self.note_on = True
                    self.last_trigger_time = current_time

        self.prev_key_note = self.current_key_note


class NoteTrail:
    def __init__(self, max_notes=24):
        self.notes = deque(maxlen=max_notes)

    def add(self, note_name):
        self.notes.append((note_name, time.time()))

    def draw(self, frame, y=98):
        now = time.time()
        x = 12
        for note, timestamp in self.notes:
            age = now - timestamp
            if age > 5.0:
                continue
            brightness = max(0, 1.0 - age / 5.0)
            color = (0, int(210 * brightness), int(255 * brightness))
            cv2.putText(frame, note, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x += 50
            if x > frame.shape[1] - 50:
                break


class Metronome:
    def __init__(self, bpm=120, enabled=False):
        self.bpm = bpm
        self.enabled = enabled
        self.last_beat = time.time()
        self.beat_count = 0
        self.click = self._make_click()

    def _make_click(self):
        sample_rate = 44100
        duration = 0.03
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 50)
        wave_int16 = (wave * 32767 * 0.3).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack((wave_int16, wave_int16)))
        return pygame.sndarray.make_sound(stereo)

    def toggle(self):
        self.enabled = not self.enabled
        self.last_beat = time.time()
        self.beat_count = 0

    def change_bpm(self, delta):
        self.bpm = clamp(self.bpm + delta, 40, 240)

    def update(self, frame, frame_w):
        if not self.enabled:
            return

        now = time.time()
        interval = 60.0 / self.bpm
        if now - self.last_beat >= interval:
            self.last_beat = now
            self.beat_count += 1
            self.click.play()

        phase = clamp((now - self.last_beat) / interval, 0.0, 1.0)
        radius = int(16 * (1.0 - phase))
        color = COLOR_YELLOW if self.beat_count % 4 == 1 else (0, 200, 200)
        cx = frame_w - 42
        cy = 72
        if radius > 0:
            cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.putText(frame, f"{self.bpm} BPM", (cx - 45, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DIM, 1)


class SoundEngine:
    def __init__(self, max_volume=0.7):
        self.max_volume = clamp(max_volume, 0.0, 1.0)
        self.sounds = {}
        self.channels = {}
        self.current_instrument = "sounds"
        self.last_error = None

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)

    def load_instrument(self, folder_name="sounds"): # loads the chrods of diff instruments
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(base_dir, folder_name)
        loaded = self._load_from_dir(sounds_dir)
        if loaded:
            self.current_instrument = folder_name
        return loaded

    def _load_from_dir(self, sounds_dir):  #for loading downloaded notes from sounds folder
        self.last_error = None
        if not os.path.isdir(sounds_dir):
            self.last_error = f"Missing sound folder: {os.path.basename(sounds_dir)}"
            print(f"ERROR: {self.last_error}. Run setup_sounds.py first.")
            return 0

        print(f"Loading samples from {os.path.basename(sounds_dir)}...")
        loaded_sounds = {}
        for filename in sorted(os.listdir(sounds_dir)):
            if not filename.lower().endswith(".wav"):
                continue
            note_name = filename[:-4]
            try:
                loaded_sounds[note_name] = pygame.mixer.Sound(os.path.join(sounds_dir, filename))
            except Exception as exc:
                print(f"  Failed to load {filename}: {exc}")

        self.sounds = loaded_sounds
        loaded = len(self.sounds)
        if loaded == 0:
            self.last_error = f"No WAV samples found in {os.path.basename(sounds_dir)}"
            print(f"ERROR: {self.last_error}. Run setup_sounds.py first.")
        else:
            print(f"Loaded {loaded} samples")
        return loaded

    def note_on(self, note_name, velocity=100):
        sound = self.sounds.get(note_name)
        if sound is None:
            return
        channel = pygame.mixer.find_channel()
        if channel is None:
            return
        volume = (velocity / 127.0) * self.max_volume
        channel.set_volume(volume, volume)
        channel.play(sound)
        self.channels[note_name] = channel

    def note_off(self, note_name, fadeout_ms=300):
        channel = self.channels.pop(note_name, None)
        if channel and channel.get_busy():
            channel.fadeout(fadeout_ms)

    def all_notes_off(self, fadeout_ms=150):
        for note_name in list(self.channels):
            self.note_off(note_name, fadeout_ms)

    def cleanup(self):
        self.all_notes_off(50)
        pygame.mixer.quit()


def is_finger_extended(hand_landmarks, finger_name):  #only plays if tip joint is above the pip joint
    joints = FINGER_JOINTS[finger_name]
    tip = hand_landmarks.landmark[joints["tip"]]
    pip = hand_landmarks.landmark[joints["pip"]]

    if finger_name == "thumb":
        wrist = hand_landmarks.landmark[0]
        return abs(tip.x - wrist.x) > abs(pip.x - wrist.x)
    return tip.y < pip.y


def generate_piano_keys(frame_w, frame_h, config):
    white_keys = []
    black_keys = []
    white_key_count = 7 * config.piano_octaves
    white_key_w = frame_w / white_key_count
    piano_top = int(frame_h * config.piano_top_ratio)
    piano_bottom = int(frame_h * config.piano_bottom_ratio)
    black_key_h = int((piano_bottom - piano_top) * 0.60)
    black_key_w = int(white_key_w * 0.60)

    for index in range(white_key_count):
        octave = config.start_octave + (index // 7)
        note_idx = index % 7
        note_name = f"{WHITE_NOTES[note_idx]}{octave}"
        x1 = int(index * white_key_w)
        x2 = int((index + 1) * white_key_w)
        white_keys.append(
            {
                "note": note_name,
                "x1": x1,
                "y1": piano_top,
                "x2": x2,
                "y2": piano_bottom,
                "is_black": False,
                "color_default": COLOR_WHITE,
                "color_hover": WHITE_KEY_HOVER_COLOR,
                "color": COLOR_WHITE,
            }
        )

    for index in range(white_key_count):
        octave = config.start_octave + (index // 7)
        note_idx = index % 7
        if note_idx not in BLACK_KEY_AFTER:
            continue
        note_name = f"{BLACK_NOTES[note_idx]}{octave}"
        center_x = int((index + 1) * white_key_w)
        x1 = center_x - (black_key_w // 2)
        x2 = center_x + (black_key_w // 2)
        black_keys.append(
            {
                "note": note_name,
                "x1": x1,
                "y1": piano_top,
                "x2": x2,
                "y2": piano_top + black_key_h,
                "is_black": True,
                "color_default": COLOR_BLACK,
                "color_hover": BLACK_KEY_HOVER_COLOR,
                "color": COLOR_BLACK,
            }
        )

    return white_keys, black_keys


def draw_piano(frame, white_keys, black_keys, alpha, show_note_labels):
    if not white_keys:
        return

    overlay = frame.copy()
    for key in white_keys:
        draw_y1 = key["y1"] + 4 if key["color"] == COLOR_GREEN else key["y1"]
        draw_color = (0, 200, 0) if key["color"] == COLOR_GREEN else key["color"]
        cv2.rectangle(overlay, (key["x1"], draw_y1), (key["x2"], key["y2"]), draw_color, -1)
        cv2.rectangle(overlay, (key["x1"], draw_y1), (key["x2"], key["y2"]), (180, 180, 180), 1)

    for key in black_keys:
        draw_y1 = key["y1"] + 3 if key["color"] == COLOR_GREEN else key["y1"]
        draw_color = (0, 180, 0) if key["color"] == COLOR_GREEN else key["color"]
        cv2.rectangle(overlay, (key["x1"], draw_y1), (key["x2"], key["y2"]), draw_color, -1)
        cv2.rectangle(overlay, (key["x1"], draw_y1), (key["x2"], key["y2"]), (80, 80, 80), 1)

    piano_top = white_keys[0]["y1"]
    blended_region = cv2.addWeighted(
        overlay[piano_top:, :],
        alpha,
        frame[piano_top:, :],
        1 - alpha,
        0,
    )
    frame[piano_top:, :] = blended_region

    if show_note_labels:
        for key in white_keys:
            text_size = cv2.getTextSize(key["note"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = key["x1"] + (key["x2"] - key["x1"] - text_size[0]) // 2
            text_y = key["y2"] - 10
            cv2.putText(frame, key["note"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 75, 80), 1)

        for key in black_keys:
            text_size = cv2.getTextSize(key["note"], cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = key["x1"] + (key["x2"] - key["x1"] - text_size[0]) // 2
            text_y = key["y2"] - 8
            cv2.putText(frame, key["note"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (210, 210, 210), 1)

    cv2.line(frame, (0, piano_top), (frame.shape[1], piano_top), (0, 200, 255), 2)


def detect_finger_on_key(finger_x, finger_y, all_keys, dead_zone_px):
    for key in all_keys:
        if (key["x1"] + dead_zone_px) <= finger_x <= (key["x2"] - dead_zone_px) and key["y1"] <= finger_y <= key["y2"]:
            return key
    return None


def reset_key_colors(white_keys, black_keys):
    for key in white_keys:
        key["color"] = key["color_default"]
    for key in black_keys:
        key["color"] = key["color_default"]


def draw_label(frame, text, x, y, scale=0.55, color=COLOR_WHITE):
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 6
    cv2.rectangle(frame, (x - pad, y - size[1] - pad), (x + size[0] + pad, y + baseline + pad), COLOR_PANEL, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(frame, config, current_octave, hovered_notes, fps, sound_engine, metronome):
    frame_h, frame_w = frame.shape[:2]
    if config.show_fps:
        fps_color = COLOR_GREEN if fps >= 20 else COLOR_RED
        draw_label(frame, f"FPS {int(fps)}", 12, 34, 0.6, fps_color)

    if hovered_notes:
        notes_text = "Playing: " + ", ".join(hovered_notes[:8])
        draw_label(frame, notes_text, 12, 68, 0.52, (0, 210, 255))
    else:
        draw_label(frame, "Air Piano ready", 12, 68, 0.52, COLOR_TEXT_DIM)

    instrument_label = INSTRUMENTS.get(sound_engine.current_instrument, sound_engine.current_instrument)
    draw_label(frame, "AIR PIANO", frame_w - 190, 34, 0.55, (0, 220, 255))
    draw_label(frame, f"C{current_octave}-B{current_octave + config.piano_octaves - 1}  {instrument_label}", frame_w - 300, 68, 0.45, COLOR_TEXT_DIM)

    if sound_engine.last_error:
        draw_label(frame, sound_engine.last_error, 12, frame_h - 18, 0.45, COLOR_RED)
    else:
        draw_label(frame, "Q quit | M metronome | +/- BPM | arrows octave | 1-5 sounds", 12, frame_h - 18, 0.43, COLOR_TEXT_DIM)

    metronome.update(frame, frame_w)


def get_camera_backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def release_inactive_trackers(finger_trackers, active_tracker_keys, current_time, sound_engine, fadeout_ms):
    for tracker_key, tracker in finger_trackers.items():
        if tracker_key in active_tracker_keys or tracker.prev_key_note is None:
            continue
        tracker.update(None, current_time)
        if tracker.note_off:
            sound_engine.note_off(tracker.released_note, fadeout_ms)


def handle_keypress(key_pressed, config, current_octave, rebuild_piano, sound_engine, metronome):
    if key_pressed == -1:
        return current_octave, False

    key_lower = key_pressed
    if 65 <= key_pressed <= 90:
        key_lower = key_pressed + 32

    if key_lower in {ord("q"), 27}:
        return current_octave, True

    if key_lower == ord("m"):
        metronome.toggle()
        print(f"Metronome: {'ON' if metronome.enabled else 'OFF'} ({metronome.bpm} BPM)")
        return current_octave, False

    if key_pressed in PLUS_KEYS:
        metronome.change_bpm(5)
        print(f"Metronome BPM: {metronome.bpm}")
        return current_octave, False

    if key_pressed in MINUS_KEYS:
        metronome.change_bpm(-5)
        print(f"Metronome BPM: {metronome.bpm}")
        return current_octave, False

    if key_pressed in LEFT_ARROW_KEYS:
        if current_octave > 1:
            current_octave -= 1
            config.start_octave = current_octave
            rebuild_piano()
            sound_engine.all_notes_off()
            print(f"Octave: C{current_octave} - B{current_octave + config.piano_octaves - 1}")
        return current_octave, False

    if key_pressed in RIGHT_ARROW_KEYS:
        max_start_octave = 8 - config.piano_octaves
        if current_octave < max_start_octave:
            current_octave += 1
            config.start_octave = current_octave
            rebuild_piano()
            sound_engine.all_notes_off()
            print(f"Octave: C{current_octave} - B{current_octave + config.piano_octaves - 1}")
        return current_octave, False

    instrument_folder = INSTRUMENT_HOTKEYS.get(key_pressed)
    if instrument_folder:
        if sound_engine.load_instrument(instrument_folder):
            print(f"Loaded instrument: {INSTRUMENTS[instrument_folder]}")
        return current_octave, False

    print(f"Unhandled key code: {key_pressed}")
    return current_octave, False


def main(argv=None):
    config = parse_args(argv)
    if config.mode != "air":
        print("Desk mode is planned, but this runtime currently supports air mode only.")
        return 2

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.max_hands,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )

    cap = cv2.VideoCapture(config.camera, get_camera_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    sound_engine = None
    try:
        if not cap.isOpened():
            print(f"Error: Webcam {config.camera} not found or already in use.")
            return 2

        ret, temp_frame = cap.read()
        if not ret:
            print("Error: Unable to read a frame from the selected camera.")
            return 2
        if config.mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        frame_h, frame_w = temp_frame.shape[:2]
        white_keys, black_keys = generate_piano_keys(frame_w, frame_h, config)
        all_keys = black_keys + white_keys

        def rebuild_piano():
            nonlocal white_keys, black_keys, all_keys, frame_w, frame_h
            white_keys, black_keys = generate_piano_keys(frame_w, frame_h, config)
            all_keys = black_keys + white_keys

        finger_trackers = {}

        def get_tracker(tracker_key, finger_name):
            if tracker_key not in finger_trackers:
                finger_trackers[tracker_key] = HoverTracker(finger_name, config.hover_cooldown)
            return finger_trackers[tracker_key]

        prev_time = 0.0
        note_trail = NoteTrail()
        smoother = PositionSmoother(window=config.smoothing_window)
        sound_engine = SoundEngine(max_volume=config.volume / 100.0)
        sound_engine.load_instrument(config.instrument)
        metronome = Metronome(bpm=config.metronome_bpm, enabled=config.metronome)

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        print("InstrumentalCV Air Piano is running. Press Q in the camera window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Webcam stopped returning frames.")
                break

            if config.mirror:
                frame = cv2.flip(frame, 1)
            new_h, new_w = frame.shape[:2]
            if new_h != frame_h or new_w != frame_w:
                frame_h, frame_w = new_h, new_w
                rebuild_piano()

            current_time = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            reset_key_colors(white_keys, black_keys)
            hovered_notes = []
            active_tracker_keys = set()

            if result.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    hand_label = "Hand"
                    if result.multi_handedness:
                        hand_label = result.multi_handedness[hand_idx].classification[0].label

                    if config.show_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1),
                        )

                    for finger_name, tip_id in FINGERTIP_IDS.items():
                        tracker_key = f"{hand_idx}_{hand_label}_{finger_name}"
                        active_tracker_keys.add(tracker_key)
                        tracker = get_tracker(tracker_key, finger_name)

                        if not is_finger_extended(hand_landmarks, finger_name):
                            tracker.update(None, current_time)
                            if tracker.note_off:
                                sound_engine.note_off(tracker.released_note, config.fadeout_ms)
                            continue

                        tip = hand_landmarks.landmark[tip_id]
                        tip_x, tip_y = int(tip.x * frame_w), int(tip.y * frame_h)
                        tip_x, tip_y = smoother.smooth(tracker_key, tip_x, tip_y)
                        hovered_key = detect_finger_on_key(tip_x, tip_y, all_keys, config.dead_zone_px)
                        tracker.update(hovered_key, current_time)

                        dot_color = COLOR_GREEN if tracker.note_on or tracker.is_held else FINGER_COLORS[finger_name]
                        cv2.circle(frame, (tip_x, tip_y), 10, dot_color, -1)
                        cv2.circle(frame, (tip_x, tip_y), 12, COLOR_WHITE, 1)

                        if hovered_key is not None:
                            if tracker.is_held or tracker.note_on:
                                hovered_key["color"] = COLOR_GREEN
                            else:
                                hovered_key["color"] = hovered_key["color_hover"]
                            hovered_notes.append(hovered_key["note"])
                            key_center_x = (hovered_key["x1"] + hovered_key["x2"]) // 2
                            cv2.line(frame, (tip_x, tip_y), (key_center_x, hovered_key["y1"]), dot_color, 1, cv2.LINE_AA)

                        if tracker.note_on and hovered_key is not None:
                            note = hovered_key["note"]
                            print(f"NOTE ON: {note} ({hand_label} {finger_name})")
                            sound_engine.note_on(note)
                            if config.show_note_trail:
                                note_trail.add(note)
                            key_cx = (hovered_key["x1"] + hovered_key["x2"]) // 2
                            key_cy = (hovered_key["y1"] + hovered_key["y2"]) // 2
                            cv2.circle(frame, (key_cx, key_cy), 20, COLOR_GREEN, 2)
                            cv2.circle(frame, (key_cx, key_cy), 30, COLOR_GREEN, 1)

                        if tracker.note_off:
                            print(f"NOTE OFF: {tracker.released_note} ({hand_label} {finger_name})")
                            sound_engine.note_off(tracker.released_note, config.fadeout_ms)

                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * frame_w), int(wrist.y * frame_h)
                    cv2.putText(frame, hand_label, (wrist_x - 30, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)

            release_inactive_trackers(finger_trackers, active_tracker_keys, current_time, sound_engine, config.fadeout_ms)
            draw_piano(frame, white_keys, black_keys, config.piano_alpha, config.show_note_labels)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            if config.show_note_trail:
                note_trail.draw(frame)
            draw_hud(frame, config, config.start_octave, hovered_notes, fps, sound_engine, metronome)
            cv2.imshow(WINDOW_TITLE, frame)

            key_pressed = cv2.waitKeyEx(1)
            config.start_octave, should_quit = handle_keypress(
                key_pressed,
                config,
                config.start_octave,
                rebuild_piano,
                sound_engine,
                metronome,
            )
            if should_quit:
                break

        return 0
    finally:
        if sound_engine is not None:
            sound_engine.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Piano closed.")


if __name__ == "__main__":
    sys.exit(main())
