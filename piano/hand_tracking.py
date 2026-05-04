import argparse
import heapq
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from roast_lines import AUTOPLAY_ROASTS, RUSH_E_RUNTIME_LOCK_LINE
from song import is_rush_e_name, load_song, remap_song
from learning_renderer import FallingNoteRenderer
import cv2
import mediapipe as mp
import numpy as np
import pygame


WINDOW_TITLE = "InstrumentalCV Piano"

FINGERTIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_JOINTS = {
    "thumb": {"tip": 4, "pip": 2, "mcp": 2},
    "index": {"tip": 8, "pip": 6, "mcp": 5},
    "middle": {"tip": 12, "pip": 10, "mcp": 9},
    "ring": {"tip": 16, "pip": 14, "mcp": 13},
    "pinky": {"tip": 20, "pip": 18, "mcp": 17},
}
PALM_ANCHOR_IDS = (0, 5, 9, 13, 17)

COLOR_GREEN = (116, 255, 158)
COLOR_RED = (92, 92, 255)
COLOR_BLUE = (255, 170, 76)
COLOR_MAGENTA = (238, 104, 255)
COLOR_YELLOW = (76, 205, 255)
COLOR_CYAN = (255, 230, 96)
COLOR_ORANGE = (68, 168, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PANEL = (18, 22, 31)
COLOR_PANEL_ALT = (34, 43, 58)
COLOR_TEXT_DIM = (198, 210, 224)
COLOR_ACCENT = (255, 214, 70)
COLOR_WARM = (88, 184, 255)

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

WHITE_KEY_HOVER_COLOR = (255, 236, 198)
BLACK_KEY_HOVER_COLOR = (190, 118, 255)

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
LEARNING_MIN_PIANO_ALPHA = 0.78
PERFORMANCE_AUTOPLAY_NOTE_THRESHOLD = 80_000
AUTOPLAY_BATCH_SECONDS = 0.006
AUTOPLAY_MAX_AUDIO_NOTES_PER_BATCH = 96
PLAY_TRIGGERS = {"hover", "tap", "precision"}
PLAY_TRIGGER_LABELS = {
    "hover": "Hover",
    "tap": "Tap",
    "precision": "Precision Tap",
}
TAP_PRESS_FRACTION = 0.095
TAP_PRESS_MIN_PX = 13.0
TAP_PRESS_MAX_PX = 36.0
TAP_RELEASE_FRACTION = 0.42
TAP_DOWN_STEP_FRACTION = 0.026
TAP_DOWN_STEP_MIN_PX = 3.0
TAP_DOWN_STEP_MAX_PX = 10.0
TAP_UP_STEP_FRACTION = 0.022
TAP_MIN_HOLD_SECONDS = 0.045


@dataclass
class RuntimeConfig:
    mode: str
    camera: int
    instrument: str
    start_octave: int
    piano_octaves: int
    volume: int
    play_trigger: str
    camera_width: int
    camera_height: int
    tracking_scale: float
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
    learning_song: str
    learning_autoplay: bool
    performance_autoplay: bool
    learning_fit: str
    rush_e_autoplay_lock: bool


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run InstrumentalCV Air Piano.")
    parser.add_argument("--mode", choices=["air"], default="air")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--instrument", type=str, default="sounds")
    parser.add_argument("--octave", type=int, default=3)
    parser.add_argument("--piano-octaves", type=int, default=2)
    parser.add_argument("--volume", type=int, default=70)
    parser.add_argument("--play-trigger", choices=sorted(PLAY_TRIGGERS), default="precision")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--tracking-scale", type=float, default=0.55)
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
    parser.add_argument("--piano-alpha", type=float, default=0.78)
    parser.add_argument("--hover-cooldown", "--tap-cooldown", dest="hover_cooldown", type=float, default=0.15)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--dead-zone", type=int, default=3)
    parser.add_argument("--fadeout-ms", type=int, default=300)
    parser.add_argument("--metronome", action="store_true")
    parser.add_argument("--metronome-bpm", type=int, default=120)

    parser.add_argument("--learning-song", type=str, default="")
    parser.add_argument("--learning-autoplay", action="store_true")
    parser.add_argument("--performance-autoplay", action="store_true")
    parser.add_argument("--learning-fit", choices=["retune", "original"], default="retune")

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
        play_trigger=args.play_trigger,
        camera_width=clamp(args.camera_width, 320, 3840),
        camera_height=clamp(args.camera_height, 240, 2160),
        tracking_scale=clamp(args.tracking_scale, 0.45, 1.0),
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
        learning_song=args.learning_song,
        learning_autoplay=args.learning_autoplay,
        performance_autoplay=args.performance_autoplay,
        learning_fit=args.learning_fit,
        rush_e_autoplay_lock=is_rush_e_name(args.learning_song),
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
        self.active_note = None
        self.active_voice_id = None
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.released_voice_id = None
        self.last_trigger_time = 0.0
        self.tap_progress = 0.0

    def _clear_events(self):
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.released_voice_id = None

    def _release_active(self):
        if self.active_note is None:
            return
        self.note_off = True
        self.released_note = self.active_note
        self.released_voice_id = self.active_voice_id
        self.active_note = None
        self.active_voice_id = None
        self.is_held = False

    def has_active_state(self):
        return self.prev_key_note is not None or self.active_note is not None

    def set_active_voice(self, voice_id):
        if self.active_note is not None:
            self.active_voice_id = voice_id

    def update(self, current_key, current_time, **_kwargs):
        self._clear_events()
        self.current_key_note = current_key["note"] if current_key else None

        if self.current_key_note == self.prev_key_note:
            if self.active_note is not None:
                self.is_held = True
            self.prev_key_note = self.current_key_note
            return

        self._release_active()
        if self.current_key_note is not None:
            elapsed = current_time - self.last_trigger_time
            if elapsed >= self.cooldown_seconds:
                self.note_on = True
                self.is_held = True
                self.active_note = self.current_key_note
                self.active_voice_id = None
                self.last_trigger_time = current_time

        self.prev_key_note = self.current_key_note


class TapTracker:
    def __init__(self, finger_name, cooldown_seconds, precision=False):
        self.finger_name = finger_name
        self.cooldown_seconds = cooldown_seconds
        self.precision = precision
        self.prev_key_note = None
        self.current_key_note = None
        self.active_note = None
        self.active_voice_id = None
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.released_voice_id = None
        self.last_trigger_time = 0.0
        self.filtered_tap_y = None
        self.prev_filtered_tap_y = None
        self.rest_tap_y = None
        self.recent_tap_y = deque(maxlen=6)
        self.press_tap_y = None
        self.press_time = 0.0
        self.downstroke_armed = True
        self.tap_depth = 0.0
        self.tap_progress = 0.0
        self.motion_delta = 0.0
        self.prev_hand_angle = None

    def _clear_events(self):
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None
        self.released_voice_id = None

    def _release_active(self):
        if self.active_note is None:
            return
        self.note_off = True
        self.released_note = self.active_note
        self.released_voice_id = self.active_voice_id
        self.active_note = None
        self.active_voice_id = None
        self.press_tap_y = None
        self.is_held = False

    def _reset_motion(self):
        self.filtered_tap_y = None
        self.prev_filtered_tap_y = None
        self.rest_tap_y = None
        self.recent_tap_y.clear()
        self.press_tap_y = None
        self.downstroke_armed = True
        self.tap_depth = 0.0
        self.tap_progress = 0.0
        self.motion_delta = 0.0
        self.prev_hand_angle = None

    def has_active_state(self):
        return self.prev_key_note is not None or self.active_note is not None

    def set_active_voice(self, voice_id):
        if self.active_note is not None:
            self.active_voice_id = voice_id

    def update(self, current_key, current_time, tap_y=None, hand_span_px=120.0, screen_y=None, hand_angle=None):
        self._clear_events()
        self.current_key_note = current_key["note"] if current_key else None

        if tap_y is None or self.current_key_note is None:
            self._release_active()
            self._reset_motion()
            self.prev_key_note = self.current_key_note
            return

        tap_y = float(tap_y)
        hand_span_px = max(1.0, float(hand_span_px))
        press_threshold = clamp(hand_span_px * TAP_PRESS_FRACTION, TAP_PRESS_MIN_PX, TAP_PRESS_MAX_PX)
        release_threshold = max(6.0, press_threshold * TAP_RELEASE_FRACTION)
        down_step = clamp(hand_span_px * TAP_DOWN_STEP_FRACTION, TAP_DOWN_STEP_MIN_PX, TAP_DOWN_STEP_MAX_PX)
        up_step = clamp(hand_span_px * TAP_UP_STEP_FRACTION, 2.5, 8.0)
        if self.precision:
            press_threshold *= 1.16
            release_threshold *= 0.90
            down_step *= 1.12

        hand_rotation_delta = 0.0
        if hand_angle is not None:
            if self.prev_hand_angle is not None:
                hand_rotation_delta = abs(math.atan2(
                    math.sin(hand_angle - self.prev_hand_angle),
                    math.cos(hand_angle - self.prev_hand_angle),
                ))
            self.prev_hand_angle = hand_angle

        if self.filtered_tap_y is None:
            self.filtered_tap_y = tap_y
            self.prev_filtered_tap_y = tap_y
            self.rest_tap_y = tap_y
            self.recent_tap_y.append(tap_y)
            self.prev_key_note = self.current_key_note
            return

        self.filtered_tap_y = (self.filtered_tap_y * 0.45) + (tap_y * 0.55)
        self.motion_delta = self.filtered_tap_y - self.prev_filtered_tap_y
        self.recent_tap_y.append(self.filtered_tap_y)

        if self.rest_tap_y is None:
            self.rest_tap_y = self.filtered_tap_y

        if self.active_note is not None and self.current_key_note != self.active_note:
            self._release_active()
            self.downstroke_armed = False

        raw_depth = self.filtered_tap_y - self.rest_tap_y
        self.tap_depth = max(0.0, raw_depth)
        self.tap_progress = clamp(self.tap_depth / press_threshold, 0.0, 1.0)
        recent_high_y = min(self.recent_tap_y) if self.recent_tap_y else self.filtered_tap_y
        downstroke_travel = self.filtered_tap_y - recent_high_y

        if self.active_note is None:
            near_rest = self.tap_depth <= release_threshold
            moving_up = self.motion_delta <= 0.0
            if near_rest or moving_up:
                baseline_alpha = 0.10 if near_rest else 0.035
                self.rest_tap_y = (self.rest_tap_y * (1.0 - baseline_alpha)) + (self.filtered_tap_y * baseline_alpha)
                raw_depth = self.filtered_tap_y - self.rest_tap_y
                self.tap_depth = max(0.0, raw_depth)
                self.tap_progress = clamp(self.tap_depth / press_threshold, 0.0, 1.0)
            if self.tap_depth <= release_threshold:
                self.downstroke_armed = True

        elapsed = current_time - self.last_trigger_time
        travel_fraction = 0.82 if self.precision else 0.68
        intentional_downstroke = self.motion_delta >= down_step or downstroke_travel >= press_threshold * travel_fraction
        if self.precision and hand_rotation_delta > 0.075 and self.motion_delta < down_step * 1.5:
            intentional_downstroke = False
        if (
            self.active_note is None
            and self.downstroke_armed
            and elapsed >= self.cooldown_seconds
            and self.tap_depth >= press_threshold
            and intentional_downstroke
        ):
            self.note_on = True
            self.is_held = True
            self.active_note = self.current_key_note
            self.active_voice_id = None
            self.press_tap_y = self.filtered_tap_y
            self.press_time = current_time
            self.downstroke_armed = False
            self.last_trigger_time = current_time

        if self.active_note is not None:
            self.is_held = True
            held_long_enough = current_time - self.press_time >= TAP_MIN_HOLD_SECONDS
            lifted_near_rest = self.tap_depth <= release_threshold
            lifted_from_press = (
                self.press_tap_y is not None
                and self.filtered_tap_y <= self.press_tap_y - release_threshold
                and self.motion_delta <= -up_step
            )
            if held_long_enough and (lifted_near_rest or lifted_from_press):
                self._release_active()
                if lifted_near_rest:
                    self.downstroke_armed = True

        self.prev_key_note = self.current_key_note
        self.prev_filtered_tap_y = self.filtered_tap_y


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
        self.voice_channels = {}
        self.next_voice_id = 1
        self.lock = threading.RLock()
        self.current_instrument = "sounds"
        self.last_error = None

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(192)

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

        with self.lock:
            self.sounds = loaded_sounds
            self.channels.clear()
            self.voice_channels.clear()
        loaded = len(self.sounds)
        if loaded == 0:
            self.last_error = f"No WAV samples found in {os.path.basename(sounds_dir)}"
            print(f"ERROR: {self.last_error}. Run setup_sounds.py first.")
        else:
            print(f"Loaded {loaded} samples")
        return loaded

    def _cleanup_finished_locked(self):
        for voice_id, (_note, channel) in list(self.voice_channels.items()):
            if not channel.get_busy():
                self.voice_channels.pop(voice_id, None)
        for note_name, voices in list(self.channels.items()):
            alive = [(voice_id, channel) for voice_id, channel in voices if voice_id in self.voice_channels and channel.get_busy()]
            if alive:
                self.channels[note_name] = alive
            else:
                self.channels.pop(note_name, None)

    def note_on(self, note_name, velocity=100, volume_scale=1.0):
        with self.lock:
            sound = self.sounds.get(note_name)
            if sound is None:
                return None
            self._cleanup_finished_locked()
            channel = pygame.mixer.find_channel(force=True)
            if channel is None:
                return None
            volume = clamp((velocity / 127.0) * self.max_volume * volume_scale, 0.0, 1.0)
            channel.set_volume(volume, volume)
            channel.play(sound)
            voice_id = self.next_voice_id
            self.next_voice_id += 1
            self.voice_channels[voice_id] = (note_name, channel)
            self.channels.setdefault(note_name, []).append((voice_id, channel))
            return voice_id

    def note_off(self, note_name, fadeout_ms=300, voice_id=None):
        with self.lock:
            if voice_id is not None:
                voice = self.voice_channels.pop(voice_id, None)
                if voice is None:
                    return
                _note, channel = voice
                if channel.get_busy():
                    channel.fadeout(fadeout_ms)
                voices = self.channels.get(note_name, [])
                self.channels[note_name] = [(vid, ch) for vid, ch in voices if vid != voice_id]
                if not self.channels.get(note_name):
                    self.channels.pop(note_name, None)
                return

            voices = self.channels.pop(note_name, [])
            for vid, channel in voices:
                self.voice_channels.pop(vid, None)
                if channel.get_busy():
                    channel.fadeout(fadeout_ms)

    def all_notes_off(self, fadeout_ms=150):
        with self.lock:
            voices = list(self.voice_channels.items())
            self.voice_channels.clear()
            self.channels.clear()
        for _voice_id, (_note_name, channel) in voices:
            if channel.get_busy():
                channel.fadeout(fadeout_ms)

    def cleanup(self):
        self.all_notes_off(50)
        pygame.mixer.quit()


class AutoplayScheduler:
    def __init__(
        self,
        song,
        sound_engine,
        fadeout_ms=120,
        batch_seconds=AUTOPLAY_BATCH_SECONDS,
        max_audio_notes_per_batch=AUTOPLAY_MAX_AUDIO_NOTES_PER_BATCH,
    ):
        self.song = song
        self.sound_engine = sound_engine
        self.fadeout_ms = fadeout_ms
        self.batch_seconds = batch_seconds
        self.max_audio_notes_per_batch = max_audio_notes_per_batch
        self.stop_event = threading.Event()
        self.thread = None
        self.active_voices = {}
        self.active_note_counts = {}
        self.started_notes = 0
        self.lock = threading.RLock()
        self.start_perf = None
        self.start_beat = 0.0

    def start(self, current_beat=0.0):
        self.stop()
        self.stop_event.clear()
        with self.lock:
            self.active_voices.clear()
            self.active_note_counts.clear()
            self.started_notes = 0
            self.start_perf = time.perf_counter()
            self.start_beat = current_beat
        self.thread = threading.Thread(target=self._run, args=(current_beat,), daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=0.35)
        self.thread = None
        with self.lock:
            voices = list(self.active_voices.items())
            self.active_voices.clear()
            self.active_note_counts.clear()
        for _release_id, (note_name, voice_id) in voices:
            self.sound_engine.note_off(note_name, self.fadeout_ms, voice_id=voice_id)

    def active_note_names(self):
        with self.lock:
            return {note_name for note_name, count in self.active_note_counts.items() if count > 0}

    def progress(self):
        with self.lock:
            started = self.started_notes
            start_perf = self.start_perf
            start_beat = self.start_beat
        current_beat = start_beat
        if start_perf is not None:
            current_beat += self.song.seconds_to_beats(max(0.0, time.perf_counter() - start_perf))
        return started, current_beat

    def _wait_until(self, target_time, start_perf):
        while not self.stop_event.is_set():
            remaining = target_time - (time.perf_counter() - start_perf)
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.008))
        return False

    def _batch_bucket(self, seconds):
        return round(seconds / self.batch_seconds)

    def _add_active_note(self, note_name, count=1):
        with self.lock:
            self.active_note_counts[note_name] = self.active_note_counts.get(note_name, 0) + count

    def _release_active_note(self, note_name, count=1):
        with self.lock:
            remaining = self.active_note_counts.get(note_name, 0) - count
            if remaining > 0:
                self.active_note_counts[note_name] = remaining
            else:
                self.active_note_counts.pop(note_name, None)

    def _run(self, current_beat):
        notes = self.song.notes
        note_count = len(notes)
        note_index = 0
        while note_index < note_count and notes[note_index].end_time < current_beat:
            note_index += 1

        with self.lock:
            start_perf = self.start_perf or time.perf_counter()
            self.start_perf = start_perf
        release_heap = []
        release_id = 0
        time_epsilon = max(0.001, self.batch_seconds / 2)

        while not self.stop_event.is_set() and (note_index < note_count or release_heap):
            next_start_time = float("inf")
            if note_index < note_count:
                next_start_time = max(0.0, self.song.beats_to_seconds(notes[note_index].time - current_beat))
            next_release_time = release_heap[0][0] if release_heap else float("inf")
            event_time = min(next_start_time, next_release_time)

            if not self._wait_until(event_time, start_perf):
                break

            if next_start_time <= next_release_time + time_epsilon and note_index < note_count:
                batch = []
                batch_key = self._batch_bucket(next_start_time)
                while note_index < note_count:
                    note = notes[note_index]
                    start_seconds = max(0.0, self.song.beats_to_seconds(note.time - current_beat))
                    if self._batch_bucket(start_seconds) != batch_key:
                        break
                    batch.append((note_index, note, start_seconds))
                    note_index += 1

                with self.lock:
                    self.started_notes += len(batch)

                grouped = {}
                for _idx, note, _start_seconds in batch:
                    entry = grouped.get(note.note)
                    if entry is None:
                        grouped[note.note] = [1, note]
                    else:
                        entry[0] += 1
                        if note.duration > entry[1].duration:
                            entry[1] = note

                playable_groups = list(grouped.items())[: self.max_audio_notes_per_batch]
                chord_size = max(1, len(playable_groups))
                for note_name, (count, note) in playable_groups:
                    density_gain = min(2.2, 1.0 + math.log1p(count) * 0.18)
                    volume_scale = min(1.0, density_gain / max(1.0, chord_size ** 0.45))
                    voice_id = self.sound_engine.note_on(note_name, volume_scale=volume_scale)
                    self._add_active_note(note_name)
                    release_id += 1
                    if voice_id is not None:
                        with self.lock:
                            self.active_voices[release_id] = (note_name, voice_id)
                    release_seconds = max(0.0, self.song.beats_to_seconds(note.end_time - current_beat))
                    heapq.heappush(release_heap, (release_seconds, release_id, note_name, voice_id))

            while release_heap and release_heap[0][0] <= event_time + time_epsilon:
                _release_seconds, done_release_id, note_name, voice_id = heapq.heappop(release_heap)
                self._release_active_note(note_name)
                with self.lock:
                    voice = self.active_voices.pop(done_release_id, None)
                if voice is not None:
                    note_name, voice_id = voice
                    self.sound_engine.note_off(note_name, self.fadeout_ms, voice_id=voice_id)


def is_finger_extended(hand_landmarks, finger_name):  #only plays if tip joint is above the pip joint
    joints = FINGER_JOINTS[finger_name]
    tip = hand_landmarks.landmark[joints["tip"]]
    pip = hand_landmarks.landmark[joints["pip"]]

    if finger_name == "thumb":
        wrist = hand_landmarks.landmark[0]
        return abs(tip.x - wrist.x) > abs(pip.x - wrist.x)
    return tip.y < pip.y


def landmark_pixel(hand_landmarks, landmark_id, frame_w, frame_h):
    landmark = hand_landmarks.landmark[landmark_id]
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def palm_anchor_pixel(hand_landmarks, frame_w, frame_h):
    points = [landmark_pixel(hand_landmarks, landmark_id, frame_w, frame_h) for landmark_id in PALM_ANCHOR_IDS]
    avg_x = int(sum(point[0] for point in points) / len(points))
    avg_y = int(sum(point[1] for point in points) / len(points))
    return avg_x, avg_y


def estimate_hand_span_px(hand_landmarks, frame_w, frame_h):
    wrist_x, wrist_y = landmark_pixel(hand_landmarks, 0, frame_w, frame_h)
    palm_points = [landmark_pixel(hand_landmarks, landmark_id, frame_w, frame_h) for landmark_id in PALM_ANCHOR_IDS[1:]]
    distances = [math.hypot(point[0] - wrist_x, point[1] - wrist_y) for point in palm_points]
    return max(45.0, max(distances, default=90.0))


def hand_axis_from_points(wrist_x, wrist_y, middle_mcp_x, middle_mcp_y):
    axis_x = middle_mcp_x - wrist_x
    axis_y = middle_mcp_y - wrist_y
    axis_len = math.hypot(axis_x, axis_y)
    if axis_len < 1.0:
        return 0.0, -1.0, -math.pi / 2
    unit_x = axis_x / axis_len
    unit_y = axis_y / axis_len
    return unit_x, unit_y, math.atan2(unit_y, unit_x)


def local_finger_tap_y(tip_x, tip_y, base_x, base_y, hand_axis_x, hand_axis_y):
    # Measures finger flexion in hand-local space, so wrist rotation is mostly ignored.
    finger_x = tip_x - base_x
    finger_y = tip_y - base_y
    return -((finger_x * hand_axis_x) + (finger_y * hand_axis_y))


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


def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    layer = np.empty_like(roi)
    layer[:] = color
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


def draw_piano(frame, white_keys, black_keys, alpha, show_note_labels):
    if not white_keys:
        return

    piano_top = white_keys[0]["y1"]
    piano_bottom = max(key["y2"] for key in white_keys)
    frame_h, frame_w = frame.shape[:2]
    blend_rect(frame, 0, max(0, piano_top - 18), frame_w, min(frame_h, piano_bottom + 12), COLOR_PANEL, 0.28)
    blend_rect(frame, 0, max(0, piano_top - 6), frame_w, piano_top + 3, COLOR_ACCENT, 0.20)
    region = frame[piano_top:piano_bottom, :]
    overlay = region.copy()

    for key in white_keys:
        active = key["color"] == COLOR_GREEN
        hovered = key["color"] != key["color_default"]
        draw_y1 = key["y1"] + 5 if active else key["y1"]
        draw_color = (94, 245, 150) if active else ((255, 238, 204) if hovered else (244, 247, 248))
        x1, x2 = key["x1"], key["x2"]
        y1, y2 = draw_y1 - piano_top, key["y2"] - piano_top
        cv2.rectangle(overlay, (x1, y1), (x2, y2), draw_color, -1)
        cv2.rectangle(overlay, (x1 + 2, y1 + 4), (x2 - 2, min(y2, y1 + 18)), (255, 255, 255), -1)
        cv2.line(overlay, (x1, y1), (x1, y2), (142, 151, 165), 1)
        cv2.line(overlay, (x2, y1), (x2, y2), (142, 151, 165), 1)
        if active:
            cv2.rectangle(overlay, (x1 + 2, y1 + 2), (x2 - 2, y2 - 3), COLOR_ACCENT, 2)

    for key in black_keys:
        active = key["color"] == COLOR_GREEN
        hovered = key["color"] != key["color_default"]
        draw_y1 = key["y1"] + 4 if active else key["y1"]
        draw_color = (82, 220, 128) if active else ((162, 93, 220) if hovered else (12, 15, 22))
        x1, x2 = key["x1"], key["x2"]
        y1, y2 = draw_y1 - piano_top, key["y2"] - piano_top
        cv2.rectangle(overlay, (x1, y1), (x2, y2), draw_color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (38, 45, 60), 1)
        cv2.line(overlay, (x1 + 2, y1 + 2), (x2 - 2, y1 + 2), (92, 102, 118), 1)
        if active:
            cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 2), COLOR_ACCENT, 2)

    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)

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

    cv2.line(frame, (0, piano_top), (frame.shape[1], piano_top), COLOR_ACCENT, 2)
    cv2.line(frame, (0, piano_bottom), (frame.shape[1], piano_bottom), COLOR_PANEL_ALT, 1)


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


def mark_active_song_keys(all_keys, active_notes):
    if not active_notes:
        return
    active_note_names = {note if isinstance(note, str) else note.note for note in active_notes}
    for key in all_keys:
        if key["note"] in active_note_names:
            key["color"] = COLOR_GREEN


def draw_autoplay_hud(frame, song, scheduler):
    if scheduler is None:
        return
    h, w = frame.shape[:2]
    started, current_beat = scheduler.progress()
    total_notes = max(1, len(song.notes))
    total_beats = max(song.duration_beats, 1.0)
    progress = min(max(current_beat / total_beats, 0.0), 1.0)
    panel_w = min(w - 24, 520)
    x1 = 12
    y1 = 96
    x2 = x1 + panel_w
    y2 = y1 + 58
    blend_rect(frame, x1, y1, x2, y2, COLOR_PANEL, 0.78)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PANEL_ALT, 1)
    cv2.line(frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), COLOR_ACCENT, 1)
    title = song.title
    if len(title) > 42:
        title = title[:39] + "..."
    cv2.putText(frame, f"AUTOPLAYER  {title}", (x1 + 12, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47, COLOR_ACCENT, 1, cv2.LINE_AA)
    note_text = f"{started:,}/{total_notes:,} notes"
    cv2.putText(frame, note_text, (x1 + 12, y1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    bar_x1 = x1 + 170
    bar_y = y1 + 37
    bar_x2 = x2 - 14
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 8), COLOR_PANEL_ALT, -1)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + int((bar_x2 - bar_x1) * progress), bar_y + 8), COLOR_ACCENT, -1)


def draw_label(frame, text, x, y, scale=0.55, color=COLOR_WHITE, anchor="left"):
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x = 10
    pad_y = 7
    if anchor == "right":
        x = x - size[0] - (pad_x * 2)
    x1 = x - pad_x
    y1 = y - size[1] - pad_y
    x2 = x + size[0] + pad_x
    y2 = y + baseline + pad_y
    blend_rect(frame, x1, y1, x2, y2, COLOR_PANEL, 0.78)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PANEL_ALT, 1)
    cv2.line(frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), color, 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(frame, config, current_octave, hovered_notes, fps, sound_engine, metronome):
    frame_h, frame_w = frame.shape[:2]
    blend_rect(frame, 0, 0, frame_w, 92, COLOR_PANEL, 0.18)
    if config.show_fps:
        fps_color = COLOR_GREEN if fps >= 20 else COLOR_RED
        draw_label(frame, f"FPS {int(fps)}", 12, 34, 0.6, fps_color)

    if config.learning_song:
        if sound_engine.last_error:
            draw_label(frame, sound_engine.last_error, 12, frame_h - 18, 0.45, COLOR_RED)
        else:
            if config.rush_e_autoplay_lock:
                hint = "Autoplay locked (Rush E)  |  Q quit  |  M metronome  |  < > octave"
            elif config.learning_autoplay:
                hint = "P  disable autoplay  |  Q quit  |  M metronome  |  < > octave"
            else:
                hint = "P  enable autoplay  |  Q quit  |  M metronome  |  < > octave"
            draw_label(frame, hint, 12, frame_h - 18, 0.43, COLOR_TEXT_DIM)
        metronome.update(frame, frame_w)
        return

    if hovered_notes:
        notes_text = "Playing: " + ", ".join(hovered_notes[:8])
        draw_label(frame, notes_text, 12, 68, 0.52, COLOR_ACCENT)
    else:
        trigger_label = PLAY_TRIGGER_LABELS.get(config.play_trigger, "Tap")
        draw_label(frame, f"{trigger_label} ready", 12, 68, 0.52, COLOR_TEXT_DIM)

    instrument_label = INSTRUMENTS.get(sound_engine.current_instrument, sound_engine.current_instrument)
    draw_label(frame, "AIR PIANO", frame_w - 12, 34, 0.55, COLOR_ACCENT, anchor="right")
    draw_label(
        frame,
        f"C{current_octave}-B{current_octave + config.piano_octaves - 1}  {instrument_label}",
        frame_w - 12,
        68,
        0.45,
        COLOR_TEXT_DIM,
        anchor="right",
    )

    if sound_engine.last_error:
        draw_label(frame, sound_engine.last_error, 12, frame_h - 18, 0.45, COLOR_RED)
    else:
        draw_label(frame, "Q quit | P autoplay | M metronome | arrows octave | 1-5 sounds", 12, frame_h - 18, 0.43, COLOR_TEXT_DIM)

    metronome.update(frame, frame_w)


def get_camera_backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def configure_camera_capture(cap, width, height, fps=30):
    if os.name == "nt":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def release_tracker_note(tracker, sound_engine, fadeout_ms, learning_renderer=None):
    if tracker.released_note is None:
        return
    if tracker.released_voice_id is not None:
        sound_engine.note_off(tracker.released_note, fadeout_ms, voice_id=tracker.released_voice_id)
    if learning_renderer is not None:
        learning_renderer.check_release(tracker.released_note)


def release_inactive_trackers(finger_trackers, active_tracker_keys, current_time, sound_engine, fadeout_ms, learning_renderer=None):
    for tracker_key, tracker in finger_trackers.items():
        if tracker_key in active_tracker_keys or not tracker.has_active_state():
            continue
        tracker.update(None, current_time)
        if tracker.note_off:
            release_tracker_note(tracker, sound_engine, fadeout_ms, learning_renderer)


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

    if key_lower == ord("p") and config.learning_song:
        if config.rush_e_autoplay_lock:
            config.learning_autoplay = True
            sound_engine.all_notes_off(80)
            print("Rush E autoplay lock: ON. Your fingers lost custody.")
            return current_octave, False
        config.learning_autoplay = not config.learning_autoplay
        sound_engine.all_notes_off(80)
        label = "Performance autoplay" if config.performance_autoplay else "Learning autoplay"
        print(f"{label}: {'ON' if config.learning_autoplay else 'OFF'}")
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
        model_complexity=0,
        max_num_hands=config.max_hands,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )

    cap = cv2.VideoCapture(config.camera, get_camera_backend())
    configure_camera_capture(cap, config.camera_width, config.camera_height)

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
        learning_renderer = None
        autoplay_audio_song = None
        performance_autoplay = False

        def build_key_map():
            return {key["note"]: key for key in all_keys}

        def rebuild_piano():
            nonlocal white_keys, black_keys, all_keys, frame_w, frame_h, learning_renderer
            white_keys, black_keys = generate_piano_keys(frame_w, frame_h, config)
            all_keys = black_keys + white_keys
            if learning_renderer is not None:
                learning_renderer.key_map = build_key_map()
                learning_renderer.piano_top_y = white_keys[0]["y1"] if white_keys else int(frame_h * 0.3)

        # learning mode setup
        if config.learning_song:
            try:
                source_song = load_song(config.learning_song)
                if is_rush_e_name(config.learning_song) or is_rush_e_name(source_song.title):
                    config.rush_e_autoplay_lock = True
                    config.learning_autoplay = True
                    print("Rush E detected: autoplay locked ON. Your fingers lost custody.")
                song = source_song
                key_map = build_key_map()
                has_unplayable_notes = any(note.note not in key_map for note in song.notes)
                if has_unplayable_notes and config.learning_fit == "retune":
                    song = remap_song(song, config.piano_octaves, config.start_octave)
                    key_map = build_key_map()
                    print(
                        f"Retuned '{source_song.title}' to fit C{config.start_octave}-"
                        f"B{config.start_octave + config.piano_octaves - 1}."
                    )
                elif has_unplayable_notes:
                    print(
                        "Warning: selected song has notes outside the visible piano range. "
                        "Some notes may not be playable or visible."
                    )
                autoplay_audio_song = song
                performance_autoplay = bool(
                    config.learning_autoplay
                    and (config.performance_autoplay or len(song.notes) >= PERFORMANCE_AUTOPLAY_NOTE_THRESHOLD)
                )
                if performance_autoplay:
                    config.performance_autoplay = True
                if performance_autoplay:
                    print(f"Autoplayer performance mode: '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM)")
                else:
                    piano_top_y = white_keys[0]["y1"] if white_keys else int(frame_h * 0.3)
                    learning_renderer = FallingNoteRenderer(
                        song=song,
                        key_map=key_map,
                        piano_top_y=piano_top_y,
                    )
                    print(f"Learning mode: '{song.title}' ({len(song.notes)} notes, {song.bpm} BPM)")
            except Exception as exc:
                print(f"Failed to load learning song: {exc}")
                learning_renderer = None

        finger_trackers = {}

        def get_tracker(tracker_key, finger_name):
            if tracker_key not in finger_trackers:
                if config.play_trigger == "hover":
                    finger_trackers[tracker_key] = HoverTracker(finger_name, config.hover_cooldown)
                else:
                    finger_trackers[tracker_key] = TapTracker(
                        finger_name,
                        config.hover_cooldown,
                        precision=config.play_trigger == "precision",
                    )
            return finger_trackers[tracker_key]

        prev_time = 0.0
        note_trail = NoteTrail()
        smoother = PositionSmoother(window=config.smoothing_window)
        sound_engine = SoundEngine(max_volume=config.volume / 100.0)
        sound_engine.load_instrument(config.instrument)
        metronome = Metronome(bpm=config.metronome_bpm, enabled=config.metronome)
        learning_started = False
        autoplay_scheduler = None
        autoplay_played = set()
        autoplay_active = {}

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

            if learning_renderer is not None and not learning_started:
                learning_renderer.start()
                learning_started = True

            current_time = time.time()
            if config.tracking_scale < 0.99:
                tracking_frame = cv2.resize(
                    frame,
                    None,
                    fx=config.tracking_scale,
                    fy=config.tracking_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                tracking_frame = frame
            rgb_tracking_frame = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
            rgb_tracking_frame.flags.writeable = False
            result = hands.process(rgb_tracking_frame)
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

                    palm_x, palm_y = palm_anchor_pixel(hand_landmarks, frame_w, frame_h)
                    palm_x, palm_y = smoother.smooth(f"{hand_idx}_{hand_label}_palm", palm_x, palm_y)
                    wrist_x, wrist_y = landmark_pixel(hand_landmarks, 0, frame_w, frame_h)
                    wrist_x, wrist_y = smoother.smooth(f"{hand_idx}_{hand_label}_wrist_axis", wrist_x, wrist_y)
                    middle_mcp_x, middle_mcp_y = landmark_pixel(hand_landmarks, 9, frame_w, frame_h)
                    middle_mcp_x, middle_mcp_y = smoother.smooth(
                        f"{hand_idx}_{hand_label}_middle_mcp_axis",
                        middle_mcp_x,
                        middle_mcp_y,
                    )
                    hand_axis_x, hand_axis_y, hand_angle = hand_axis_from_points(
                        wrist_x,
                        wrist_y,
                        middle_mcp_x,
                        middle_mcp_y,
                    )
                    hand_span_px = estimate_hand_span_px(hand_landmarks, frame_w, frame_h)

                    for finger_name, tip_id in FINGERTIP_IDS.items():
                        tracker_key = f"{hand_idx}_{hand_label}_{finger_name}"
                        active_tracker_keys.add(tracker_key)
                        tracker = get_tracker(tracker_key, finger_name)

                        if config.play_trigger == "hover" and not is_finger_extended(hand_landmarks, finger_name):
                            tracker.update(None, current_time)
                            if tracker.note_off:
                                release_tracker_note(tracker, sound_engine, config.fadeout_ms, learning_renderer)
                            continue

                        tip = hand_landmarks.landmark[tip_id]
                        tip_x, tip_y = int(tip.x * frame_w), int(tip.y * frame_h)
                        tip_x, tip_y = smoother.smooth(tracker_key, tip_x, tip_y)
                        hovered_key = detect_finger_on_key(tip_x, tip_y, all_keys, config.dead_zone_px)
                        base_id = FINGER_JOINTS[finger_name]["mcp"]
                        base_x, base_y = landmark_pixel(hand_landmarks, base_id, frame_w, frame_h)
                        base_x, base_y = smoother.smooth(f"{tracker_key}_base", base_x, base_y)
                        tap_y = local_finger_tap_y(tip_x, tip_y, base_x, base_y, hand_axis_x, hand_axis_y)
                        tracker.update(
                            hovered_key,
                            current_time,
                            tap_y=tap_y,
                            hand_span_px=hand_span_px,
                            screen_y=tip_y - palm_y,
                            hand_angle=hand_angle,
                        )

                        dot_color = COLOR_GREEN if tracker.note_on or tracker.is_held else FINGER_COLORS[finger_name]
                        cv2.circle(frame, (tip_x, tip_y), 18, dot_color, 1, cv2.LINE_AA)
                        cv2.circle(frame, (tip_x, tip_y), 10, dot_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (tip_x, tip_y), 4, COLOR_WHITE, -1, cv2.LINE_AA)

                        if hovered_key is not None:
                            if tracker.is_held or tracker.note_on:
                                hovered_key["color"] = COLOR_GREEN
                            else:
                                hovered_key["color"] = hovered_key["color_hover"]
                            if tracker.is_held or tracker.note_on:
                                hovered_notes.append(hovered_key["note"])
                            key_center_x = (hovered_key["x1"] + hovered_key["x2"]) // 2
                            cv2.line(frame, (tip_x, tip_y), (key_center_x, hovered_key["y1"]), dot_color, 2, cv2.LINE_AA)

                        if tracker.note_on and hovered_key is not None:
                            note = hovered_key["note"]
                            voice_id = sound_engine.note_on(note)
                            tracker.set_active_voice(voice_id)
                            if config.show_note_trail:
                                note_trail.add(note)
                            hit_ok = True
                            if learning_renderer is not None:
                                hit_ok = learning_renderer.check_hit(note)
                                if not hit_ok:
                                    hovered_key["color"] = COLOR_RED
                            key_cx = (hovered_key["x1"] + hovered_key["x2"]) // 2
                            key_cy = (hovered_key["y1"] + hovered_key["y2"]) // 2
                            feedback_color = COLOR_GREEN if hit_ok else COLOR_RED
                            cv2.circle(frame, (key_cx, key_cy), 22, feedback_color, 2, cv2.LINE_AA)
                            cv2.circle(frame, (key_cx, key_cy), 36, feedback_color, 1, cv2.LINE_AA)

                        if tracker.note_off:
                            release_tracker_note(tracker, sound_engine, config.fadeout_ms, learning_renderer)

                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * frame_w), int(wrist.y * frame_h)
                    cv2.putText(frame, hand_label, (wrist_x - 30, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)

            release_inactive_trackers(finger_trackers, active_tracker_keys, current_time, sound_engine, config.fadeout_ms, learning_renderer)

            if (
                performance_autoplay
                and config.learning_autoplay
                and autoplay_scheduler is None
                and autoplay_audio_song is not None
            ):
                autoplay_scheduler = AutoplayScheduler(
                    autoplay_audio_song,
                    sound_engine,
                    fadeout_ms=min(config.fadeout_ms, 80),
                )
                autoplay_played.clear()
                autoplay_active.clear()
                autoplay_scheduler.start(current_beat=0.0)
            elif (
                not performance_autoplay
                and learning_renderer is not None
                and config.learning_autoplay
                and not learning_renderer.is_counting_in()
                and autoplay_scheduler is None
                and autoplay_audio_song is not None
            ):
                autoplay_scheduler = AutoplayScheduler(
                    autoplay_audio_song,
                    sound_engine,
                    fadeout_ms=min(config.fadeout_ms, 120),
                )
                autoplay_start_beat = max(0.0, learning_renderer.current_beat())
                learning_renderer.seek_autoplay(autoplay_start_beat)
                autoplay_played.clear()
                autoplay_active.clear()
                autoplay_scheduler.start(current_beat=autoplay_start_beat)
            elif (not config.learning_autoplay or (learning_renderer is None and not performance_autoplay)) and autoplay_scheduler is not None:
                autoplay_scheduler.stop()
                autoplay_scheduler = None

            if performance_autoplay and autoplay_scheduler is not None:
                autoplay_active_notes = autoplay_scheduler.active_note_names()
            elif learning_renderer is not None and config.learning_autoplay and not learning_renderer.is_counting_in():
                current_beat = learning_renderer.current_beat()
                visual_release_grace = learning_renderer.song.seconds_to_beats(0.08)
                for idx, note in learning_renderer.due_autoplay_notes(current_beat, autoplay_played):
                    autoplay_played.add(idx)
                    autoplay_active[idx] = note
                    learning_renderer.mark_autoplay_hit(idx)
                for idx, note in list(autoplay_active.items()):
                    if current_beat >= note.end_time + visual_release_grace:
                        learning_renderer.check_release(note.note)
                        autoplay_active.pop(idx, None)
                autoplay_active_notes = autoplay_active.values()
            elif autoplay_active:
                for idx, note in list(autoplay_active.items()):
                    if learning_renderer is not None:
                        learning_renderer.check_release(note.note)
                    autoplay_active.pop(idx, None)
                autoplay_active_notes = []
            else:
                autoplay_active_notes = []

            mark_active_song_keys(all_keys, autoplay_active_notes)

            piano_alpha = max(config.piano_alpha, LEARNING_MIN_PIANO_ALPHA) if learning_renderer is not None else config.piano_alpha
            draw_piano(frame, white_keys, black_keys, piano_alpha, config.show_note_labels)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            if config.show_note_trail:
                note_trail.draw(frame)
            draw_hud(frame, config, config.start_octave, hovered_notes, fps, sound_engine, metronome)
            if performance_autoplay and autoplay_audio_song is not None:
                draw_autoplay_hud(frame, autoplay_audio_song, autoplay_scheduler)

            # Learning visuals should sit above the normal play HUD.
            if learning_renderer is not None:
                learning_renderer.update_and_draw(frame, autoplay_active=config.learning_autoplay)
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
            if 'autoplay_scheduler' in locals() and autoplay_scheduler is not None:
                autoplay_scheduler.stop()
            sound_engine.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Piano closed.")


if __name__ == "__main__":
    sys.exit(main())
