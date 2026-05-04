import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from roast_lines import LEARNING_HUD_ROASTS, AUTOPLAY_ROASTS
from song import Song, SongNote, is_rush_e_name

COLOR_UPCOMING_WHITE = (178, 255, 198)      # soft green for white key notes
COLOR_UPCOMING_BLACK = (255, 150, 220)      # pink-purple for black key notes
COLOR_HIT = (116, 255, 158)                 # green when hit correctly
COLOR_MISSED = (92, 92, 255)                # red when missed
COLOR_WRONG = (58, 76, 255)                 # immediate wrong-key flash
COLOR_HOLD_BG = (36, 42, 54)                # hold bar background
COLOR_HOLD_FILL = (90, 255, 180)            # hold bar fill (green)
COLOR_HINT_GLOW = (80, 218, 255)            # glow on target key before note arrives
COLOR_NEXT_ARROW = (255, 224, 72)
COLOR_ACTIVE_HOLD = (110, 255, 190)
COLOR_HIT_ZONE = (255, 224, 72)
COLOR_SPARK = (255, 255, 255)               # spark color
COLOR_SCORE_TEXT = (240, 240, 240)
COLOR_PROGRESS_BG = (34, 42, 56)
COLOR_PROGRESS_FILL = (255, 175, 54)
COLOR_PANEL = (18, 22, 31)
COLOR_PANEL_BORDER = (54, 69, 90)

def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    color_layer = np.empty_like(roi)
    color_layer[:] = color
    cv2.addWeighted(color_layer, alpha, roi, 1.0 - alpha, 0, roi)

@dataclass
class Spark:
    x: int
    y: int
    spawn_time: float
    vx: float = 0.0
    vy: float = 0.0 
    lifetime: float = 0.45


@dataclass
class KeyFlash:
    note: str
    color: Tuple[int, int, int]
    spawn_time: float
    lifetime: float = 0.55
    

class FallingNoteRenderer:
    
    def __init__(
        self,
        song: Song,
        key_map: Dict[str, dict],
        piano_top_y: int,
        look_ahead_seconds: float = 3.0,
        hit_window_ms: float = 250,
        count_in_beats: int = 4,
    ):
        self.song = song
        self.key_map = key_map              # {note_name: {"x1", "x2", "y1", "y2", "is_black", ...}}
        self.piano_top_y = piano_top_y      # y coordinate where piano starts
        self.look_ahead_seconds = look_ahead_seconds
        self.hit_window_ms = hit_window_ms
        self.count_in_beats = count_in_beats
        self.note_times = [note.time for note in self.song.notes]
        self._next_pending_cursor = 0
        self._autoplay_cursor = 0

        self.start_time: Optional[float] = None
        self.paused = False
        self.pause_offset = 0.0             # accumulated paused time
        self.pause_start: Optional[float] = None
        self.finished = False

        # freeze time when waiting
        self.wait_frozen = False
        self.wait_freeze_time: Optional[float] = None

        # tracking
        self.hit_notes: Set[int] = set()        # indices of correctly hit notes
        self.missed_notes: Set[int] = set()     # indices of missed notes
        self.active_holds: Dict[str, int] = {}  # note_name note index being held
        self.hold_start_times: Dict[int, float] = {}   # note_index when hold started
        self.combo = 0
        self.max_combo = 0
        self.wrong_hits = 0
        self.feedback_text = ""
        self.feedback_color = COLOR_SCORE_TEXT
        self.feedback_time = 0.0
        self.feedback_lifetime = 0.8
        self.hold_release_grace_beats = 0.25
        self.chord_time_tolerance_beats = 0.03
        self.rush_e_mode = is_rush_e_name(song.title)

        # particles
        self.sparks: List[Spark] = []
        self.key_flashes: List[KeyFlash] = []
        self._rng = np.random.default_rng()
        

    def start(self):
        count_in_seconds = self.song.beats_to_seconds(self.count_in_beats)
        self.start_time = time.time() + count_in_seconds
        self.finished = False
        self.wait_frozen = False
        self.wait_freeze_time = None
        self.pause_offset = 0.0
        self.pause_start = None
        self.paused = False
        self.hit_notes.clear()
        self.missed_notes.clear()
        self.active_holds.clear()
        self.hold_start_times.clear()
        self.combo = 0
        self.max_combo = 0
        self.wrong_hits = 0
        self.feedback_text = ""
        self.sparks.clear()
        self.key_flashes.clear()
        self._next_pending_cursor = 0
        self._autoplay_cursor = 0
    
    def _elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.wait_frozen and self.wait_freeze_time is not None:
            raw = self.wait_freeze_time - self.start_time - self.pause_offset
        elif self.paused and self.pause_start is not None:
            raw = self.pause_start - self.start_time - self.pause_offset
        else:
            raw = time.time() - self.start_time - self.pause_offset
        return raw

    def _hint_seconds(self) -> float:
        return 1.6
    
    def _current_beat(self) -> float:
        return self.song.seconds_to_beats(self._elapsed_seconds())

    def current_beat(self) -> float:
        return self._current_beat()
    
    def toggle_pause(self):
        if self.paused:
            self.pause_offset += time.time() - self.pause_start
            self.pause_start = None
            self.paused = False
        else:
            self.pause_start = time.time()
            self.paused = True
            
    def is_counting_in(self) -> bool:
        return self.start_time is not None and time.time() < self.start_time

    def _count_in_remaining(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(0.0, self.start_time - time.time())

    def _set_feedback(self, text: str, color: Tuple[int, int, int], lifetime: float = 0.8):
        self.feedback_text = text
        self.feedback_color = color
        self.feedback_time = time.time()
        self.feedback_lifetime = lifetime

    def _flash_key(self, note_name: str, color: Tuple[int, int, int], lifetime: float = 0.55):
        if note_name in self.key_map:
            self.key_flashes.append(KeyFlash(note_name, color, time.time(), lifetime))

    def _pending_notes(self):
        self._advance_pending_cursor()
        for i in range(self._next_pending_cursor, len(self.song.notes)):
            note = self.song.notes[i]
            if i in self.hit_notes or i in self.missed_notes:
                continue
            yield i, note

    def _advance_pending_cursor(self):
        while (
            self._next_pending_cursor < len(self.song.notes)
            and (
                self._next_pending_cursor in self.hit_notes
                or self._next_pending_cursor in self.missed_notes
            )
        ):
            self._next_pending_cursor += 1

    def _next_pending_time(self):
        next_time = None
        for _idx, note in self._pending_notes():
            if next_time is None or note.time < next_time:
                next_time = note.time
        return next_time

    def _pending_group_at(self, target_time: float):
        group = []
        for idx, note in self._pending_notes():
            if abs(note.time - target_time) <= self.chord_time_tolerance_beats:
                group.append((idx, note))
        return group

    def _nearest_pending_note(self, current_beat: float, note_name: Optional[str] = None):
        best_idx = None
        best_note = None
        best_dist = float("inf")
        for i, note in self._pending_notes():
            if note_name is not None and note.note != note_name:
                continue
            dist = abs(note.time - current_beat)
            if dist < best_dist:
                best_idx = i
                best_note = note
                best_dist = dist
        return best_idx, best_note, best_dist

    def _active_hold_notes(self, current_beat: float):
        result = []
        active_indices = set(self.active_holds.values())
        active_indices.update(
            idx
            for idx in self.hit_notes
            if idx < len(self.song.notes) and current_beat <= self.song.notes[idx].end_time + self.hold_release_grace_beats
        )
        for idx in active_indices:
            if idx >= len(self.song.notes):
                continue
            note = self.song.notes[idx]
            if current_beat <= note.end_time + self.hold_release_grace_beats:
                result.append((idx, note))
        result.sort(key=lambda item: item[1].time)
        return result

    def _upcoming_notes(self, current_beat: float, include_late_window: bool = True):
        self._advance_pending_cursor()
        late_window = self.song.seconds_to_beats(self.hit_window_ms / 1000.0) if include_late_window else 0.0
        start = max(self._next_pending_cursor, bisect_left(self.note_times, current_beat - late_window))
        stop = bisect_right(self.note_times, current_beat + self.song.seconds_to_beats(self._hint_seconds()) + 0.25)
        for i in range(start, min(stop, len(self.song.notes))):
            if i in self.hit_notes or i in self.missed_notes:
                continue
            yield i, self.song.notes[i]

    def _focus_note(self, current_beat: float):
        active = self._active_hold_notes(current_beat)
        if active:
            return active[0][0], active[0][1], True

        best_idx = None
        best_note = None
        best_beats_until = float("inf")
        for i, note in self._upcoming_notes(current_beat):
            beats_until = note.time - current_beat
            if beats_until < best_beats_until:
                best_idx = i
                best_note = note
                best_beats_until = beats_until
        return best_idx, best_note, False

    def due_autoplay_notes(self, current_beat: float, already_played: Set[int]):
        due = []
        due_until = current_beat + self.chord_time_tolerance_beats
        while self._autoplay_cursor < len(self.song.notes):
            idx = self._autoplay_cursor
            note = self.song.notes[idx]
            if note.time > due_until:
                break
            self._autoplay_cursor += 1
            if idx in already_played or idx in self.missed_notes:
                continue
            due.append((idx, note))
        return due

    def seek_autoplay(self, current_beat: float):
        cursor = bisect_left(self.note_times, current_beat)
        while cursor > 0 and self.song.notes[cursor - 1].end_time >= current_beat:
            cursor -= 1
        self._autoplay_cursor = cursor

    def mark_autoplay_hit(self, note_index: int):
        if note_index < 0 or note_index >= len(self.song.notes):
            return
        note = self.song.notes[note_index]
        if note_index not in self.hit_notes:
            self.hit_notes.add(note_index)
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
        self.active_holds[note.note] = note_index
        self.hold_start_times[note_index] = time.time()
        if self.wait_frozen:
            self._maybe_unfreeze_wait_group(note.time)
        self._flash_key(note.note, COLOR_HIT, 0.30)
        self._advance_pending_cursor()

    def _maybe_unfreeze_wait_group(self, target_time: float):
        for idx, _note in self._pending_group_at(target_time):
            if idx not in self.hit_notes:
                return
        self.wait_frozen = False
        if self.wait_freeze_time is not None:
            self.pause_offset += time.time() - self.wait_freeze_time
            self.wait_freeze_time = None
        

    def check_hit(self, note_name:str) -> bool:
        
        if self.start_time is None or self.finished:
            return False
        if self.is_counting_in():
            self._flash_key(note_name, (90, 160, 255), 0.25)
            self._set_feedback("Wait for the count-in", (90, 160, 255), 0.55)
            return False

        current_beat = self._current_beat()
        hit_window_beats = self.song.seconds_to_beats(self.hit_window_ms / 1000.0)

        if self.wait_frozen:
            group_time = self._next_pending_time()
            candidate = None
            if group_time is not None:
                for idx, note in self._pending_group_at(group_time):
                    if note.note == note_name:
                        candidate = (idx, note, abs(note.time - current_beat))
                        break
            best_idx, best_note, best_dist = candidate if candidate else (None, None, float("inf"))
        else:
            best_idx, best_note, best_dist = self._nearest_pending_note(current_beat, note_name)
        
        if best_idx is not None and best_dist <= hit_window_beats:
            self.hit_notes.add(best_idx)
            self._advance_pending_cursor()
            self.active_holds[note_name] = best_idx
            self.hold_start_times[best_idx] = time.time()
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            
            if self.wait_frozen:
                self._maybe_unfreeze_wait_group(best_note.time)
                    
            self._spawn_sparks(note_name)
            timing_ms = self.song.beats_to_seconds(current_beat - best_note.time) * 1000.0
            if abs(timing_ms) <= 80:
                timing_label = "Perfect"
            elif timing_ms < 0:
                timing_label = "Early"
            else:
                timing_label = "Late"
            self._flash_key(note_name, COLOR_HIT, 0.45)
            self._set_feedback(f"{timing_label}: {note_name}", COLOR_HIT, 0.75)
            return True

        self.wrong_hits += 1
        self.combo = 0
        self._flash_key(note_name, COLOR_WRONG)
        _target_idx, target_note, target_dist = self._nearest_pending_note(current_beat)
        if target_note is None:
            message = "No target note"
        else:
            target_offset_ms = self.song.beats_to_seconds(target_note.time - current_beat) * 1000.0
            if target_dist <= hit_window_beats * 1.75:
                message = f"Wrong key: play {target_note.note}"
            elif target_offset_ms > 0:
                message = f"Too early: next {target_note.note}"
            else:
                message = f"Timing missed: next {target_note.note}"
        roast = LEARNING_HUD_ROASTS[int(time.time() * 100) % len(LEARNING_HUD_ROASTS)]
        self._set_feedback(f"{message}  \u2014  {roast}", COLOR_WRONG, 1.2)
        return False
    
    def check_release(self, note_name: str):
        if note_name in self.active_holds:
            idx = self.active_holds.pop(note_name)
            self.hold_start_times.pop(idx, None)
            
    def _spawn_sparks(self, note_name: str, count: int = 12):
        key = self.key_map.get(note_name)
        if key is None:
            return
        cx = (key["x1"] + key["x2"]) // 2
        cy = self.piano_top_y
        now = time.time()
        for _ in range(count):
            angle = self._rng.uniform(0, 2 * np.pi)
            speed = self._rng.uniform(40, 160)
            self.sparks.append(
                Spark(
                    x=cx,
                    y=cy,
                    spawn_time=now,
                    vx=np.cos(angle) * speed,
                    vy=-abs(np.sin(angle) * speed) - 30,   # upward bias
                    
                    )
                )
            
    @property
    def total_notes(self):
        return len(self.song.notes)

    @property
    def accuracy(self):
        attempted = len(self.hit_notes) + len(self.missed_notes) + self.wrong_hits
        return (len(self.hit_notes) / attempted * 100) if attempted > 0 else 100.0
    
    def update_and_draw(self, frame, autoplay_active: bool = False):
        if self.start_time is None:
            return
        self._autoplay_was_active = autoplay_active  # used by _cheeky_line

        if autoplay_active and self.wait_frozen:
            self.wait_frozen = False
            self.wait_freeze_time = None

        current_beat = self._current_beat()
        look_ahead_beats = self.song.seconds_to_beats(self.look_ahead_seconds)
        fall_height = self.piano_top_y # pixel from top screen to piano
        pixels_per_beat = fall_height / look_ahead_beats if look_ahead_beats > 0 else 1
        self._draw_hit_zone(frame)
        # freeze if previous note has not been played
        if not autoplay_active and not self.wait_frozen and not self.finished:
            next_time = self._next_pending_time()
            if next_time is not None and next_time - current_beat <= 0:
                self.wait_frozen = True
                self.wait_freeze_time = time.time()
                current_beat = next_time
                # draws the falling notes
        old_stop = bisect_left(self.note_times, current_beat - 2.0)
        for i in range(self._next_pending_cursor, min(old_stop, len(self.song.notes))):
            if i in self.hit_notes or i in self.missed_notes:
                continue
            note = self.song.notes[i]
            if current_beat > note.end_time + 2.0:
                self.missed_notes.add(i)
                self.combo = 0
        self._advance_pending_cursor()

        draw_start = bisect_left(self.note_times, current_beat - 4.0)
        draw_stop = bisect_right(self.note_times, current_beat + look_ahead_beats + 1.0)
        for i in range(draw_start, min(draw_stop, len(self.song.notes))):
            note = self.song.notes[i]
            beats_until = note.time - current_beat
            key = self.key_map.get(note.note)
            if key is None:
                continue
                # skips notes too far ahead or far behind
            if beats_until > look_ahead_beats + 1:
                continue
            if beats_until < -note.duration - 2: # mark the missed notes
                if i not in self.hit_notes and i not in self.missed_notes:
                    self.missed_notes.add(i)
                    self.combo = 0
                continue
            
            note_bottom_y = int(self.piano_top_y - beats_until * pixels_per_beat)
            note_height = max(int(note.duration * pixels_per_beat), 10)
            note_top_y = note_bottom_y - note_height
            
            draw_top = max(0, note_top_y)
            draw_bottom = min(self.piano_top_y, note_bottom_y)
            if draw_bottom <= draw_top:
                continue
            
            if i in self.hit_notes:
                color = COLOR_HIT
            elif i in self.missed_notes:
                color = COLOR_MISSED
            elif key.get("is_black", False):
                color = COLOR_UPCOMING_BLACK
            else:
                color = COLOR_UPCOMING_WHITE
                
            x1 = key["x1"] + 2
            x2 = key["x2"] - 2
            blend_rect(frame, x1 - 4, draw_top, x2 + 4, draw_bottom, color, 0.22)
            cv2.rectangle(frame, (x1, draw_top), (x2, draw_bottom), color, -1)
            cv2.rectangle(frame, (x1, draw_top), (x2, draw_bottom), (255, 255, 255), 1)
            cv2.line(frame, (x1 + 1, draw_top + 2), (x2 - 1, draw_top + 2), (255, 255, 255), 1)
            
            block_visible_height = draw_bottom - draw_top
            
            if block_visible_height > 22:
                label = note.note
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
                tx = x1 + (x2 - x1 - text_size[0]) // 2
                ty = draw_top  + 16
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                

            if i in self.active_holds.values() and i in self.hold_start_times:
                hold_elapsed = time.time() - self.hold_start_times[i]
                hold_required = self.song.beats_to_seconds(note.duration)
                progress = min(hold_elapsed / hold_required, 1.0) if hold_required > 0 else 1.0
                bar_y = draw_bottom - 6
                bar_w = x2 - x1
                cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 5), COLOR_HOLD_BG, -1)
                cv2.rectangle(frame, (x1, bar_y), (x1 + int(bar_w * progress), bar_y + 5), COLOR_HOLD_FILL, -1)
        
        self._draw_key_hints(frame, current_beat)
        self._draw_key_flashes(frame)
        self._draw_sparks(frame)
        self._draw_hud(frame, current_beat)
        
        if current_beat > self.song.duration_beats + 2:
            self.finished = True
            self._draw_results(frame)

    def _cheeky_line(self):
        autoplay = getattr(self, '_autoplay_was_active', False)
        if autoplay:
            return AUTOPLAY_ROASTS[int(time.time() // 4) % len(AUTOPLAY_ROASTS)]
        if self.rush_e_mode:
            return LEARNING_HUD_ROASTS[int(time.time() // 3) % len(LEARNING_HUD_ROASTS)]
        return ""

    def _draw_hit_zone(self, frame):
        h, w = frame.shape[:2]
        zone_h = 7
        y1 = max(0, self.piano_top_y - zone_h)
        y2 = min(h, self.piano_top_y + zone_h)
        blend_rect(frame, 0, y1, w, y2, COLOR_HIT_ZONE, 0.16)
        cv2.line(frame, (0, self.piano_top_y), (w, self.piano_top_y), COLOR_HIT_ZONE, 2)

    def _draw_key_hints(self, frame, current_beat: float):
        hint_beats = self.song.seconds_to_beats(self._hint_seconds())

        # 1. Draw ALL active hold notes with hold markers
        active_holds = self._active_hold_notes(current_beat)
        held_indices = set()
        for idx, note in active_holds:
            held_indices.add(idx)
            key = self.key_map.get(note.note)
            if key is not None:
                blend_rect(frame, key["x1"], key["y1"], key["x2"], key["y2"], COLOR_ACTIVE_HOLD, 0.28)
                cv2.rectangle(frame, (key["x1"], key["y1"]), (key["x2"], key["y2"]), COLOR_ACTIVE_HOLD, 3)
                self._draw_next_note_marker(frame, note, color=COLOR_ACTIVE_HOLD, is_hold=True)

        # 2. Find the next upcoming note that is NOT currently being held
        next_focus_idx = None
        next_focus_note = None
        next_focus_beats = float("inf")
        for i, note in self._upcoming_notes(current_beat):
            if i in held_indices:
                continue
            beats_until = note.time - current_beat
            if beats_until < next_focus_beats:
                next_focus_idx = i
                next_focus_note = note
                next_focus_beats = beats_until

        # 3. Draw focus marker for next non-hold note (yellow arrow)
        if next_focus_note is not None:
            should_mark = next_focus_beats <= hint_beats or self.wait_frozen
            if should_mark:
                key = self.key_map.get(next_focus_note.note)
                if key is not None:
                    blend_rect(frame, key["x1"], key["y1"], key["x2"], key["y2"], COLOR_NEXT_ARROW, 0.18)
                    cv2.rectangle(frame, (key["x1"], key["y1"]), (key["x2"], key["y2"]), COLOR_NEXT_ARROW, 3)
                    self._draw_next_note_marker(frame, next_focus_note, color=COLOR_NEXT_ARROW, is_hold=False)

        # 4. Draw glow hints for other upcoming notes
        for i, note in self._upcoming_notes(current_beat, include_late_window=False):
            if i == next_focus_idx or i in held_indices:
                continue
            beats_until = note.time - current_beat
            if 0 <= beats_until <= hint_beats:
                key = self.key_map.get(note.note)
                if key is None:
                    continue
                intensity = max(0.0, 1.0 - beats_until / hint_beats) if hint_beats > 0 else 1.0
                pulse = 0.5 + 0.5 * np.sin(time.time() * 9.0)
                glow_alpha = 0.10 + (0.18 * intensity * pulse)
                blend_rect(frame, key["x1"], key["y1"], key["x2"], key["y2"], COLOR_HINT_GLOW, glow_alpha)
                cv2.rectangle(frame, (key["x1"], key["y1"]), (key["x2"], key["y2"]), COLOR_HINT_GLOW, 2)

    def _draw_next_note_marker(self, frame, note: SongNote, color=COLOR_NEXT_ARROW, is_hold: bool = False):
        key = self.key_map.get(note.note)
        if key is None:
            return
        cx = (key["x1"] + key["x2"]) // 2
        top_y = max(18, key["y1"] - 30)
        points = np.array(
            [
                [cx, key["y1"] - 4],
                [cx - 11, top_y],
                [cx + 11, top_y],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, points, color)
        ty = max(16, top_y - 5)
        label = note.note if not is_hold else f"{note.note} hold"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        tx = cx - text_size[0] // 2
        cv2.putText(frame, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_key_flashes(self, frame):
        now = time.time()
        alive = []
        for flash in self.key_flashes:
            age = now - flash.spawn_time
            if age >= flash.lifetime:
                continue
            key = self.key_map.get(flash.note)
            if key is None:
                continue
            alive.append(flash)
            strength = 1.0 - age / flash.lifetime
            blend_rect(frame, key["x1"], key["y1"], key["x2"], key["y2"], flash.color, 0.52 * strength)
            cv2.rectangle(frame, (key["x1"], key["y1"]), (key["x2"], key["y2"]), flash.color, 3)
        self.key_flashes = alive
            
    def _draw_sparks(self, frame):
        now = time.time()
        alive = []
        for spark in self.sparks:
            age = now - spark.spawn_time
            if age >= spark.lifetime:
                continue
            alive.append(spark)
            
            sx = int(spark.x + spark.vx * age)
            sy = int(spark.y + spark.vy * age)
            brightness = max(0.0, 1.0 - age / spark.lifetime)
            radius = max(1, int(4 * brightness))
            color = (
                int(255 * brightness),
                int(255 * brightness),
                int(200 * brightness),
            )
            cv2.circle(frame, (sx, sy), radius, color, -1)
            
        self.sparks = alive
            
    def _draw_hud(self, frame, current_beat):
        h, w = frame.shape[:2]
        cheeky_line = self._cheeky_line()
        panel_bottom = 72 if cheeky_line else 50
        blend_rect(frame, 24, 10, w - 24, panel_bottom, COLOR_PANEL, 0.78)
        cv2.rectangle(frame, (24, 10), (w - 24, panel_bottom), COLOR_PANEL_BORDER, 1)
        cv2.line(frame, (26, 11), (w - 26, 11), COLOR_NEXT_ARROW, 1)

        # Row 1: title (left) + score stats (right)
        score_text = f"{len(self.hit_notes)}/{self.total_notes}  Miss {len(self.missed_notes)}  Wrong {self.wrong_hits}  Combo {self.combo}  {self.accuracy:.0f}%"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        title_text = f"{self.song.title}  |  LEARNING"
        max_title_width = max(120, w - score_size[0] - 130)
        while len(title_text) > 16 and cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] > max_title_width:
            title_text = title_text[:-4] + "..."
        cv2.putText(frame, title_text, (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_SCORE_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, score_text, (w - score_size[0] - 40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_SCORE_TEXT, 1, cv2.LINE_AA)

        # Row 2: progress bar
        bar_y = 35
        bar_h = 5
        bar_margin = 40
        total_beats = max(self.song.duration_beats, 1)
        progress = min(max(current_beat / total_beats, 0), 1.0)
        cv2.rectangle(frame, (bar_margin, bar_y), (w - bar_margin, bar_y + bar_h), COLOR_PROGRESS_BG, -1)
        fill_w = int((w - 2 * bar_margin) * progress)
        cv2.rectangle(frame, (bar_margin, bar_y), (bar_margin + fill_w, bar_y + bar_h), COLOR_PROGRESS_FILL, -1)

        # Row 3: cheeky line — drawn *after* the bar so it's always on top and clearly below it
        if cheeky_line:
            cv2.putText(frame, cheeky_line, (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_NEXT_ARROW, 1, cv2.LINE_AA)

        if self.wait_frozen:
            cv2.putText(frame, "Play The Note!", (w // 2 - 90, panel_bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEXT_ARROW, 2, cv2.LINE_AA)

        self._draw_feedback(frame)
        self._draw_count_in(frame)
            
        if self.paused:
            cv2.putText(frame, "PAUSED", (w // 2-50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    def _draw_feedback(self, frame):
        if not self.feedback_text:
            return
        age = time.time() - self.feedback_time
        if age >= self.feedback_lifetime:
            return
        h, w = frame.shape[:2]
        fade = 1.0 - age / self.feedback_lifetime
        scale = 0.72
        thickness = 2
        size, baseline = cv2.getTextSize(self.feedback_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (w - size[0]) // 2
        y = min(h - 42, self.piano_top_y + 44)
        pad = 10
        cv2.rectangle(frame, (x - pad, y - size[1] - pad), (x + size[0] + pad, y + baseline + pad), COLOR_PANEL, -1)
        cv2.putText(frame, self.feedback_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, self.feedback_color, thickness, cv2.LINE_AA)

    def _draw_count_in(self, frame):
        remaining = self._count_in_remaining()
        if remaining <= 0:
            return

        beat_seconds = self.song.beats_to_seconds(1)
        total_seconds = self.song.beats_to_seconds(self.count_in_beats)
        if beat_seconds <= 0 or total_seconds <= 0:
            return

        h, w = frame.shape[:2]
        elapsed = max(0.0, total_seconds - remaining)
        current_step = int(elapsed / beat_seconds)
        display_number = max(1, self.count_in_beats - current_step)
        beat_phase = (elapsed % beat_seconds) / beat_seconds
        pulse = 1.0 - beat_phase

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

        cx = w // 2
        cy = max(130, self.piano_top_y // 2)
        label = "BRACE FOR E" if self.rush_e_mode else "GET READY"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0]
        cv2.putText(frame, label, (cx - label_size[0] // 2, cy - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.78, COLOR_SCORE_TEXT, 2, cv2.LINE_AA)

        number = str(display_number)
        scale = 2.6 + 0.35 * pulse
        thickness = 5
        number_size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        nx = cx - number_size[0] // 2
        ny = cy + number_size[1] // 2
        cv2.putText(frame, number, (nx + 3, ny + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, number, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_NEXT_ARROW, thickness, cv2.LINE_AA)

        dot_y = cy + 74
        spacing = 24
        start_x = cx - ((self.count_in_beats - 1) * spacing) // 2
        for i in range(self.count_in_beats):
            color = COLOR_NEXT_ARROW if i <= current_step else (85, 95, 105)
            radius = 6 if i == current_step else 4
            cv2.circle(frame, (start_x + i * spacing, dot_y), radius, color, -1)
            
    
    def _draw_results(self, frame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cx = w // 2
        cy = h // 2 - 40
        
        complete_label = "E SURVIVED!" if self.rush_e_mode else "SONG COMPLETE!"
        cv2.putText(frame, complete_label, (cx - 140, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(frame, self.song.title, (cx - 120, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Hits: {len(self.hit_notes)} / {self.total_notes}", (cx - 80, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SCORE_TEXT, 1)
        cv2.putText(frame, f"Missed: {len(self.missed_notes)}", (cx - 80, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_MISSED, 1)
        cv2.putText(frame, f"Wrong: {self.wrong_hits}", (cx - 80, cy + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WRONG, 1)
        cv2.putText(frame, f"Accuracy: {self.accuracy:.0f}%", (cx - 80, cy + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_HIT, 1)
        cv2.putText(frame, f"Max Combo: {self.max_combo}", (cx - 80, cy + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Press Q to exit", (cx - 70, cy + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    

