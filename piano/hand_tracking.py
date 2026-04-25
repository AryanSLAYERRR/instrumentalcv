import cv2
import mediapipe as mp
import time
import pygame
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

fingertip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}



COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

FINGER_COLORS = {
    "thumb": COLOR_YELLOW,
    "index": COLOR_GREEN,
    "middle": COLOR_BLUE,
    "ring": COLOR_ORANGE,
    "pinky": COLOR_MAGENTA
}

PIANO_HEIGHT_RATIO = 0.7
PIANO_BOTTOM_MARGIN = 0.4
PIANO_ALPHA = 0.6


HOVER_COOLDOWN_SECONDS = 0.15   # Minimum time between triggers on the SAME key
                                 # 0.15 = can play ~6 notes/sec max (plenty fast)
                                 # Lower = faster playing but more accidental retriggers
                                 # Higher = cleaner but feels sluggish


NOTE_TO_MIDI = {}

CHROMATIC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

for octave in range(0,9):
    for i,note in enumerate(CHROMATIC_NOTES):
        midi_num = (octave + 1) * 12 + i
        note_name = f"{note}{octave}"
        NOTE_TO_MIDI[note_name] = midi_num

WHITE_KEY_COUNT = 14    # 7 white keys per octave, 2 total octaves

WHITE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

BLACK_KEY_AFTER = {0, 1 , 3, 4, 5}

BLACK_NOTES = {0: "C#", 1: "D#", 3: "F#", 4: "G#", 5: "A#"}  # skipped 2 no e sharp

WHITE_KEY_HOVER_COLOR = (255,220,180) #light oranange
BLACK_KEY_HOVER_COLOR = (120,80,200) # purple 

class HoverTracker: # Tracks which key a finger is hovering over, frame by frame
    # Reports: note_on (entered new key), is_held (same key), note_off (left key)

    def __init__(self, finger_name):
        self.finger_name = finger_name
        self.prev_key_note = None
        self.current_key_note = None

        # Event flags (reset each frame)
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None

        self.last_trigger_time = 0.0
    
    def update(self, current_key, current_time):
        # Reset event flags from last frame
        self.note_on = False
        self.is_held = False
        self.note_off = False
        self.released_note = None

        self.current_key_note = current_key["note"] if current_key else None

        # Same key as last frame (or still no key)?
        if self.current_key_note == self.prev_key_note:
            if self.current_key_note is not None:
                self.is_held = True     # Finger still on same key = sustain
        else:
            # Key changed — check for note_off and note_on
            if self.prev_key_note is not None:
                self.note_off = True
                self.released_note = self.prev_key_note

            if self.current_key_note is not None:
                elapsed = current_time - self.last_trigger_time
                if elapsed >= HOVER_COOLDOWN_SECONDS:
                    self.note_on = True
                    self.last_trigger_time = current_time

        # Remember for next frame
        self.prev_key_note = self.current_key_note

def generate_piano_keys(frame_w, frame_h): # generates piano keys and returns there coordinates and corresponding notes
    white_keys = []
    black_keys = []
    
    white_key_w = frame_w / WHITE_KEY_COUNT

    piano_top = int(frame_h * (1 - PIANO_HEIGHT_RATIO))
    piano_bottom = int(frame_h * (1 - PIANO_BOTTOM_MARGIN))

    black_key_h = int((piano_bottom - piano_top) * 0.60)
    black_key_w = int(white_key_w * 0.6)

    for i in range(WHITE_KEY_COUNT):
        octave = 3 + (i // 7) # first 7 keys octave 4, next 7 octave 5
        note_idx = i % 7
        note_name = f"{WHITE_NOTES[note_idx]}{octave}"

        x1 = int(i * white_key_w) # left edge of white key
        x2 = int((i + 1 ) * white_key_w) # right edge of white key
        y1 = piano_top
        y2 = piano_bottom

        white_keys.append({"note": note_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "is_black": False, "color_default": COLOR_WHITE, "color_hover": WHITE_KEY_HOVER_COLOR, "color": COLOR_WHITE})

    for i in range(WHITE_KEY_COUNT):
        octave = 3 + (i // 7)
        note_idx = i % 7
        if note_idx in BLACK_KEY_AFTER:
            note_name = f"{BLACK_NOTES[note_idx]}{octave}"

            center_x = int((i + 1) * white_key_w)     # center between 2 keys

            x1 = center_x - (black_key_w // 2)
            x2 = center_x + (black_key_w // 2)
            y1 = piano_top
            y2 = piano_top + black_key_h

            black_keys.append({"note": note_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "is_black": True, "color_default": COLOR_BLACK, "color_hover": BLACK_KEY_HOVER_COLOR, "color": COLOR_BLACK})

    return white_keys, black_keys

def draw_piano(frame, white_keys, black_keys): # draws semi transparent piano keys, so hands are visible
    
    # Create a copy to draw solid keys on
    overlay = frame.copy()

    for key in white_keys:
        # Filled white rectangle
        cv2.rectangle(overlay, (key["x1"], key["y1"]), (key["x2"], key["y2"]),
                      key["color"], -1)
        # Border
        cv2.rectangle(overlay, (key["x1"], key["y1"]), (key["x2"], key["y2"]),
                      (180, 180, 180), 1)

    for key in black_keys:
        # Filled dark rectangle
        cv2.rectangle(overlay, (key["x1"], key["y1"]), (key["x2"], key["y2"]),
                      key["color"], -1)
        # Subtle highlight line at top for 3D effect
        cv2.line(overlay, (key["x1"] + 2, key["y1"] + 2),
                 (key["x2"] - 2, key["y1"] + 2), (80, 80, 80), 1)
        
    # Only blend the piano region (not the whole frame, saves resources)
    piano_top = white_keys[0]["y1"]
    blended_region = cv2.addWeighted(
        overlay[piano_top:, :],     # Piano area from overlay (has keys drawn)
        PIANO_ALPHA,                # 0.6 = 60% the piano
        frame[piano_top:, :],       
        1 - PIANO_ALPHA,            # 0.4 = 40% hands showing through
        0                           # No brightness offset
    )
    # Write blended result back into the frame
    frame[piano_top:, :] = blended_region
    # draws NOTE LABELS (after blending, so text stays fully opaque)
    for key in white_keys:
        note_text = key["note"]
        text_size = cv2.getTextSize(note_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = key["x1"] + (key["x2"] - key["x1"] - text_size[0]) // 2
        text_y = key["y2"] - 10
        cv2.putText(frame, note_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    for key in black_keys:
        note_text = key["note"]
        text_size = cv2.getTextSize(note_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        text_x = key["x1"] + (key["x2"] - key["x1"] - text_size[0]) // 2
        text_y = key["y2"] - 8
        cv2.putText(frame, note_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    if white_keys:
        cv2.line(frame, (0, piano_top), (frame.shape[1], piano_top), (0, 200, 255), 2) # top edge of piano with orange color

def detect_finger_on_key(finger_x, finger_y, all_keys): # checks if finger is inside a key, using the coordiantes of the finger, if they lie between the right and left edge of the key
    for key in all_keys:
        if key["x1"] <= finger_x <= key["x2"] and key["y1"] <= finger_y <= key["y2"]:
            return key
    return None

class SoundEngine:
    """
    Plays real Salamander Grand Piano WAV samples via pygame.mixer.
    No MIDI devices needed - just loads WAV files and plays them.
    Supports polyphony (multiple notes at once) via pygame channels.
    """
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(20)
        
        self.sounds = {}
        self.channels = {}
        
        sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
        
        if not os.path.exists(sounds_dir):
            print("ERROR: No sounds directory! Run 'python setup_sounds.py' first.")
            return
        
        print("Loading piano samples...")
        loaded = 0
        for filename in os.listdir(sounds_dir):
            if filename.endswith('.wav'):
                note_name = filename.replace('.wav', '')
                filepath = os.path.join(sounds_dir, filename)
                try:
                    self.sounds[note_name] = pygame.mixer.Sound(filepath)
                    loaded += 1
                except Exception as e:
                    print(f"  Failed to load {filename}: {e}")
        
        print(f"Loaded {loaded} piano samples (Salamander Grand Piano V3)")
    
    def note_on(self, note_name, velocity=100):
        if note_name not in self.sounds:
            return
        channel = pygame.mixer.find_channel()
        if channel:
            self.sounds[note_name].set_volume(velocity / 127.0)
            channel.play(self.sounds[note_name])
            self.channels[note_name] = channel
    
    def note_off(self, note_name, fadeout_ms=300):
        if note_name in self.channels:
            channel = self.channels[note_name]
            if channel and channel.get_busy():
                channel.fadeout(fadeout_ms)
            del self.channels[note_name]
    
    def cleanup(self):
        pygame.mixer.quit()

def reset_key_colors(white_keys, black_keys): # resets the keys color after the finger leaves the specified area
    for key in white_keys:
        key["color"] = key["color_default"]
    for key in black_keys:
        key["color"] = key["color_default"]

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error:Webcam not found/Is in use of other application")
    exit()


finger_trackers = {}

def get_tracker(hand_label, finger_name): # creates hover tracker for each finger of each hand
    key = f"{hand_label}_{finger_name}"
    if key not in finger_trackers:
        finger_trackers[key] = HoverTracker(finger_name)
    return finger_trackers[key]

# Generate keys
ret, temp_frame = cap.read()
if ret:
    temp_frame = cv2.flip(temp_frame, 1)
    h, w, _ = temp_frame.shape
    white_keys, black_keys = generate_piano_keys(w, h)
else: 
    print("Error: Unable to read from frame")
    exit()

all_keys = black_keys + white_keys # black keys are checked first in hover detection, so they take priority over white keys when hovering in the overlapping area

prev_time = 0

hovered_notes = []

sound_engine = SoundEngine()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam not found/Is in use of other application")
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    reset_key_colors(white_keys, black_keys)
    hovered_notes = []

    if result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), mp_draw.DrawingSpec(color=(150,150,150), thickness=1, circle_radius=1))
            hand_label = "Unknown"
            if result.multi_handedness:
                    hand_label = result.multi_handedness[hand_idx].classification[0].label

            for finger_name, tip_id in fingertip_ids.items():
                tip = hand_landmarks.landmark[tip_id]
                tip_x, tip_y = int(tip.x * frame_w), int(tip.y * frame_h)

                # Which key is this finger over?
                hovered_key = detect_finger_on_key(tip_x, tip_y, all_keys)

                # Update the hover tracker
                tracker = get_tracker(hand_label, finger_name)
                tracker.update(hovered_key, time.time())

                # Fingertip color: green when active, normal when idle
                if tracker.note_on or tracker.is_held:
                    dot_color = COLOR_GREEN
                else:
                    dot_color = FINGER_COLORS[finger_name]

                cv2.circle(frame, (tip_x, tip_y), 10, dot_color, -1)
                cv2.circle(frame, (tip_x, tip_y), 12, COLOR_WHITE, 1)

                # Key visual feedback
                if hovered_key is not None:
                    if tracker.is_held or tracker.note_on:
                        hovered_key["color"] = COLOR_GREEN              # Pressed/held
                    else:
                        hovered_key["color"] = hovered_key["color_hover"]

                    hovered_notes.append(hovered_key["note"])
                    key_center_x = (hovered_key["x1"] + hovered_key["x2"]) // 2
                    cv2.line(frame, (tip_x, tip_y), (key_center_x, hovered_key["y1"]),
                             dot_color, 1, cv2.LINE_AA)

                # NOTE ON finger just entered a new key
                if tracker.note_on:
                    note = hovered_key["note"]
                    print(f"🎹 NOTE ON:  {note} ({hand_label} {finger_name})")
                    sound_engine.note_on(note)
                    key_cx = (hovered_key["x1"] + hovered_key["x2"]) // 2
                    key_cy = (hovered_key["y1"] + hovered_key["y2"]) // 2
                    cv2.circle(frame, (key_cx, key_cy), 20, COLOR_GREEN, 2)
                    cv2.circle(frame, (key_cx, key_cy), 30, COLOR_GREEN, 1)

                # NOTE OFF finger just left a key
                if tracker.note_off:
                    print(f"   NOTE OFF: {tracker.released_note} ({hand_label} {finger_name})")
                    sound_engine.note_off(tracker.released_note)

            # Hand label near wrist
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * frame_w), int(wrist.y * frame_h)
            cv2.putText(frame, hand_label, (wrist_x - 30, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2)
                
                    

    draw_piano(frame, white_keys, black_keys)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN if fps > 20 else COLOR_RED, 2)   
    # displays which note is being hovered
    if hovered_notes:
        notes_text = "Hovering: " + ", ".join(hovered_notes)
        cv2.putText(frame, notes_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    else:
        cv2.putText(frame, "Move fingers over the keys!", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    #label
    cv2.putText(frame, "AIR PIANO MODE", (frame_w - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("InstrumentalCV Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
sound_engine.cleanup()
print("Piano closed. 🎹")


           