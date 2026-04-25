import urllib.request
import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import soundfile as sf  

# Notes sampled in the Salamander set (every 3 semitones)
# Format: {note_name: filename_on_CDN}
SAMPLED_NOTES = ['A', 'C', 'D#', 'F#']

# All 12 chromatic notes
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

VELOCITY = 8

BASE_URL = "https://unpkg.com/@audio-samples/piano-velocity{vel}@1.0.3/audio"

def note_to_midi(note_name):
    if '#' in note_name:
        note = note_name[:-1]
        octave = int(note_name[-1])
    else:
        note = note_name[:-1]
        octave = int(note_name[-1])
    idx = CHROMATIC.index(note)
    return (octave + 1) * 12 + idx

def find_nearest_sample(target_note, octave, available_samples):
    target_midi = note_to_midi(f"{target_note}{octave}")
    best = None
    best_dist = 999
    for sample_note, sample_oct, filename in available_samples:
        sample_midi = note_to_midi(f"{sample_note}{sample_oct}")
        dist = abs(target_midi - sample_midi)
        if dist < best_dist:
            best_dist = dist
            best = (sample_note, sample_oct, filename, target_midi - sample_midi)
    return best

def pitch_shift(audio_data, sample_rate, semitones):
    if semitones == 0:
        return audio_data
    # Pitch ratio: each semitone = 2^(1/12) frequency ratio
    ratio = 2.0 ** (semitones / 12.0)
    # Resample: fewer samples = higher pitch, more samples = lower pitch
    new_length = int(len(audio_data) / ratio)
    if new_length == 0:
        return audio_data
    shifted = resample(audio_data, new_length)
    return shifted.astype(np.float32)

def main():
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
    raw_dir = os.path.join(sounds_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Salamander Grand Piano V3 - Sample Downloader")
    print("  Source: darosh/samples-piano (MIT License)")
    print("=" * 60)
    
    # download the sampled notes (A, C, D#, F# for octaves 3 and 4)
    available_samples = []
    
    for octave in [2, 3, 4, 5]:  # download extra octaves for better pitch shifting
        for note in SAMPLED_NOTES:
            note_name = f"{note}{octave}"
            # URL uses %23 for # in note names
            url_note = note_name.replace('#', '%23')
            filename = f"{note_name}v{VELOCITY}.ogg"
            filepath = os.path.join(raw_dir, filename)
            url = f"{BASE_URL.format(vel=VELOCITY)}/{url_note}v{VELOCITY}.ogg"
            
            if os.path.exists(filepath):
                print(f"  [cached] {filename}")
            else:
                try:
                    print(f"  Downloading {filename} ...", end=" ", flush=True)
                    urllib.request.urlretrieve(url, filepath)
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"OK ({size_kb:.0f} KB)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
            
            available_samples.append((note, octave, filepath))
    
    if not available_samples:
        print("\nERROR: No samples downloaded! Check your internet connection.")
        return
    
    print(f"\nDownloaded {len(available_samples)} reference samples.")
    print("Generating all piano notes by pitch-shifting...\n")
    
    # generate all 24 notes (C3-B4) by pitch-shifting nearest sample
    generated = 0
    for octave in [3, 4]:
        for note in CHROMATIC:
            note_name = f"{note}{octave}"
            out_path = os.path.join(sounds_dir, f"{note_name}.wav")
            
            if os.path.exists(out_path):
                print(f"  [cached] {note_name}.wav")
                generated += 1
                continue
            
            # Find nearest sample
            nearest = find_nearest_sample(note, octave, available_samples)
            if nearest is None:
                print(f"  SKIP {note_name} (no nearby sample)")
                continue
            
            src_note, src_oct, src_path, semitone_shift = nearest
            
            try:
                # Read the OGG sample
                audio, sr = sf.read(src_path)
                
                # if stereo convert to mono for pitch shifting, then back
                if len(audio.shape) > 1:
                    left = pitch_shift(audio[:, 0], sr, semitone_shift)
                    right = pitch_shift(audio[:, 1], sr, semitone_shift)
                    min_len = min(len(left), len(right))
                    shifted = np.column_stack((left[:min_len], right[:min_len]))
                else:
                    shifted = pitch_shift(audio, sr, semitone_shift)
                
                # save as WAV
                fade_in_samples = int(0.005 * sr)  # 5ms — very short, won't affect attack
                if len(shifted.shape) > 1:
                    for ch in range(shifted.shape[1]):
                        shifted[:fade_in_samples, ch] *= np.linspace(0, 1, fade_in_samples)
                else:
                    shifted[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
                
                # Fade-out (50ms) at end to prevent crackle on sample end
                fade_out_samples = int(0.05 * sr)
                if len(shifted.shape) > 1:
                    for ch in range(shifted.shape[1]):
                        shifted[-fade_out_samples:, ch] *= np.linspace(1, 0, fade_out_samples)
                else:
                    shifted[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
                
                sf.write(out_path, shifted, sr)
                
                shift_str = f"+{semitone_shift}" if semitone_shift >= 0 else str(semitone_shift)
                print(f"  Generated {note_name}.wav (from {src_note}{src_oct}, shift {shift_str} semitones)")
                generated += 1
                
            except Exception as e:
                print(f"  ERROR generating {note_name}: {e}")
    
    print(f"\nDone! {generated}/24 piano sounds ready in: {sounds_dir}")
    print("You can now run the piano! Press 'q' to quit.\n")

if __name__ == "__main__":
    main()
