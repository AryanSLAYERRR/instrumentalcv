import argparse
import os
import sys
import urllib.request

import numpy as np
from scipy.signal import resample
import soundfile as sf

# Notes sampled in the Salamander set (every 3 semitones)
# Format: {note_name: filename_on_CDN}
SAMPLED_NOTES = ['A', 'C', 'D#', 'F#']

# All 12 chromatic notes
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

VELOCITY = 8

BASE_URL = "https://unpkg.com/@audio-samples/piano-velocity{vel}@1.0.3/audio"

INSTRUMENT_LABELS = {
    "grand": "Grand Piano",
    "bright": "Bright Piano",
    "electronic": "Electric Piano",
    "organ": "Organ",
    "reverb": "Reverb Piano",
}


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
def make_bright(audio, sr):
    from scipy.signal import butter, lfilter
    b, a = butter(2, 2000 / (sr / 2), btype='high')
    highs = lfilter(b, a, audio, axis=0)
    return np.clip(audio + highs * 1.5, -1, 1).astype(np.float32)

def make_electronic(audio, sr):
    from scipy.signal import butter, lfilter
    b, a = butter(4, 3000 / (sr / 2), btype='low')
    return lfilter(b, a, audio, axis=0).astype(np.float32)

def make_organ(audio, sr):
    """Organ: heavy reverb + low-pass for a smooth sustained pad sound."""
    from scipy.signal import butter, lfilter
    
    # Low-pass filter — remove the sharp piano "hammer" attack
    b, a = butter(3, 1500 / (sr / 2), btype='low')
    smooth = lfilter(b, a, audio, axis=0)
    
    # Layer delayed copies for sustained drone effect
    result = smooth.astype(np.float64).copy()
    delays = [(50, 0.5), (100, 0.35), (200, 0.25), (350, 0.15)]
    
    for delay_ms, gain in delays:
        n = int(delay_ms * sr / 1000)
        delayed = np.zeros_like(result)
        if len(result.shape) > 1:
            delayed[n:] = result[:-n] * gain
        else:
            delayed[n:] = result[:-n] * gain
        result += delayed
    
    # Normalize
    mx = np.max(np.abs(result))
    if mx > 0:
        result /= mx
    
    return (result * 0.8).astype(np.float32)

def make_reverb(audio, sr, amount=0.35):
    """Add room reverb to piano samples."""
    result = audio.astype(np.float64).copy()
    delays = [(23, 0.4), (47, 0.25), (71, 0.15)]
    for delay_ms, gain in delays:
        n = int(delay_ms * sr / 1000)
        delayed = np.zeros_like(result)
        if len(result.shape) > 1:
            delayed[n:] = result[:-n] * gain
        else:
            delayed[n:] = result[:-n] * gain
        result += delayed
    mx = np.max(np.abs(result))
    if mx > 0:
        result /= mx
    return ((1 - amount) * audio + amount * result).astype(np.float32)

VARIANT_OPTIONS = {
    "bright": make_bright,
    "electronic": make_electronic,
    "organ": make_organ,
    "reverb": lambda audio, sr: make_reverb(audio, sr),
}


def generate_variant(base_dir, variant_name, process_fn):
    src_dir = os.path.join(base_dir, "sounds")
    dst_dir = os.path.join(base_dir, "sounds_" + variant_name)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\nGenerating {variant_name} variant...")
    count = 0
    for filename in os.listdir(src_dir):
        if filename.endswith('.wav'):
            dst_path = os.path.join(dst_dir, filename)
            if os.path.exists(dst_path):
                count += 1
                continue
            src_path = os.path.join(src_dir, filename)
            audio, sr = sf.read(src_path)
            processed = process_fn(audio, sr)
            sf.write(dst_path, processed, sr)
            count += 1
    print(f"Generated {count} {variant_name} samples in: {dst_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and generate InstrumentalCV piano samples.")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["grand"],
        choices=["grand", "bright", "electronic", "organ", "reverb", "all"],
        help="Sample packs to prepare. Variants require the base Grand Piano samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = set(args.instruments)
    if "all" in selected:
        selected = set(INSTRUMENT_LABELS)

    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
    raw_dir = os.path.join(sounds_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Salamander Grand Piano V3 - Sample Downloader")
    print("  Source: darosh/samples-piano (MIT License)")
    print("  Selected: " + ", ".join(INSTRUMENT_LABELS[name] for name in sorted(selected)))
    print("=" * 60)
    
    # download the sampled notes (A, C, D#, F# for octaves 3 and 4)
    available_samples = []

    for octave in range(1, 8):  # download extra octaves for better pitch shifting
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
        return 1
    
    print(f"\nDownloaded {len(available_samples)} reference samples.")
    print("Generating all piano notes by pitch-shifting...\n")
    
    # generate all 24 notes (C3-B4) by pitch-shifting nearest sample
    generated = 0
    for octave in range (1, 8):
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
                fade_in_samples = int(0.005 * sr)  # 5ms
                if len(shifted.shape) > 1:
                    for ch in range(shifted.shape[1]):
                        shifted[:fade_in_samples, ch] *= np.linspace(0, 1, fade_in_samples)
                else:
                    shifted[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
                
                # fadeout (50ms) at end to prevent crackle on sample end
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

    if generated == 0:
        print("\nERROR: No playable WAV files were generated.")
        return 1

    base = os.path.dirname(os.path.abspath(__file__))
    for variant_name, process_fn in VARIANT_OPTIONS.items():
        if variant_name in selected:
            generate_variant(base, variant_name, process_fn)

    print(f"\nDone! {generated}/84 piano sounds ready in: {sounds_dir}")
    print("Selected instruments ready: " + ", ".join(INSTRUMENT_LABELS[name] for name in sorted(selected)))
    print("You can now run the piano! Press 'q' to quit.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
