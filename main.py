import os
import sys
from pathlib import Path

PIANO_DIR = Path(__file__).resolve().parent / "piano"

sys.path.insert(0, str(PIANO_DIR))
os.chdir(PIANO_DIR)

from launcher import PianoLauncher 

def main():
    app = PianoLauncher()
    app.mainloop()

if __name__ == "__main__":
    main()
