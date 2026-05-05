import os
import sys
from pathlib import Path

# Ensure correct Python version
if sys.version_info.major != 3 or sys.version_info.minor != 11:
    print("ERROR: This project requires Python 3.11 (preferably 3.11.9).")
    print(f"Current version: {sys.version}")
    print("Please install Python 3.11 and create a new virtual environment.")
    sys.exit(1)

PIANO_DIR = Path(__file__).resolve().parent / "piano"

sys.path.insert(0, str(PIANO_DIR))
os.chdir(PIANO_DIR)

from launcher import PianoLauncher 

def main():
    app = PianoLauncher()
    app.mainloop()

if __name__ == "__main__":
    main()
