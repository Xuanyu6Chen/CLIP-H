# Makes the code folders of this module importable: put this folder on sys.path and `import code_paths` before any other import.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Pipeline order. Every .py file in these folders is imported by its file name.
CODE_FOLDERS = [
    "common",
    "generate",
    "prediction",
]

for folder in reversed(CODE_FOLDERS):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
