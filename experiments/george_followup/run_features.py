"""Extract the features of every combination in combinations.py that does not have them yet.

    python run_features.py                  # everything that needs no download
    python run_features.py --all            # everything (after the downloads in combinations.NEEDS)

Runs exp3/features.py one combination at a time, training/validation cameras then test cameras.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import exp3_path  # noqa: F401
from combinations import COMBINATIONS, needs
from settings import feature_folder

FEATURES_SCRIPT = Path(__file__).resolve().parent.parent / "exp3" / "features.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also the combinations that need downloads")
    args = parser.parse_args()
    for c in COMBINATIONS:
        if needs(c) and not args.all:
            print(f"skip {c.name}: needs {', '.join(needs(c))}")
            continue
        for test in (False, True):
            if (feature_folder(c.backbone, c.view, test, c.size, c.detector) / "features.npy").exists():
                continue
            command = [sys.executable, str(FEATURES_SCRIPT), "--backbone", c.backbone, "--size", str(c.size),
                       "--view", c.view, "--detector", c.detector] + (["--test"] if test else [])
            print(f"== {c.name} {'test' if test else 'train/val'}", flush=True)
            subprocess.run(command, check=True, cwd=FEATURES_SCRIPT.parent)


if __name__ == "__main__":
    main()
