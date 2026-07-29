"""Unified CLI dispatcher for the SAE pipeline.

    python -m src.sae.steps <command> [args...]

Forwards the remaining args to the chosen step's own argparse. Examples:
    python -m src.sae.steps inventory --markdown docs/sae/EXP_INVENTORY.md
    python -m src.sae.steps extract --encoder HuBERT --dataset CREMA-D \
        --data-dir /scratch1/kelidari/ser_data/CREMA-D --layers 13
    python -m src.sae.steps step1 --encoder HuBERT --dataset CREMA-D --layer 13
    python -m src.sae.steps step4 --encoder HuBERT --dataset CREMA-D
    python -m src.sae.steps summary
"""
import sys
from importlib import import_module

from src.sae.steps import STEP_MODULES


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        print("commands: " + ", ".join(STEP_MODULES))
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    cmd = sys.argv[1]
    if cmd not in STEP_MODULES:
        print(f"unknown command '{cmd}'. commands: {', '.join(STEP_MODULES)}")
        sys.exit(1)
    # drop the command token so the step's own argparse sees a clean argv
    del sys.argv[1]
    sys.argv[0] = f"python -m src.sae.steps {cmd}"
    import_module(STEP_MODULES[cmd]).main()


if __name__ == "__main__":
    main()
