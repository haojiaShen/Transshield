#!/usr/bin/env python3
"""Compatibility wrapper for the canonical stagewise threshold report entrypoint."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_stagewise_threshold_report import main


if __name__ == "__main__":
    main()
