#!/usr/bin/env python3
"""Compatibility wrapper for the canonical OpenBumbleBee pipeline entrypoint."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.transshield_openbumblebee_pipeline import main


if __name__ == '__main__':
    main()
