#!/usr/bin/env python3
"""Compatibility wrapper for the canonical network-kth bridge entrypoint."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "integrations"
    / "openbumblebee"
    / "transshield_network_kth_bridge"
    / "transshield_network_kth_bridge.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
