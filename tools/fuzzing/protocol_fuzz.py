#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = REPO_ROOT / "results" / "fuzzing" / "protocol_fuzz_final.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export preserved protocol fuzz evidence from the final TransShield repository."
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Ignored. The live web demo fuzz runner has been removed from the final repository.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULT),
        help="Where to write the preserved final protocol fuzz JSON.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    source = DEFAULT_RESULT.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing preserved protocol fuzz evidence: {source}")

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output != source:
        shutil.copyfile(source, output)

    print("live protocol fuzz execution has been removed from the final repository")
    print(f"exported preserved evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
