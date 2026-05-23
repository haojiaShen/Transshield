#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = REPO_ROOT / "results" / "guard_stress" / "guard_stress_final.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export preserved guard-stress evidence from the final TransShield repository."
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Ignored. The live web demo guard runner has been removed from the final repository.",
    )
    parser.add_argument(
        "--checks",
        default="",
        help="Ignored. Preserved final evidence is exported as-is.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULT),
        help="Where to write the preserved final guard-stress JSON.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    source = DEFAULT_RESULT.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing preserved guard-stress evidence: {source}")

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output != source:
        shutil.copyfile(source, output)

    print("live guard-stress execution has been removed from the final repository")
    print(f"exported preserved evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
