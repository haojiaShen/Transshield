#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_list_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--list must use label=path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty list label")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"image list does not exist: {path}")
    return label, path


def load_images(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_counts(images: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for image in images:
        parts = Path(image).parts
        label = None
        for index, part in enumerate(parts[:-1]):
            if part == "val" and parts[index + 1] in {"0", "1"}:
                label = parts[index + 1]
                break
        if label is None:
            label = "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def overlap_rows(datasets: List[Dict]) -> List[Dict]:
    rows = []
    for i, left in enumerate(datasets):
        left_set = set(left["images"])
        for right in datasets[i + 1 :]:
            right_set = set(right["images"])
            overlap = sorted(left_set & right_set)
            rows.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "left_count": len(left_set),
                    "right_count": len(right_set),
                    "overlap_count": len(overlap),
                    "overlap_ratio_left": len(overlap) / len(left_set) if left_set else None,
                    "overlap_ratio_right": len(overlap) / len(right_set) if right_set else None,
                    "overlap_preview": overlap[:8],
                }
            )
    return rows


def write_markdown(path: Path, report: Dict) -> None:
    lines = [
        "# E2E Image List Overlap Report",
        "",
        f"- label: `{report['label']}`",
        "",
        "| list | samples | class 0 | class 1 | unknown |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in report["datasets"]:
        counts = dataset["class_counts"]
        lines.append(
            "| {label} | {count} | {c0} | {c1} | {unknown} |".format(
                label=dataset["label"],
                count=dataset["sample_count"],
                c0=counts.get("0", 0),
                c1=counts.get("1", 0),
                unknown=counts.get("unknown", 0),
            )
        )
    lines.extend(["", "## Pairwise Overlap", "", "| left | right | overlap | left ratio | right ratio |"])
    lines.append("|---|---|---:|---:|---:|")
    for row in report["pairwise_overlap"]:
        lines.append(
            "| {left} | {right} | {overlap} | {left_ratio:.6g} | {right_ratio:.6g} |".format(
                left=row["left"],
                right=row["right"],
                overlap=row["overlap_count"],
                left_ratio=row["overlap_ratio_left"] or 0.0,
                right_ratio=row["overlap_ratio_right"] or 0.0,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize E2E image-list class balance and pairwise overlap.")
    parser.add_argument("--list", action="append", type=parse_list_arg, required=True)
    parser.add_argument("--label", default="e2e_image_list_overlap")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    datasets = []
    for label, path in args.list:
        images = load_images(path)
        datasets.append(
            {
                "label": label,
                "path": str(path),
                "sample_count": len(images),
                "class_counts": class_counts(images),
                "images": images,
            }
        )
    report = {
        "manifest_type": "transshield_e2e_image_list_overlap_report_v0",
        "label": args.label,
        "datasets": datasets,
        "pairwise_overlap": overlap_rows(datasets),
    }
    write_json(Path(args.output_json).expanduser().resolve(), report)
    if args.output_md:
        write_markdown(Path(args.output_md).expanduser().resolve(), report)
    print(json.dumps({"label": args.label, "pairwise_overlap": report["pairwise_overlap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
