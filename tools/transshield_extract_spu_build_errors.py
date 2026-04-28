#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ERROR_LINE_RE = re.compile(r"^(ERROR:|FAILED:)")
DIAGNOSTIC_RE = re.compile(
    r"(?P<path>[^:\s][^:]*)"
    r":(?P<line>\d+)"
    r"(?::(?P<column>\d+))?"
    r": (?P<kind>fatal error|error|warning): (?P<message>.+)"
)


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def line_block(lines: list[str], center: int, before: int, after: int) -> dict:
    start = max(0, center - before)
    end = min(len(lines), center + after + 1)
    return {
        "start_line": start + 1,
        "end_line": end,
        "lines": [
            {
                "line_no": index + 1,
                "text": lines[index],
            }
            for index in range(start, end)
        ],
    }


def nearby_error_headers(lines: list[str], center: int, radius: int) -> list[dict]:
    headers: list[dict] = []
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    for index in range(start, end):
        text = lines[index]
        if ERROR_LINE_RE.search(text):
            headers.append({"line_no": index + 1, "text": text})
    return headers


def nearby_diagnostics(lines: list[str], center: int, radius: int) -> list[dict]:
    diagnostics: list[dict] = []
    seen: set[tuple[int, str]] = set()
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    for index in range(start, end):
        text = lines[index]
        match = DIAGNOSTIC_RE.search(text)
        if not match:
            continue
        key = (index, text)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            {
                "line_no": index + 1,
                "path": match.group("path"),
                "source_line": int(match.group("line")),
                "source_column": (
                    int(match.group("column"))
                    if match.group("column") is not None
                    else None
                ),
                "kind": match.group("kind"),
                "message": match.group("message"),
                "text": text,
            }
        )
    return diagnostics


def collect_global_error_headers(lines: list[str], limit: int) -> list[dict]:
    headers = [
        {"line_no": index + 1, "text": text}
        for index, text in enumerate(lines)
        if ERROR_LINE_RE.search(text)
    ]
    return headers[-limit:]


def build_phase_start(lines: list[str]) -> int:
    for index, text in enumerate(lines):
        if text.strip() == "[build] command_end":
            return index + 1
    return 0


def analyze_pattern(
    lines: list[str],
    pattern: str,
    block_before: int,
    block_after: int,
    search_radius: int,
) -> dict:
    start = build_phase_start(lines)
    match_indexes = [
        index for index, text in enumerate(lines[start:], start=start) if pattern in text
    ]
    if not match_indexes:
        return {
            "pattern": pattern,
            "matched": False,
        }

    first = match_indexes[0]
    last = match_indexes[-1]
    focus = last
    return {
        "pattern": pattern,
        "matched": True,
        "match_count": len(match_indexes),
        "first_match_line": first + 1,
        "last_match_line": last + 1,
        "focus_line": focus + 1,
        "error_headers": nearby_error_headers(lines, focus, search_radius),
        "diagnostics": nearby_diagnostics(lines, focus, search_radius),
        "excerpt": line_block(lines, focus, block_before, block_after),
    }


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"# SPU Build Error Extract — `{report['log_path']}`")
    lines.append("")
    lines.append(f"- total lines: `{report['total_lines']}`")
    lines.append(f"- matched patterns: `{report['matched_patterns']}` / `{report['requested_patterns']}`")
    lines.append("")

    if report.get("global_error_headers"):
        lines.append("## Global Error Headers")
        lines.append("")
        for header in report["global_error_headers"]:
            lines.append(f"- line `{header['line_no']}`: `{header['text']}`")
        lines.append("")

    if report.get("fallback_excerpt_head"):
        lines.append("## Fallback Head")
        lines.append("")
        lines.append("```text")
        for entry in report["fallback_excerpt_head"]:
            lines.append(f"{entry['line_no']:06d}: {entry['text']}")
        lines.append("```")
        lines.append("")

    if report.get("fallback_excerpt_tail"):
        lines.append("## Fallback Tail")
        lines.append("")
        lines.append("```text")
        for entry in report["fallback_excerpt_tail"]:
            lines.append(f"{entry['line_no']:06d}: {entry['text']}")
        lines.append("```")
        lines.append("")

    for item in report["patterns"]:
        lines.append(f"## Pattern: `{item['pattern']}`")
        lines.append("")
        if not item["matched"]:
            lines.append("- status: not found")
            lines.append("")
            continue

        lines.append(f"- match count: `{item['match_count']}`")
        lines.append(f"- first match line: `{item['first_match_line']}`")
        lines.append(f"- last match line: `{item['last_match_line']}`")
        lines.append(f"- focus line: `{item['focus_line']}`")
        lines.append("")

        if item["error_headers"]:
            lines.append("### Nearby error headers")
            lines.append("")
            for header in item["error_headers"]:
                lines.append(f"- line `{header['line_no']}`: `{header['text']}`")
            lines.append("")

        if item["diagnostics"]:
            lines.append("### Nearby compiler diagnostics")
            lines.append("")
            for diag in item["diagnostics"]:
                location = f"{diag['path']}:{diag['source_line']}"
                if diag["source_column"] is not None:
                    location += f":{diag['source_column']}"
                lines.append(f"- `{location}` `{diag['kind']}`: {diag['message']}")
            lines.append("")

        lines.append("### Excerpt")
        lines.append("")
        lines.append("```text")
        for entry in item["excerpt"]["lines"]:
            lines.append(f"{entry['line_no']:06d}: {entry['text']}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract actionable compiler-error excerpts for known SPU build blockers."
    )
    parser.add_argument("log_path", help="Path to Bazel build log")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Substring to locate in the log; may be passed multiple times",
    )
    parser.add_argument(
        "--block-before",
        type=int,
        default=12,
        help="How many lines to keep before the first match",
    )
    parser.add_argument(
        "--block-after",
        type=int,
        default=80,
        help="How many lines to keep after the first match",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=120,
        help="How many nearby lines to scan for ERROR headers and compiler diagnostics",
    )
    parser.add_argument(
        "--global-error-limit",
        type=int,
        default=80,
        help="How many global ERROR/FAILED lines to keep in the report",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    parser.add_argument("--output-md", default="", help="Optional Markdown output path")
    args = parser.parse_args()

    patterns = args.pattern or ["type_inference.cc", "beaver_ttp.cc"]
    log_path = Path(args.log_path).resolve()
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")

    lines = load_lines(log_path)
    items = [
        analyze_pattern(
            lines,
            pattern=pattern,
            block_before=args.block_before,
            block_after=args.block_after,
            search_radius=args.search_radius,
        )
        for pattern in patterns
    ]
    fallback_head = [
        {"line_no": index + 1, "text": lines[index]}
        for index in range(0, min(len(lines), 60))
    ]
    fallback_tail_start = max(0, len(lines) - 60)
    fallback_tail = [
        {"line_no": index + 1, "text": lines[index]}
        for index in range(fallback_tail_start, len(lines))
    ]
    report = {
        "log_path": str(log_path),
        "total_lines": len(lines),
        "requested_patterns": len(patterns),
        "matched_patterns": sum(int(item["matched"]) for item in items),
        "global_error_headers": collect_global_error_headers(
            lines, limit=args.global_error_limit
        ),
        "fallback_excerpt_head": fallback_head,
        "fallback_excerpt_tail": fallback_tail,
        "patterns": items,
    }

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_md(report), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
