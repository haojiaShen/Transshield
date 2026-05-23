#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DOCX = REPO_ROOT / "docs" / "密捷竞赛作品报告.docx"
SHOWCASE_ROOT = REPO_ROOT / "showcase"
SHOWCASE_PUBLIC_ASSETS = SHOWCASE_ROOT / "public" / "report-assets"
SHOWCASE_GENERATED_JSON = SHOWCASE_ROOT / "src" / "generated" / "report_content.json"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W_NS, "a": A_NS}
FIGURE_ID_RE = re.compile(r"^图\s*([2-5]-[1-9])")

SECTION_HEADINGS = [
    "摘要",
    "第一章 作品概述",
    "第二章 系统设计",
    "第三章 系统实现",
    "第四章 测试方案与结果分析",
    "第五章 创新性与局限性",
    "第六章 复现与参赛声明",
    "参考文献",
    "附录A 关键代码实现",
]


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    chapter_route: str
    output_name: str


FIGURE_SPECS = [
    FigureSpec("图2-1", "design", "fig2-1-system-topology.png"),
    FigureSpec("图2-2", "design", "fig2-2-software-sequence.png"),
    FigureSpec("图2-3", "design", "fig2-3-pruning-rewrite.png"),
    FigureSpec("图3-1", "implementation", "fig3-1-project-architecture.png"),
    FigureSpec("图3-2", "implementation", "fig3-2-browser-worker-collaboration.png"),
    FigureSpec("图3-3", "implementation", "fig3-3-control-plane-gates.jpeg"),
    FigureSpec("图4-1", "results", "fig4-1-threshold-shift.png"),
    FigureSpec("图4-2", "results", "fig4-2-guard-matrix.png"),
    FigureSpec("图5-1", "innovation", "fig5-1-capability-matrix.jpeg"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paragraph_text(node: ET.Element) -> str:
    return "".join(text.text or "" for text in node.findall(".//w:t", NS)).strip()


def normalize_figure_token(text: str) -> str | None:
    value = text.strip()
    if not value.startswith("图"):
        return None
    match = FIGURE_ID_RE.match(value)
    if not match:
        return None
    candidate = f"图{match.group(1)}"
    return candidate if candidate in {item.figure_id for item in FIGURE_SPECS} else None


def extract_sections(paragraphs: Iterable[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    for text in paragraphs:
        if text in SECTION_HEADINGS:
            current_heading = text
            sections[current_heading] = []
            continue
        if current_heading is None:
            continue
        sections[current_heading].append(text)
    return sections


def first_nonempty(items: Iterable[str]) -> str | None:
    for item in items:
        value = item.strip()
        if value:
            return value
    return None


def collect_figure_records(docx_path: Path) -> list[dict]:
    with ZipFile(docx_path) as zip_file:
        rels_root = ET.fromstring(zip_file.read("word/_rels/document.xml.rels"))
        relationship_map = {
            rel.attrib.get("Id"): rel.attrib.get("Target")
            for rel in rels_root.findall(f"{{{P_REL_NS}}}Relationship")
        }
        document_root = ET.fromstring(zip_file.read("word/document.xml"))
        paragraphs = document_root.findall(".//w:body/w:p", NS)
        records = []
        for index, paragraph in enumerate(paragraphs):
            embeds = []
            for blip in paragraph.findall(".//a:blip", NS):
                rel_id = blip.attrib.get(f"{{{R_NS}}}embed")
                if rel_id:
                    embeds.append((rel_id, relationship_map.get(rel_id)))
            if not embeds:
                continue
            prev_text = paragraph_text(paragraphs[index - 1]) if index > 0 else ""
            next_text = paragraph_text(paragraphs[index + 1]) if index + 1 < len(paragraphs) else ""
            figure_id = first_nonempty(
                normalized
                for normalized in [normalize_figure_token(prev_text), normalize_figure_token(next_text)]
                if normalized
            )
            records.append(
                {
                    "figure_id": figure_id,
                    "prev_text": prev_text,
                    "next_text": next_text,
                    "target": embeds[0][1],
                }
            )
        return records


def copy_figure_assets(docx_path: Path, records: list[dict]) -> list[dict]:
    output = []
    figure_by_id = {item.figure_id: item for item in FIGURE_SPECS}
    SHOWCASE_PUBLIC_ASSETS.mkdir(parents=True, exist_ok=True)
    with ZipFile(docx_path) as zip_file:
        for record in records:
            figure_id = record["figure_id"]
            if figure_id not in figure_by_id:
                continue
            spec = figure_by_id[figure_id]
            target = record["target"]
            if not target:
                continue
            target_path = SHOWCASE_PUBLIC_ASSETS / spec.output_name
            target_path.write_bytes(zip_file.read(f"word/{target}"))
            output.append(
                {
                    "id": spec.figure_id,
                    "route": spec.chapter_route,
                    "src": f"/report-assets/{spec.output_name}",
                    "caption": record["next_text"] or record["prev_text"],
                    "context": record["prev_text"],
                }
            )
    return output


def build_generated_payload() -> dict:
    with ZipFile(REPORT_DOCX) as zip_file:
        root = ET.fromstring(zip_file.read("word/document.xml"))
        paragraphs = [paragraph_text(node) for node in root.findall(".//w:body/w:p", NS)]
        paragraphs = [item for item in paragraphs if item]

    sections = extract_sections(paragraphs)
    figures = copy_figure_assets(REPORT_DOCX, collect_figure_records(REPORT_DOCX))

    demo_summary = load_json(REPO_ROOT / "results" / "final" / "demo_content_summary_final.json")
    threshold_summary = load_json(REPO_ROOT / "results" / "final" / "medical_dynamic_threshold_calibration_final.json")
    auc_summary = load_json(REPO_ROOT / "results" / "final" / "medical_dynamic_auc_reference_final.json")
    communication_summary = load_json(REPO_ROOT / "results" / "communication" / "mainline_communication_profile_final.json")
    protocol_fuzz = load_json(REPO_ROOT / "results" / "fuzzing" / "protocol_fuzz_final.json")
    guard_stress = load_json(REPO_ROOT / "results" / "guard_stress" / "guard_stress_final.json")

    section_map = {
        "overview": "第一章 作品概述",
        "design": "第二章 系统设计",
        "implementation": "第三章 系统实现",
        "results": "第四章 测试方案与结果分析",
        "innovation": "第五章 创新性与局限性",
    }
    route_sections = {
        route: {
            "title": heading,
            "paragraphs": sections.get(heading, []),
        }
        for route, heading in section_map.items()
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_path": str(REPORT_DOCX.relative_to(REPO_ROOT)),
        "figures": figures,
        "abstract": sections.get("摘要", []),
        "sections": route_sections,
        "demo_summary": demo_summary,
        "formal_metrics": {
            "medical_threshold_accuracy": float(threshold_summary["best_threshold_accuracy"]),
            "medical_threshold": float(threshold_summary["best_threshold"]),
            "medical_auc": float(auc_summary["auc"]),
            "medical_sec_per_sample": float(communication_summary["medical"]["sec_per_sample"]),
            "medical_dual_total_gib": float(communication_summary["medical"]["dual_total_gib"]),
            "fuzz_passed": bool(protocol_fuzz["passed"]),
            "guard_passed": bool(guard_stress["passed"]),
        },
    }


def main():
    payload = build_generated_payload()
    SHOWCASE_GENERATED_JSON.parent.mkdir(parents=True, exist_ok=True)
    SHOWCASE_GENERATED_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generated_json": str(SHOWCASE_GENERATED_JSON.relative_to(REPO_ROOT)),
                "asset_dir": str(SHOWCASE_PUBLIC_ASSETS.relative_to(REPO_ROOT)),
                "figure_count": len(payload["figures"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
