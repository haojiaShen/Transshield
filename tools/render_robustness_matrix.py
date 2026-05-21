#!/usr/bin/env python3
"""Render a report-grade robustness validation matrix from real fuzz/guard JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.execve(
        sys.executable,
        [sys.executable, __file__, *sys.argv[1:]],
        {**os.environ, "PYTHONNOUSERSITE": "1"},
    )

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "report_evidence"
ASSET_DIR = REPO_ROOT / "docs" / "report_evidence" / "assets"
DEFAULT_PROTOCOL_JSON = RESULT_DIR / "protocol_fuzz_evidence.json"
DEFAULT_GUARD_JSON = RESULT_DIR / "control_plane_guard_evidence.json"
DEFAULT_OUTPUT = ASSET_DIR / "robustness_guard_matrix.png"

CASE_LABELS = {
    "transfer_encoding_chunked_blocked": "Chunked 编码",
    "oversized_content_length_best_effort_413": "超大 CL",
    "duplicate_field_blocked": "重复字段名",
    "boundary_fanout_blocked": "Boundary 扇出",
    "nested_multipart_blocked": "嵌套 multipart",
    "oversized_json_part_blocked": "超长 JSON",
    "unexpected_field_set_blocked": "字段集漂移",
    "boundary_param_whitespace_rejected": "Boundary 参数混淆",
    "malformed_part_header_blocked": "畸形 Part Header",
    "multipart_header_null_byte_blocked": "Header 空字节",
    "non_empty_epilogue_blocked": "非空 Epilogue",
    "utf16_json_charset_blocked": "异常 JSON 字符集",
    "truncated_body_blocked": "截断请求体",
    "duplicate_nonce_concurrent": "重复 Nonce 并发重放",
    "duplicate_payload_different_nonce": "同载荷换 Nonce",
    "per_ip_inflight_limit": "同 IP 并发占满",
    "ip_window_rate_limit": "短窗限频",
}

LAYER_LABELS = {
    "http_request_body_gate": "HTTP 体门",
    "content_length_header_gate": "CL 上限门",
    "raw_multipart_precheck": "raw multipart",
    "mime_tree_validation": "MIME 校验",
    "json_bytes_gate": "JSON 字节门",
    "strict_json_decoder": "严格 JSON 解码",
    "content_type_boundary_parser": "boundary 解析",
    "multipart_header_parser": "part header 解析",
    "strict_utf8_json_decoder": "strict UTF-8",
    "json_numeric_hooks": "数值钩子",
    "streaming_body_reader": "流式 body 读取器",
    "replay_guard_nonce_cache": "Nonce 重放门",
    "payload_fingerprint_guard": "payload 指纹门",
    "replay_guard_payload_cache": "载荷重放门",
    "ip_inflight_limit": "IP inflight 门",
    "per_ip_inflight_guard": "单 IP inflight",
    "global_inflight_guard": "全局 inflight",
    "ip_sliding_window_guard": "IP 短窗限频",
    "tcp_force_close_allowed": "TCP 兜底",
    "exact_field_set_gate": "字段集门",
    "not_applicable": "不适用",
}

ERROR_CODE_LABELS = {
    "transfer_encoding_not_supported": "拒绝 chunked",
    "payload_too_large": "超上限阻断",
    "invalid_multipart": "multipart 非法",
    "json_too_large": "JSON 超字节门",
    "duplicate_nonce": "重复 nonce",
    "duplicate_payload": "重复 payload",
    "busy_retry_later": "并发配额保护",
    "rate_limited": "滑动窗口限频",
}


def pick_font_family() -> str:
    candidate_files = [
        Path("/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidate_files:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            return font_manager.FontProperties(fname=str(font_path)).get_name()

    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def wrap(value: str, width: int) -> str:
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def short_case_name(name: str) -> str:
    return CASE_LABELS.get(name, name.replace("_", " "))


def short_layer(name: str) -> str:
    return LAYER_LABELS.get(name, name.replace("_", " "))


def resource_summary(system_state: dict) -> tuple[str, str]:
    delta = (system_state or {}).get("delta") or {}
    fd = int(delta.get("fd_count") or 0)
    socket_fd = int(delta.get("socket_fd_count") or 0)
    rss_kib = int(delta.get("rss_kib") or 0)
    if fd == 0 and socket_fd == 0 and rss_kib <= 0:
        return "稳态", "good"
    if fd == 0 and socket_fd == 0:
        return f"RSS+{rss_kib / 1024:.1f}", "warn"
    return f"FD异常/RSS+{rss_kib / 1024:.1f}", "bad"


def disposition_text(item: dict, category: str) -> str:
    if category == "协议层":
        status = item.get("status")
        code = ERROR_CODE_LABELS.get(item.get("error_code"), "")
        name = item.get("name")
        special = {
            "transfer_encoding_chunked_blocked": "HTTP 400",
            "oversized_content_length_best_effort_413": "HTTP 413",
            "truncated_body_blocked": "截断拒绝",
            "oversized_json_part_blocked": "JSON 门阻断",
        }
        if name in special:
            return special[name]
        if status and code:
            return f"HTTP {status}"
        if status:
            return f"HTTP {status}"
        return code or "按预期拒绝"

    details = item.get("details") or {}
    if item.get("name") == "duplicate_nonce_concurrent":
        success = details.get("success", 0)
        dup = details.get("duplicate_nonce", 0)
        return f"{success}过/{dup}拒"
    if item.get("name") == "duplicate_payload_different_nonce":
        results = details.get("results") or []
        blocked = sum(1 for x in results if (x.get("error_code") or "") == "duplicate_payload")
        return f"{blocked}次重复载荷拒绝"
    if item.get("name") == "per_ip_inflight_limit":
        results = details.get("results") or []
        busy = sum(1 for x in results if (x.get("error_code") or "") == "busy_retry_later")
        return f"{busy}次 busy_retry_later"
    if item.get("name") == "ip_window_rate_limit":
        results = details.get("results") or []
        limited = sum(1 for x in results if x.get("status") == 429)
        return f"{limited}次 429 限频"
    return "按预期守卫"


def load_rows(protocol_json: Path, guard_json: Path) -> list[dict]:
    rows: list[dict] = []

    if protocol_json.exists():
        payload = json.loads(protocol_json.read_text(encoding="utf-8"))
        for item in payload.get("results", []):
            summary, level = resource_summary(item.get("system_state") or {})
            rows.append(
                {
                    "category": "协议层",
                    "scenario": short_case_name(item.get("name", "")),
                    "intercept": short_layer(item.get("interception_layer", "")),
                    "fallback": short_layer(item.get("fallback_layer", "")),
                    "disposition": disposition_text(item, "协议层"),
                    "resource": summary,
                    "resource_level": level,
                    "passed": bool(item.get("passed")),
                    "delta": (item.get("system_state") or {}).get("delta") or {},
                }
            )

    if guard_json.exists():
        payload = json.loads(guard_json.read_text(encoding="utf-8"))
        for item in payload.get("checks", []):
            summary, level = resource_summary(item.get("system_state") or {})
            rows.append(
                {
                    "category": "控制面",
                    "scenario": short_case_name(item.get("name", "")),
                    "intercept": short_layer(item.get("interception_layer", "")),
                    "fallback": short_layer(item.get("fallback_layer", "")),
                    "disposition": disposition_text(item, "控制面"),
                    "resource": summary,
                    "resource_level": level,
                    "passed": bool(item.get("passed")),
                    "delta": (item.get("system_state") or {}).get("delta") or {},
                }
            )
    return rows


def compute_stats(rows: list[dict]) -> dict:
    protocol = sum(1 for row in rows if row["category"] == "协议层")
    guard = sum(1 for row in rows if row["category"] == "控制面")
    passed = sum(1 for row in rows if row["passed"])
    steady = sum(
        1
        for row in rows
        if (row["delta"].get("fd_count", 0) == 0 and row["delta"].get("socket_fd_count", 0) == 0)
    )
    rss_spike = sum(1 for row in rows if (row["delta"].get("rss_kib", 0) or 0) > 0)
    max_rss_mib = max(((row["delta"].get("rss_kib", 0) or 0) / 1024 for row in rows), default=0.0)
    return {
        "total": len(rows),
        "protocol": protocol,
        "guard": guard,
        "passed": passed,
        "steady": steady,
        "rss_spike": rss_spike,
        "max_rss_mib": max_rss_mib,
    }


def pill(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str, tc: str, fs: int = 11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=1.1,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")


def draw_protocol_table(
    ax,
    *,
    title: str,
    rows: list[dict],
    x0: float,
    y0: float,
    w: float,
    h: float,
    title_color: str,
):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            linewidth=1.15,
            edgecolor="#d6deea",
            facecolor="white",
        )
    )
    ax.text(x0 + 0.018, y0 + h - 0.035, title, ha="left", va="center", fontsize=14.8, fontweight="bold", color=title_color)

    good_fc, good_ec, good_tc = "#edf9f0", "#b7e3c3", "#15803d"
    warn_fc, warn_ec, warn_tc = "#fff7ea", "#ffd59b", "#b45309"
    bad_fc, bad_ec, bad_tc = "#fff0f0", "#f5b7b7", "#b42318"

    headers = ["异常场景", "首个拦截层", "结果", "资源状态"]
    fractions = [0.27, 0.24, 0.15, 0.34]
    col_x = [x0]
    for frac in fractions[:-1]:
        col_x.append(col_x[-1] + w * frac)
    col_edges = col_x + [x0 + w]

    header_top = y0 + h - 0.075
    header_h = 0.040
    ax.add_patch(plt.Rectangle((x0, header_top - header_h), w, header_h, facecolor="#f4f7fb", edgecolor="none"))
    for edge in col_edges[1:-1]:
        ax.plot([edge, edge], [y0 + 0.018, header_top], color="#e6ebf2", lw=1.0)
    for i, label in enumerate(headers):
        ax.text((col_edges[i] + col_edges[i + 1]) / 2, header_top - header_h / 2, label, ha="center", va="center", fontsize=11.0, fontweight="bold", color="#334155")

    body_top = header_top - header_h
    body_h = body_top - (y0 + 0.018)
    row_h = body_h / max(len(rows), 1)

    for idx, row in enumerate(rows):
        y_top = body_top - idx * row_h
        y_bottom = y_top - row_h
        if idx % 2 == 1:
            ax.add_patch(plt.Rectangle((x0, y_bottom), w, row_h, facecolor="#fafbfd", edgecolor="none"))
        ax.plot([x0, x0 + w], [y_bottom, y_bottom], color="#edf1f6", lw=0.9)

        mid_y = y_bottom + row_h / 2
        ax.text(col_edges[0] + 0.013, mid_y, row["scenario"], ha="left", va="center", fontsize=9.9, color="#1f2937")
        ax.text((col_edges[1] + col_edges[2]) / 2, mid_y, row["intercept"], ha="center", va="center", fontsize=9.4, color="#475569")
        ax.text((col_edges[2] + col_edges[3]) / 2, mid_y, row["disposition"], ha="center", va="center", fontsize=9.6, color="#334155", fontweight="bold")

        pill_x = col_edges[3] + 0.05
        pill_y = y_bottom + row_h * 0.24
        pill_w = 0.12
        pill_h = row_h * 0.50
        if row["resource_level"] == "good":
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], good_fc, good_ec, good_tc, fs=8.2)
        elif row["resource_level"] == "warn":
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], warn_fc, warn_ec, warn_tc, fs=8.2)
        else:
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], bad_fc, bad_ec, bad_tc, fs=7.8)


def draw_guard_cards(ax, *, rows: list[dict], x0: float, y0: float, w: float, h: float):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            linewidth=1.15,
            edgecolor="#d6deea",
            facecolor="white",
        )
    )
    ax.text(x0 + 0.018, y0 + h - 0.030, "控制面守卫验证（4 项）", ha="left", va="center", fontsize=14.0, fontweight="bold", color="#c66a00")

    headers = ["异常场景", "首个拦截层", "结果", "资源状态"]
    fractions = [0.29, 0.25, 0.18, 0.28]
    col_x = [x0]
    for frac in fractions[:-1]:
        col_x.append(col_x[-1] + w * frac)
    col_edges = col_x + [x0 + w]

    header_top = y0 + h - 0.045
    header_h = 0.030
    ax.add_patch(plt.Rectangle((x0, header_top - header_h), w, header_h, facecolor="#f4f7fb", edgecolor="none"))
    for edge in col_edges[1:-1]:
        ax.plot([edge, edge], [y0 + 0.014, header_top], color="#e6ebf2", lw=1.0)
    for i, label in enumerate(headers):
        ax.text((col_edges[i] + col_edges[i + 1]) / 2, header_top - header_h / 2, label, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#334155")

    good_fc, good_ec, good_tc = "#edf9f0", "#b7e3c3", "#15803d"
    warn_fc, warn_ec, warn_tc = "#fff7ea", "#ffd59b", "#b45309"
    bad_fc, bad_ec, bad_tc = "#fff0f0", "#f5b7b7", "#b42318"

    body_top = header_top - header_h
    body_h = body_top - (y0 + 0.014)
    row_h = body_h / max(len(rows), 1)
    for idx, row in enumerate(rows):
        y_top = body_top - idx * row_h
        y_bottom = y_top - row_h
        if idx % 2 == 1:
            ax.add_patch(plt.Rectangle((x0, y_bottom), w, row_h, facecolor="#fafbfd", edgecolor="none"))
        ax.plot([x0, x0 + w], [y_bottom, y_bottom], color="#edf1f6", lw=0.9)
        mid_y = y_bottom + row_h / 2
        ax.text(col_edges[0] + 0.012, mid_y, row["scenario"], ha="left", va="center", fontsize=9.4, color="#1f2937")
        ax.text((col_edges[1] + col_edges[2]) / 2, mid_y, row["intercept"], ha="center", va="center", fontsize=8.9, color="#475569")
        ax.text((col_edges[2] + col_edges[3]) / 2, mid_y, row["disposition"], ha="center", va="center", fontsize=8.9, color="#334155", fontweight="bold")
        pill_x = col_edges[3] + 0.04
        pill_y = y_bottom + row_h * 0.24
        pill_w = 0.12
        pill_h = row_h * 0.52
        if row["resource_level"] == "good":
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], good_fc, good_ec, good_tc, fs=7.8)
        elif row["resource_level"] == "warn":
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], warn_fc, warn_ec, warn_tc, fs=7.8)
        else:
            pill(ax, pill_x, pill_y, pill_w, pill_h, row["resource"], bad_fc, bad_ec, bad_tc, fs=7.4)


def render(protocol_json: Path, guard_json: Path, output: Path) -> None:
    font_family = pick_font_family()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = load_rows(protocol_json, guard_json)
    stats = compute_stats(rows)

    dpi = 220
    fig_w = 16
    fig_h = 14.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f8fc")
    ax.set_facecolor("#f6f8fc")

    # Title
    ax.text(0.03, 0.955, "协议层异常输入与控制面兜底验证矩阵", fontsize=24, fontweight="bold", color="#1f2937")
    ax.text(
        0.03,
        0.920,
        "数据直接来自协议 fuzz 与控制面守卫 JSON；图中同时呈现首个拦截层、兜底层以及资源回落状态。",
        fontsize=11.5,
        color="#5b6575",
    )

    # Summary cards
    card_y = 0.81
    card_h = 0.055
    card_w = 0.14
    card_gap = 0.016
    summary = [
        (f"总用例 {stats['total']}", "#eef4ff", "#c9dafc", "#2251cc"),
        (f"协议层 {stats['protocol']}", "#effbf4", "#b9e6c7", "#15803d"),
        (f"控制面 {stats['guard']}", "#fff6ea", "#ffd7a2", "#c66a00"),
        (f"按预期 {stats['passed']}/{stats['total']}", "#eefbf4", "#b9e6c7", "#15803d"),
        (f"FD/Socket 回基线 {stats['steady']}/{stats['total']}", "#eef4ff", "#c9dafc", "#2251cc"),
        (f"RSS 抬升 {stats['rss_spike']} 例，峰值 {stats['max_rss_mib']:.1f} MiB", "#fff8ed", "#ffd7a2", "#b45309"),
    ]
    x = 0.03
    widths = [0.11, 0.11, 0.11, 0.15, 0.19, 0.22]
    for (text, fc, ec, tc), w in zip(summary, widths):
        pill(ax, x, card_y, w, card_h, text, fc, ec, tc, fs=10.5)
        x += w + card_gap

    protocol_rows = [row for row in rows if row["category"] == "协议层"]
    guard_rows = [row for row in rows if row["category"] == "控制面"]

    draw_protocol_table(
        ax,
        title="协议层异常输入验证（13 项）",
        rows=protocol_rows,
        x0=0.03,
        y0=0.28,
        w=0.94,
        h=0.50,
        title_color="#2251cc",
    )
    draw_guard_cards(ax, rows=guard_rows, x0=0.03, y0=0.04, w=0.94, h=0.20)

    # Footer note
    ax.text(
        0.03,
        0.008,
        "说明：绿色为稳态回落；橙色为 RSS 瞬时抬升；该图只保留可视化摘要，详细兜底链以正文表格为准。",
        fontsize=9.8,
        color="#5b6575",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render robustness matrix from report JSON.")
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL_JSON)
    parser.add_argument("--guard-json", type=Path, default=DEFAULT_GUARD_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.protocol_json, args.guard_json, args.output)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
