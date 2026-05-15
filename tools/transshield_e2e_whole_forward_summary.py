import argparse
import json
import re
from pathlib import Path


LINK_RE = re.compile(
    r"(?P<label>Link details|ColocatedIo sync link details): total send bytes "
    r"(?P<send_bytes>\d+), recv bytes (?P<recv_bytes>\d+), "
    r"send actions (?P<send_actions>\d+), recv actions (?P<recv_actions>\d+)"
)
CHEETAH_TRAFFIC_RE = re.compile(
    r"\[(?P<op>cheetah_[^\]]+)\].*?Recv (?P<recv_value>[0-9.]+) (?P<recv_unit>[KMGT]?iB), "
    r"Response (?P<response_value>[0-9.]+) (?P<response_unit>[KMGT]?iB)"
)


UNIT_BYTES = {
    "iB": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def human_bytes(value):
    if value is None:
        return None
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def parse_size(value, unit):
    return int(round(float(value) * UNIT_BYTES[unit]))


def parse_node_log(path: Path):
    link_matches = []
    nonzero_link_matches = []
    cheetah_events = []
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        link_match = LINK_RE.search(line)
        if link_match:
            item = {
                "label": link_match.group("label"),
                "send_bytes": int(link_match.group("send_bytes")),
                "recv_bytes": int(link_match.group("recv_bytes")),
                "send_actions": int(link_match.group("send_actions")),
                "recv_actions": int(link_match.group("recv_actions")),
            }
            item["total_bytes"] = item["send_bytes"] + item["recv_bytes"]
            link_matches.append(item)
            if item["total_bytes"] or item["send_actions"] or item["recv_actions"]:
                nonzero_link_matches.append(item)
        traffic_match = CHEETAH_TRAFFIC_RE.search(line)
        if traffic_match:
            recv_bytes = parse_size(traffic_match.group("recv_value"), traffic_match.group("recv_unit"))
            response_bytes = parse_size(traffic_match.group("response_value"), traffic_match.group("response_unit"))
            cheetah_events.append(
                {
                    "op": traffic_match.group("op"),
                    "recv_bytes": recv_bytes,
                    "response_bytes": response_bytes,
                    "total_bytes": recv_bytes + response_bytes,
                    "line": line.strip(),
                }
            )

    cheetah_recv = sum(item["recv_bytes"] for item in cheetah_events)
    cheetah_response = sum(item["response_bytes"] for item in cheetah_events)
    return {
        "log_path": str(path),
        "latest_link_details": link_matches[-1] if link_matches else None,
        "latest_nonzero_link_details": nonzero_link_matches[-1] if nonzero_link_matches else None,
        "link_detail_count": len(link_matches),
        "nonzero_link_detail_count": len(nonzero_link_matches),
        "cheetah_event_count": len(cheetah_events),
        "cheetah_recv_bytes_sum": cheetah_recv,
        "cheetah_response_bytes_sum": cheetah_response,
        "cheetah_total_bytes_sum": cheetah_recv + cheetah_response,
        "cheetah_top_events": sorted(cheetah_events, key=lambda item: item["total_bytes"], reverse=True)[:12],
    }


def summarize_logs(log_dir: Path):
    node_logs = []
    if log_dir.exists():
        for log_path in sorted(log_dir.glob("node_*.log")):
            parsed = parse_node_log(log_path)
            if parsed is not None:
                node_logs.append(parsed)

    latest_nonzero = [
        item["latest_nonzero_link_details"]
        for item in node_logs
        if item.get("latest_nonzero_link_details") is not None
    ]
    link_sum_total = sum(item["total_bytes"] for item in latest_nonzero) if latest_nonzero else None
    link_max_total = max((item["total_bytes"] for item in latest_nonzero), default=None)
    cheetah_sum_total = sum(item["cheetah_total_bytes_sum"] for item in node_logs)
    cheetah_max_total = max((item["cheetah_total_bytes_sum"] for item in node_logs), default=None)
    return {
        "spu_log_dir": str(log_dir),
        "status": "available" if node_logs else "missing",
        "node_logs": node_logs,
        "link_details": {
            "status": "available" if latest_nonzero else "missing",
            "latest_nonzero_per_node": latest_nonzero,
            "sum_total_bytes": link_sum_total,
            "sum_total_human": human_bytes(link_sum_total),
            "max_node_total_bytes": link_max_total,
            "max_node_total_human": human_bytes(link_max_total),
        },
        "cheetah_traffic_lines": {
            "status": "available" if cheetah_sum_total else "missing",
            "sum_total_bytes": cheetah_sum_total if cheetah_sum_total else None,
            "sum_total_human": human_bytes(cheetah_sum_total) if cheetah_sum_total else None,
            "max_node_total_bytes": cheetah_max_total,
            "max_node_total_human": human_bytes(cheetah_max_total),
            "note": (
                "Parsed from Cheetah op diagnostic lines. This is useful when LinkDetails are missing, "
                "but it is an operator-level diagnostic counter, not a packet capture."
            ),
        },
    }


def pick(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def compute_accuracy(predictions, targets):
    if predictions is None or targets is None:
        return None
    count = min(len(predictions), len(targets))
    if count <= 0:
        return None
    correct = sum(int(predictions[index]) == int(targets[index]) for index in range(count))
    return float(correct * 100.0 / count)


def build_target_summary(share_public_manifest):
    if not isinstance(share_public_manifest, dict):
        return None
    targets = share_public_manifest.get("targets")
    if targets is None:
        return {
            "targets_included": bool(share_public_manifest.get("targets_included", False)),
            "targets": None,
        }
    return {
        "targets_included": True,
        "targets": targets,
    }


def build_summary(candidate, log_summary, share_public_manifest=None):
    preview = candidate.get("prediction_preview") or {}
    spu = candidate.get("spu") or {}
    sample_count = candidate.get("sample_count")
    elapsed = candidate.get("elapsed_sec")
    target_summary = build_target_summary(share_public_manifest)
    targets = None if target_summary is None else target_summary.get("targets")
    argmax_predictions = preview.get("argmax_predictions")
    threshold_predictions = preview.get("threshold_predictions")
    return {
        "manifest_type": "transshield_e2e_whole_forward_summary_v0",
        "runtime": candidate.get("runtime"),
        "backend": candidate.get("backend"),
        "sample_count": sample_count,
        "elapsed_sec": elapsed,
        "sec_per_sample": (float(elapsed) / int(sample_count)) if elapsed is not None and sample_count else None,
        "finite_logits": candidate.get("finite_logits"),
        "spu_params_mode": candidate.get("spu_params_mode", spu.get("spu_params_mode")),
        "spu_layer_norm_policy": spu.get("spu_layer_norm_policy"),
        "spu_attention_policy": pick(spu, "static_forward_metadata", "attention_policy"),
        "spu_activation_kind": pick(spu, "static_forward_metadata", "activation_kind"),
        "spu_static_depth": pick(spu, "static_forward_metadata", "depth"),
        "privacy_fields": {
            "input_mode": candidate.get("input_mode", spu.get("input_mode")),
            "host_plaintext_pixel_values_materialized": candidate.get(
                "host_plaintext_pixel_values_materialized",
                spu.get("host_plaintext_pixel_values_materialized"),
            ),
            "host_private_share_tensors_loaded": candidate.get(
                "host_private_share_tensors_loaded",
                spu.get("host_private_share_tensors_loaded"),
            ),
            "private_input_paths_redacted": candidate.get(
                "private_input_paths_redacted",
                spu.get("private_input_paths_redacted"),
            ),
            "reveal_policy": candidate.get("reveal_policy", spu.get("reveal_policy")),
        },
        "prediction_preview": {
            "logits": preview.get("logits"),
            "probabilities": preview.get("probabilities"),
            "argmax_predictions": argmax_predictions,
            "threshold_predictions": threshold_predictions,
        },
        "target_metrics": {
            "targets_available": targets is not None,
            "targets": targets,
            "argmax_accuracy": compute_accuracy(argmax_predictions, targets),
            "threshold_accuracy": compute_accuracy(threshold_predictions, targets),
        },
        "communication_from_spu_logs": log_summary,
        "scope_note": (
            "This summarizes the current whole-forward SPU candidate JSON plus current logs/spu_nodes. "
            "For local colocated tests, communication byte counters approximate protocol traffic shape; "
            "two-machine wall time will also depend on bandwidth, RTT, CPU load, and runtime scheduling."
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize E2E whole-forward SPU candidate JSON and node-log communication.")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--share-public-json", default="")
    parser.add_argument("--spu-log-dir", default="logs/spu_nodes")
    parser.add_argument("--output-json", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    candidate = load_json(Path(args.candidate_json).expanduser().resolve())
    share_public_manifest = (
        load_json(Path(args.share_public_json).expanduser().resolve())
        if args.share_public_json
        else None
    )
    log_summary = summarize_logs(Path(args.spu_log_dir).expanduser().resolve())
    summary = build_summary(candidate, log_summary, share_public_manifest)
    write_json(Path(args.output_json).expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
