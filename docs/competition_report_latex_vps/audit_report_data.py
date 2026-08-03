#!/usr/bin/env python3
"""Cross-check report numbers against repository evidence and generated inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import fitz


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL_PDF = HERE / "output/pdf/final_report.pdf"

VPS_DATA = HERE / "vps_report_data.json"
RAW_MEDICAL = REPO / (
    "artifacts/vps_release_backup_20260803/results/vps_report_tests/"
    "report_sumdiff_full_20260802_v1/medical32_spu_latest_summary.json"
)
RAW_FINANCE = REPO / (
    "artifacts/vps_release_backup_20260803/results/vps_report_tests/"
    "report_sumdiff_full_20260802_v1/finance8_spu_latest_summary.json"
)
DEMO = REPO / "results/final/demo_content_summary_final.json"
CALIBRATION = REPO / "results/final/medical_dynamic_threshold_calibration_final.json"
AUC_REFERENCE = REPO / "results/final/medical_dynamic_auc_reference_final.json"
FUZZ = REPO / "results/fuzzing/protocol_fuzz_final.json"
GUARD = REPO / "results/guard_stress/guard_stress_final.json"
TRAINING_ARGS = REPO / "artifacts/frozen_bundle_medical_dynamic_mainline/args_snapshot.json"


class Audit:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.notes: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed.append(name)
        else:
            self.failed.append(f"{name}: {detail}" if detail else name)

    def equal(self, name: str, actual: Any, expected: Any, *, tol: float = 1e-9) -> None:
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            ok = math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)
        else:
            ok = actual == expected
        self.check(name, ok, f"actual={actual!r}, expected={expected!r}")

    def note(self, text: str) -> None:
        self.notes.append(text)

    def finish(self) -> None:
        print(f"PASS {len(self.passed)}")
        for item in self.passed:
            print(f"  [PASS] {item}")
        if self.notes:
            print(f"NOTE {len(self.notes)}")
            for item in self.notes:
                print(f"  [NOTE] {item}")
        if self.failed:
            print(f"FAIL {len(self.failed)}")
            for item in self.failed:
                print(f"  [FAIL] {item}")
            raise SystemExit(1)
        print("REPORT DATA AUDIT PASSED")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy(values: list[float], labels: list[int], threshold: float) -> float:
    return sum((value >= threshold) == bool(label) for value, label in zip(values, labels)) / len(values)


def binary_auc(values: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC with average ranks for ties."""

    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def find_profile(demo: dict[str, Any], profile_id: str) -> dict[str, Any]:
    for profile in demo["standardized_secure_benchmark"]["profiles"]:
        if profile["profile_id"] == profile_id:
            return profile
    raise KeyError(profile_id)


def audit_vps_evidence(audit: Audit, vps: dict[str, Any]) -> None:
    audit.equal("VPS evidence schema", vps["schema"], "transshield_report_vps_evidence_excerpt_v1")
    environment = vps["environment"]
    expected_environment = {
        "platform": "Alibaba Cloud KVM",
        "cpu": "Intel Xeon Platinum",
        "vcpu": 16,
        "memory_gib": 61,
        "swap_gib": 0,
        "os": "Ubuntu 24.04.4 LTS",
        "kernel": "Linux 6.8.0-136-generic",
        "python": "3.9.25",
        "spu": "0.9.3b0",
        "jax": "0.4.30",
        "numpy": "1.26.4",
        "pytorch": "1.13.1+cpu",
        "gpu": None,
    }
    for key, value in expected_environment.items():
        audit.equal(f"VPS environment {key}", environment[key], value)

    for name, raw_path in (("medical", RAW_MEDICAL), ("finance", RAW_FINANCE)):
        excerpt = vps[name]
        audit.equal(f"{name} evidence source path", excerpt["source_path"], str(raw_path.relative_to(REPO)))
        audit.check(f"{name} evidence SHA-256 format", bool(re.fullmatch(r"[0-9a-f]{64}", excerpt["source_sha256"])))
        if raw_path.is_file():
            audit.equal(f"{name} raw source SHA-256", sha256_file(raw_path), excerpt["source_sha256"])
            raw = read_json(raw_path)
            for key in (
                "sample_count",
                "elapsed_sec",
                "sec_per_sample",
                "threshold",
                "threshold_accuracy",
                "auc",
                "finite_logits",
                "finite_probabilities",
            ):
                audit.equal(f"{name} excerpt/raw {key}", excerpt[key], raw[key])
            for key in ("tx_delta_bytes", "rx_delta_bytes", "total_bytes", "total_gib", "per_sample_gib"):
                audit.equal(f"{name} excerpt/raw network {key}", excerpt["network"][key], raw["network"][key])
            static = raw["runtime_metadata"]["spu"]["static_forward_metadata"]
            runtime = excerpt["runtime"]
            audit.equal(f"{name} excerpt/raw base_rate", runtime["base_rate"], static["base_rate"])
            audit.equal(f"{name} excerpt/raw token ratios", runtime["token_ratio"], static["token_ratio"])
            audit.equal(f"{name} excerpt/raw token counts", runtime["token_keep_counts"], static["token_keep_counts"])
            audit.equal(f"{name} excerpt/raw depth", runtime["effective_depth"], static["depth"])
            audit.equal(
                f"{name} excerpt/raw SPU batch size",
                runtime["spu_batch_size"],
                raw["runtime_metadata"]["spu"]["spu_batch_size"],
            )
            if name == "medical":
                audit.equal(
                    f"{name} excerpt/raw first sample",
                    excerpt["first_sample"],
                    {key: raw["per_sample"][0][key] for key in excerpt["first_sample"]},
                )
                audit.equal(f"{name} excerpt/raw activation", runtime["activation_kind"], static["activation_kind"])
                audit.equal(f"{name} excerpt/raw attention", runtime["attention_policy"], static["attention_policy"])
                audit.equal(
                    f"{name} excerpt/raw LayerNorm",
                    runtime["layer_norm_policy"],
                    raw["runtime_metadata"]["spu"]["spu_layer_norm_policy"],
                )
                audit.equal(
                    f"{name} excerpt/raw parameter mode",
                    runtime["params_mode"],
                    raw["runtime_metadata"]["spu"]["spu_params_mode"],
                )
                audit.equal(f"{name} excerpt/raw reveal policy", runtime["reveal_policy"], raw["privacy_facts"]["reveal_policy"])
            else:
                audit.equal(
                    f"{name} excerpt/raw parameter retention",
                    runtime["parameter_retention_ratio"],
                    raw["report_scope"]["parameter_retention_ratio"],
                )
                audit.equal(
                    f"{name} excerpt/raw reference comparison",
                    excerpt["reference_comparison"],
                    {
                        "argmax_match_ratio": raw["reference_comparison"]["argmax_match_ratio"],
                        "threshold_match_ratio": raw["reference_comparison"]["threshold_match_ratio"],
                    },
                )
        else:
            audit.note(f"未找到可选原始备份 {raw_path}；已使用仓库内数据摘录及其来源 SHA-256 审计 {name} 数据。")


def audit_secure_runs(audit: Audit, medical: dict[str, Any], finance: dict[str, Any]) -> None:
    audit.equal("medical sample count", medical["sample_count"], 32)
    audit.equal("medical elapsed/sample arithmetic", medical["sec_per_sample"], medical["elapsed_sec"] / 32)
    audit.equal("medical network byte/GiB arithmetic", medical["network"]["total_gib"], medical["network"]["total_bytes"] / 2**30)
    audit.equal("medical network per-sample arithmetic", medical["network"]["per_sample_gib"], medical["network"]["total_gib"] / 32)
    audit.equal("medical loopback TX single-count", medical["network"]["total_bytes"], medical["network"]["tx_delta_bytes"])
    audit.equal("medical loopback RX mirrors TX", medical["network"]["rx_delta_bytes"], medical["network"]["tx_delta_bytes"])
    audit.equal("medical deployment threshold accuracy", medical["threshold_accuracy"], 0.9375)
    audit.equal("medical deployment AUC", medical["auc"], 0.96484375)
    audit.check("medical finite outputs", medical["finite_logits"] and medical["finite_probabilities"])

    metadata = medical["runtime"]
    audit.equal("medical base_rate", metadata["base_rate"], 0.7)
    audit.check("medical token ratios", all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(metadata["token_ratio"], [0.7, 0.49, 0.343])))
    audit.equal("medical token keep counts", metadata["token_keep_counts"], [137, 96, 67])
    audit.equal("medical effective depth", metadata["effective_depth"], 10)
    audit.equal("medical activation", metadata["activation_kind"], "fixed_square")
    audit.equal("medical attention", metadata["attention_policy"], "uniform")
    audit.equal("medical SPU batch size", metadata["spu_batch_size"], 16)
    audit.equal("medical LayerNorm policy", metadata["layer_norm_policy"], "exact")
    audit.equal("medical secret parameter mode", metadata["params_mode"], "secret")
    audit.equal("medical reveal policy", metadata["reveal_policy"], "final_logits_only")

    audit.equal("finance sample count", finance["sample_count"], 8)
    audit.equal("finance elapsed/sample arithmetic", finance["sec_per_sample"], finance["elapsed_sec"] / 8)
    audit.equal("finance network byte/GiB arithmetic", finance["network"]["total_gib"], finance["network"]["total_bytes"] / 2**30)
    audit.equal("finance network per-sample arithmetic", finance["network"]["per_sample_gib"], finance["network"]["total_gib"] / 8)
    audit.equal("finance loopback TX single-count", finance["network"]["total_bytes"], finance["network"]["tx_delta_bytes"])
    audit.equal("finance loopback RX mirrors TX", finance["network"]["rx_delta_bytes"], finance["network"]["tx_delta_bytes"])
    audit.equal("finance argmax match", finance["reference_comparison"]["argmax_match_ratio"], 1.0)
    audit.equal("finance threshold match", finance["reference_comparison"]["threshold_match_ratio"], 1.0)
    audit.equal("finance parameter retention", finance["runtime"]["parameter_retention_ratio"], 0.6839)
    audit.equal("finance SPU batch size", finance["runtime"]["spu_batch_size"], 8)
    audit.check("finance finite outputs", finance["finite_logits"] and finance["finite_probabilities"])


def audit_fullval_and_figures(
    audit: Audit,
    calibration: dict[str, Any],
    auc_reference: dict[str, Any],
    demo: dict[str, Any],
    figure_data: dict[str, Any],
) -> None:
    audit.equal("fullval sample count", calibration["sample_count"], 524)
    audit.equal("fullval AUC sample count", auc_reference["sample_count"], 524)
    for key in ("argmax_accuracy", "best_threshold", "best_threshold_accuracy"):
        audit.equal(f"fullval calibration/AUC {key}", calibration[key], auc_reference[key])
    external = demo["external_comparison"]
    audit.equal("demo fullval threshold accuracy", external["transshield_verified"]["threshold_accuracy"], round(calibration["best_threshold_accuracy"] * 100, 4))
    audit.equal("demo fullval argmax accuracy", external["transshield_verified"]["argmax_accuracy"], round(calibration["argmax_accuracy"] * 100, 4))
    audit.equal("demo fullval AUC", external["transshield_verified"]["auc"], auc_reference["auc"])

    probability = figure_data["probability_distribution"]
    dynamic = read_json(REPO / probability["dynamic_prediction_json"])
    values = [float(item[1]) for item in dynamic["prediction_preview"]["probabilities"]]
    image_paths = (REPO / probability["dynamic_image_list"]).read_text(encoding="utf-8").splitlines()
    labels = [int(Path(item).parent.name) for item in image_paths if item.strip()]
    audit.equal("Figure 4-3 dynamic probability count", len(values), 524)
    audit.equal("Figure 4-3 dynamic label count", len(labels), 524)
    audit.equal("Figure 4-3 dynamic argmax recomputation", accuracy(values, labels, 0.5), calibration["argmax_accuracy"], tol=1e-7)
    audit.equal("Figure 4-3 dynamic threshold recomputation", accuracy(values, labels, calibration["best_threshold"]), calibration["best_threshold_accuracy"], tol=1e-7)
    audit.equal("Figure 4-3 dynamic AUC recomputation", binary_auc(values, labels), auc_reference["auc"], tol=1e-7)

    dense_values: list[float] = []
    dense_labels: list[int] = []
    with (REPO / probability["densenet_prediction_csv"]).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            dense_values.append(float(row["prob_positive"]))
            dense_labels.append(int(row["label"]))
    dense_summary = read_json(REPO / probability["densenet_summary_json"])
    audit.equal("Figure 4-3 DenseNet sample count", len(dense_values), dense_summary["val_sample_count"])
    audit.equal("Figure 4-3 DenseNet argmax recomputation", accuracy(dense_values, dense_labels, 0.5), dense_summary["best_val_argmax_accuracy"])
    audit.equal("Figure 4-3 DenseNet threshold recomputation", accuracy(dense_values, dense_labels, dense_summary["best_threshold"]), dense_summary["best_val_threshold_accuracy"])
    audit.equal("Figure 4-3 DenseNet AUC recomputation", binary_auc(dense_values, dense_labels), dense_summary["best_val_auc"], tol=1e-9)

    rows = figure_data["baseline_comparison"]
    expected_rows = external["additional_rows"]
    for row, expected in zip(rows, expected_rows):
        audit.equal(f"Figure 4-2 {row['method']} threshold", row["threshold_accuracy"], round(expected["threshold_accuracy"], 4))
        audit.equal(f"Figure 4-2 {row['method']} AUC", row["auc"], round(expected["auc"], 4))
    audit.equal("Figure 4-2 citation label", rows[2]["method"], "MPCViT [30]")
    audit.equal("Figure 4-2 dynamic-static delta", round(rows[0]["threshold_accuracy"] - rows[1]["threshold_accuracy"], 4), 0.7634)
    audit.equal("Figure 4-2 MPCViT gap", round(rows[2]["threshold_accuracy"] - rows[0]["threshold_accuracy"], 4), 4.2366)
    audit.equal("Figure 4-2 DeiT gap", round(rows[3]["threshold_accuracy"] - rows[0]["threshold_accuracy"], 4), 3.2419)
    audit.equal("Figure 4-2 original DynamicViT delta", round(rows[0]["threshold_accuracy"] - rows[4]["threshold_accuracy"], 4), 17.3664)

    scan = figure_data["base_rate_scan"]
    selected = scan["base_rate"].index(scan["selected_base_rate"])
    audit.equal("Figure 4-1 selected threshold accuracy", scan["threshold_accuracy"][selected], 92.7481)
    audit.equal("Figure 4-1 selected AUC", scan["auc"][selected], 0.9639)
    audit.check("Figure 4-1 selected base_rate maximizes threshold accuracy", scan["threshold_accuracy"][selected] == max(scan["threshold_accuracy"]))
    audit.check("Figure 4-1 selected base_rate maximizes AUC", scan["auc"][selected] == max(scan["auc"]))
    audit.equal("Figure 4-1 selected keep-rate label", scan["three_stage_keep_rate"][selected], "0.70 / 0.49 / 0.343")

    ablation = figure_data["ablation"]
    audit.equal("Figure 4-8 dynamic-static ablation delta", round(ablation[0]["threshold_accuracy"] - ablation[1]["threshold_accuracy"], 4), 3.0534)
    audit.equal("Figure 4-8 dynamic-original delta", round(ablation[0]["threshold_accuracy"] - ablation[2]["threshold_accuracy"], 4), 17.3664)


def audit_proxy_and_chart(
    audit: Audit,
    medical: dict[str, Any],
    finance: dict[str, Any],
    demo: dict[str, Any],
    chart: dict[str, Any],
) -> None:
    transshield = find_profile(demo, "transshield_ops_same_shape_proxy")["metrics"]
    baseline = find_profile(demo, "baseline_ops_same_shape_proxy")["metrics"]
    time_ratio = transshield["total_time_mean_sec"] / baseline["total_time_mean_sec"]
    comm_ratio = transshield["module_comm_mib_mean"] / baseline["module_comm_mib_mean"]
    audit.equal("same-shape Transshield time", transshield["total_time_mean_sec"], 8.10450275739034)
    audit.equal("same-shape baseline time", baseline["total_time_mean_sec"], 15.336534976959229)
    audit.equal("same-shape Transshield communication", transshield["module_comm_mib_mean"], 881.049072265625)
    audit.equal("same-shape baseline communication", baseline["module_comm_mib_mean"], 5918.689208984375)
    audit.equal("same-shape time reduction", round((1 - time_ratio) * 100, 2), 47.16)
    audit.equal("same-shape communication reduction", round((1 - comm_ratio) * 100, 2), 85.11)
    audit.equal("same-shape report time ratio", round(time_ratio, 3), 0.528)
    audit.equal("same-shape report communication ratio", round(comm_ratio, 3), 0.149)

    expected_metrics = {
        "sample_count": [medical["sample_count"], finance["sample_count"]],
        "total_seconds": [medical["elapsed_sec"], finance["elapsed_sec"]],
        "seconds_per_sample": [medical["sec_per_sample"], finance["sec_per_sample"]],
        "communication_gib": [medical["network"]["total_gib"], finance["network"]["total_gib"]],
        "communication_per_sample_gib": [medical["network"]["per_sample_gib"], finance["network"]["per_sample_gib"]],
    }
    metrics = {item["key"]: item for item in chart["metrics"]}
    for key, values in expected_metrics.items():
        for index, value in enumerate(values):
            audit.equal(f"Figure 4-4 {key}[{index}]", metrics[key]["values"][index], value)
            expected_display = str(int(value)) if key == "sample_count" else f"{value:.2f}"
            audit.equal(f"Figure 4-4 {key}[{index}] display", metrics[key]["display"][index], expected_display)


def audit_robustness(audit: Audit) -> None:
    fuzz = read_json(FUZZ)
    guard = read_json(GUARD)
    audit.check("protocol fuzz overall pass", fuzz["passed"])
    audit.equal("protocol fuzz case count", len(fuzz["results"]), 13)
    audit.check("protocol fuzz all cases pass", all(item["passed"] for item in fuzz["results"]))
    audit.check("guard stress overall pass", guard["passed"])
    audit.equal("guard stress case count", len(guard["checks"]), 4)
    audit.check("guard stress all cases pass", all(item["passed"] for item in guard["checks"]))
    all_cases = fuzz["results"] + guard["checks"]
    audit.equal("robustness total case count", len(all_cases), 17)
    audit.check("robustness no FD leak", all(item["system_state"]["delta"]["fd_count"] == 0 for item in all_cases))
    audit.check("robustness no socket-FD leak", all(item["system_state"]["delta"]["socket_fd_count"] == 0 for item in all_cases))
    expected_rss = [0, 0, 152, 0, 1024, 5120, 192, 0, 0, 0, 0, 6604, 0, 30632, 16032, 47004, 128]
    actual_rss = [item["system_state"]["delta"]["rss_kib"] for item in all_cases]
    audit.equal("Appendix B RSS deltas", actual_rss, expected_rss)


def audit_algorithm_counts(audit: Audit, medical: dict[str, Any]) -> None:
    ratios = [0.7, 0.49, 0.343]
    expected = [int(196 * ratio) for ratio in ratios]
    audit.equal("token floor arithmetic", expected, [137, 96, 67])
    audit.equal("token formula/runtime alignment", expected, medical["runtime"]["token_keep_counts"])
    source = (REPO / "integrations/transshield_runtime/e2e_secure_vit/static_vit_params.py").read_text(encoding="utf-8")
    audit.check("token implementation uses int floor", "tuple(int(init_n * r) for r in token_ratio)" in source)


def audit_appendix_inputs(
    audit: Audit,
    medical: dict[str, Any],
    demo: dict[str, Any],
    calibration: dict[str, Any],
    figure_data: dict[str, Any],
) -> None:
    training = read_json(TRAINING_ARGS)
    expected_training = {
        "epochs": 8,
        "batch_size": 32,
        "num_workers": 4,
        "device": "cuda",
        "base_rate": 0.7,
        "secure_static_train_depth": 12,
        "cls_distill_weight": 1.0,
        "token_distill_weight": 0.02,
    }
    for key, value in expected_training.items():
        audit.equal(f"Appendix C offline training {key}", training[key], value)
    sample = medical["first_sample"]
    audit.equal("Appendix C report sample path", sample["relative_path"], "0/00003.png")
    audit.equal("Appendix C report sample threshold prediction", sample["threshold_prediction"], 0)
    audit.equal("Appendix C report sample positive probability", sample["probabilities"][1], 0.5356374382972717)
    generated = HERE / "output/intermediate/strict_format/generated_ui_snapshots"
    expected_sizes = {
        "admin_overview_generated.png": (1600, 962),
        "admin_task_create_generated.png": (1280, 1002),
        "admin_task_detail_generated.png": (1680, 818),
        "admin_model_assets_generated.png": (1680, 789),
        "admin_evidence_generated.png": (1680, 1002),
        "pruning_overview_generated.png": (1680, 1577),
        "user_report_generated.png": (1836, 1182),
    }
    for filename, size in expected_sizes.items():
        path = generated / filename
        audit.check(f"Appendix C generated image exists: {filename}", path.is_file())
        if path.is_file():
            pixmap = fitz.Pixmap(str(path))
            audit.equal(f"Appendix C generated image size: {filename}", (pixmap.width, pixmap.height), size)
    generator_source = (HERE / "generate_report_ui_snapshots.py").read_text(encoding="utf-8")
    for stale in ("89.06", "84.47", "146 / 196", "103 / 196", "72 / 196", "3096.742", "1.17 GiB", "88.17"):
        audit.check(f"Appendix generator excludes stale {stale}", stale not in generator_source)
    comparison = demo["external_comparison"]["additional_rows"]
    scan = figure_data["base_rate_scan"]
    scan_index = scan["base_rate"].index(0.8)
    audit.equal("Appendix C task dynamic value", round(calibration["best_threshold_accuracy"] * 100, 2), 92.75)
    audit.equal("Appendix C task static value", round(comparison[1]["threshold_accuracy"], 2), 91.98)
    audit.equal("Appendix C task base-rate scan value", round(scan["threshold_accuracy"][scan_index], 2), 89.89)
    audit.equal("Appendix C task plaintext reference value", round(comparison[-1]["threshold_accuracy"], 2), 75.38)


def audit_pdf(audit: Audit) -> None:
    audit.check("final PDF exists", FINAL_PDF.is_file())
    if not FINAL_PDF.is_file():
        return
    doc = fitz.open(FINAL_PDF)
    audit.equal("final PDF page count", doc.page_count, 93)
    text = "\n".join(page.get_text("text") for page in doc).replace("\xa0", " ")
    required = [
        "47.16%",
        "15.3365",
        "8.1045",
        "30.13 秒/样本",
        "42.57 GiB",
        "0.96484",
        "TX增量一次",
        "92.7481%和AUC 0.9639来自524条全量验证样本",
    ]
    for value in required:
        audit.check(f"PDF contains {value}", value in text)
    forbidden = ["46.62%", "15.1696", "8.0973", "28.54 秒/样本", "40.49 GiB", "0.98438", "89.06", "84.47"]
    for value in forbidden:
        audit.check(f"PDF excludes stale {value}", value not in text)
    page59 = doc[58].get_text("text")
    audit.check("PDF page 59 token counts 137/96/67", all(re.search(rf"=\s*{value}[,\.]", page59) for value in (137, 96, 67)))
    page69 = doc[68].get_text("text")
    audit.check("PDF page 69 no detached punctuation blocks", not any(line.strip() in {"、", "；"} for line in page69.splitlines()))
    doc.close()


def main() -> None:
    audit = Audit()
    vps = read_json(VPS_DATA)
    medical = vps["medical"]
    finance = vps["finance"]
    demo = read_json(DEMO)
    calibration = read_json(CALIBRATION)
    auc_reference = read_json(AUC_REFERENCE)
    figure_data = read_json(HERE / "report_figure_data.json")
    chart = read_json(HERE / "performance_chart_data.json")

    audit_vps_evidence(audit, vps)
    audit_secure_runs(audit, medical, finance)
    audit_fullval_and_figures(audit, calibration, auc_reference, demo, figure_data)
    audit_proxy_and_chart(audit, medical, finance, demo, chart)
    audit_robustness(audit)
    audit_algorithm_counts(audit, medical)
    audit_appendix_inputs(audit, medical, demo, calibration, figure_data)
    audit_pdf(audit)

    audit.note("图4-5、图4-6及表4-10沿用原始报告冻结数据；最终仓没有对应独立原始跑数文件，本审计确认其内部数值与原报告一致，但不能替代重新实测。")
    audit.finish()


if __name__ == "__main__":
    main()
