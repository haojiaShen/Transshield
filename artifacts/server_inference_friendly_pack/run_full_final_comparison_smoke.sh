#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/load_local_env.sh"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-8}"
# load_local_env 会把两个变量默认导成 0；smoke 必须显式覆盖成小样本，
# 否则 bash 会把字符串 "0" 当成“已设置”，导致退回全量验证集。
export PLAINTEXT_MAX_SAMPLES="$SMOKE_MAX_SAMPLES"
export SECURE_MAX_SAMPLES="$SMOKE_MAX_SAMPLES"

echo "[smoke] 运行小样本链路验证。"
echo "[smoke] 本次将 plaintext 与 secure 输入都截断到前 ${SMOKE_MAX_SAMPLES} 个样本。"
echo "[smoke] smoke 只用于验证脚本和闭环能否跑通，不用于判断最终模型性能。"

"$SCRIPT_DIR/run_full_final_comparison_suite.sh"

echo "[smoke] 校验 smoke 输出样本数契约。"
"${PYTHON_BIN:-python}" - "$SECURE_RUN_DIR" "$SMOKE_MAX_SAMPLES" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
expected_max = int(sys.argv[2])


def load_json_count(name: str):
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing smoke artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = payload.get("sample_count")
    if not isinstance(count, int):
        raise ValueError(f"{path} missing integer sample_count")
    if count <= 0:
        raise ValueError(f"{path} has non-positive sample_count={count}")
    if count > expected_max:
        raise ValueError(
            f"{path} sample_count={count} exceeds SMOKE_MAX_SAMPLES={expected_max}"
        )
    return count


def csv_rows(name: str):
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing smoke artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.reader(handle)) - 1
    if row_count < 0:
        raise ValueError(f"{path} has invalid csv row count={row_count}")
    return row_count


json_counts = {
    "baseline_eval": load_json_count("plaintext_baseline_eval.json"),
    "modified_eval": load_json_count("plaintext_modified_eval.json"),
    "secure_input": load_json_count("stage2_secure_network_kth_input_smoke8.json"),
    "secure_replay": load_json_count("pipeline_inference_replay_summary.json"),
    "secure_compare": load_json_count("plaintext_vs_secure_score_compare.json"),
}

if len(set(json_counts.values())) != 1:
    raise ValueError(f"smoke sample_count mismatch across artifacts: {json_counts}")

baseline_rows = csv_rows("plaintext_baseline_eval.csv")
modified_rows = csv_rows("plaintext_modified_eval.csv")
expected_rows = json_counts["baseline_eval"]
if baseline_rows != expected_rows:
    raise ValueError(
        f"plaintext_baseline_eval.csv rows={baseline_rows} != json sample_count={expected_rows}"
    )
if modified_rows != json_counts["modified_eval"]:
    raise ValueError(
        f"plaintext_modified_eval.csv rows={modified_rows} != json sample_count={json_counts['modified_eval']}"
    )

print(
    json.dumps(
        {
            "run_dir": str(run_dir),
            "smoke_max_samples": expected_max,
            "validated_sample_count": expected_rows,
            "validated_artifacts": sorted(json_counts.keys()),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
)
PY
