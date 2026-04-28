#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No usable python interpreter found." >&2
  return 1
}

choose_bazel() {
  if [[ -n "${SPU_BAZEL_BIN:-}" ]]; then
    printf '%s\n' "$SPU_BAZEL_BIN"
    return 0
  fi
  local candidates=(
    "/data/wyb/bin/bazel"
    "/usr/local/bin/bazel"
    "/usr/bin/bazel"
    "bazel"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
      printf '%s\n' "$candidate"
      return 0
    fi
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "No usable bazel binary found." >&2
  return 1
}

require_bazel_major_6() {
  local version_output
  version_output="$("$SPU_BAZEL_BIN" --output_user_root="$SPU_BAZEL_OUTPUT_USER_ROOT" version 2>/dev/null || true)"
  local build_label
  build_label="$(printf '%s\n' "$version_output" | awk -F': ' '/Build label/ {print $2; exit}')"
  if [[ -z "$build_label" ]]; then
    echo "Cannot determine bazel version from: $SPU_BAZEL_BIN" >&2
    return 1
  fi
  local major="${build_label%%.*}"
  if [[ "$major" != "6" ]]; then
    echo "Unsupported bazel version: $build_label ($SPU_BAZEL_BIN). Expected major version 6." >&2
    echo "Hint: on the server, try SPU_BAZEL_BIN=/usr/local/bin/bazel if that is the 6.5.0 binary." >&2
    return 1
  fi
}

choose_cmake() {
  if [[ -n "${SPU_CMAKE_BIN:-}" ]]; then
    printf '%s\n' "$SPU_CMAKE_BIN"
    return 0
  fi
  local candidates=(
    "/data/wyb/bin/cmake"
    "/data/wyb/tools/cmake-3.28/bin/cmake"
    "/data/wyb/cmake-3.28/bin/cmake"
    "cmake"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
      printf '%s\n' "$candidate"
      return 0
    fi
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "No usable cmake binary found." >&2
  return 1
}

require_cmake_pre4() {
  local version_output
  version_output="$("$SPU_CMAKE_BIN" --version 2>/dev/null || true)"
  local first_line
  first_line="$(printf '%s\n' "$version_output" | head -n 1)"
  local version
  version="$(printf '%s\n' "$first_line" | awk '{print $3}')"
  if [[ -z "$version" ]]; then
    echo "Cannot determine cmake version from: $SPU_CMAKE_BIN" >&2
    return 1
  fi
  local major="${version%%.*}"
  if [[ "$major" -ge 4 ]]; then
    echo "Unsupported cmake version: $version ($SPU_CMAKE_BIN). Expected CMake 3.x, preferably 3.28.x." >&2
    echo "Hint: set SPU_CMAKE_BIN=/path/to/cmake-3.28.x/bin/cmake." >&2
    return 1
  fi
}

setup_defaults() {
  PYTHON_BIN="$(choose_python)"
  SPU_BAZEL_BIN="$(choose_bazel)"
  SPU_CMAKE_BIN="$(choose_cmake)"
  SPU_SRC_ROOT="${SPU_SRC_ROOT:-/data/wyb/spu_src_diag/spu}"
  SPU_CONDA_PREFIX="${SPU_CONDA_PREFIX:-/data/wyb/conda_envs/transshield}"
  SPU_BAZEL_DISTDIR="${SPU_BAZEL_DISTDIR:-/data/wyb/bazel_clean/distdir}"
  SPU_BAZEL_OUTPUT_USER_ROOT="${SPU_BAZEL_OUTPUT_USER_ROOT:-/data/wyb/bazel_clean/output_root}"
  SPU_BAZEL_REPOSITORY_CACHE="${SPU_BAZEL_REPOSITORY_CACHE:-/data/wyb/bazel_clean/repository_cache}"
  SPU_TMPDIR="${SPU_TMPDIR:-/data/wyb/bazel_clean/tmp}"
  SPU_PROXY_HOST="${SPU_PROXY_HOST:-127.0.0.1}"
  SPU_PROXY_PORT="${SPU_PROXY_PORT:-17897}"
  SPU_BUILD_LOG="${SPU_BUILD_LOG:-/data/wyb/bazel_clean/logs/spu_build_cmake328.log}"
  SPU_REPORT_DIR="${SPU_REPORT_DIR:-$REPO_ROOT/artifacts/spu_build_reports}"
  SPU_WRAPPER_DIR="${SPU_WRAPPER_DIR:-$REPO_ROOT/tools/generated/spu_toolchain_wrappers}"
  SPU_REAL_CC="${SPU_REAL_CC:-$SPU_CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc}"
  SPU_REAL_CXX="${SPU_REAL_CXX:-$SPU_CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}"
  SPU_LD="${SPU_LD:-$SPU_CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ld.bfd}"
  SPU_ASM_NASM="${SPU_ASM_NASM:-$SPU_CONDA_PREFIX/bin/nasm}"
  SPU_BAZEL_TARGETS="${SPU_BAZEL_TARGETS:-//spu:libspu.so //spu:libpsi.so}"
  SPU_EXTRA_PATTERNS="${SPU_EXTRA_PATTERNS:-}"
  SPU_JOBS="${SPU_JOBS:-50}"
  SPU_LOCAL_CPU_RESOURCES="${SPU_LOCAL_CPU_RESOURCES:-50}"
  SPU_RUN_PATCH="${SPU_RUN_PATCH:-1}"
  SPU_IO_CC="${SPU_IO_CC:-}"
  SPU_PATCH_LOG="${SPU_PATCH_LOG:-$SPU_REPORT_DIR/patch_status.txt}"
  SPU_BAZEL_EXTRA_ARGS="${SPU_BAZEL_EXTRA_ARGS:-}"
  SPU_REPORT_STEM="${SPU_REPORT_STEM:-$(date +%Y%m%d_%H%M%S)}"
}

ensure_prereqs() {
  mkdir -p "$SPU_REPORT_DIR" "$SPU_BAZEL_DISTDIR" "$SPU_BAZEL_OUTPUT_USER_ROOT" "$SPU_BAZEL_REPOSITORY_CACHE" "$SPU_TMPDIR"
  mkdir -p "$(dirname "$SPU_BUILD_LOG")"

  if [[ ! -d "$SPU_SRC_ROOT" ]]; then
    echo "Missing SPU source root: $SPU_SRC_ROOT" >&2
    exit 1
  fi
  if [[ ! -x "$SPU_REAL_CC" ]]; then
    echo "Missing compiler: $SPU_REAL_CC" >&2
    exit 1
  fi
  if [[ ! -x "$SPU_REAL_CXX" ]]; then
    echo "Missing compiler: $SPU_REAL_CXX" >&2
    exit 1
  fi

  require_bazel_major_6
  require_cmake_pre4
}

prepare_wrappers_and_patch() {
  "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_prepare_spu_compiler_wrappers.py" \
    --output-dir "$SPU_WRAPPER_DIR" \
    --real-cc "$SPU_REAL_CC" \
    --real-cxx "$SPU_REAL_CXX"

  if [[ "$SPU_RUN_PATCH" == "1" ]]; then
    PATCH_ARGS=()
    if [[ -n "$SPU_IO_CC" ]]; then
      PATCH_ARGS+=(--io-cc "$SPU_IO_CC")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_patch_spu_colocated_sync_profile.py" \
      "${PATCH_ARGS[@]}" | tee "$SPU_PATCH_LOG"
  fi
}

export_build_env() {
  export PATH="$(dirname "$SPU_CMAKE_BIN"):$SPU_WRAPPER_DIR:$SPU_CONDA_PREFIX/bin:$PATH"
  export CMAKE="$SPU_CMAKE_BIN"
  export CMAKE_COMMAND="$SPU_CMAKE_BIN"
  export CC="$SPU_WRAPPER_DIR/x86_64-conda-linux-gnu-cc"
  export CXX="$SPU_WRAPPER_DIR/x86_64-conda-linux-gnu-c++"
  export LD="$SPU_LD"
  export ASM_NASM="$SPU_ASM_NASM"
  export CFLAGS="-B$SPU_CONDA_PREFIX/bin"
  export CXXFLAGS="-B$SPU_CONDA_PREFIX/bin"
  export LDFLAGS="-B$SPU_CONDA_PREFIX/bin"
  export TRANSHIELD_TMP_ROOT="$SPU_TMPDIR"
  source "$REPO_ROOT/scripts/use_data_tmp_env.sh"

  HTTP_PROXY="http://$SPU_PROXY_HOST:$SPU_PROXY_PORT"
  HTTPS_PROXY="$HTTP_PROXY"
  ALL_PROXY="$HTTP_PROXY"
  export HTTP_PROXY HTTPS_PROXY ALL_PROXY
}

write_preflight_log() {
  {
    echo "[preflight] date=$(date -Is)"
    echo "[preflight] pwd=$PWD"
    echo "[preflight] bazel_bin=$SPU_BAZEL_BIN"
    echo "[preflight] bazel_path=$(command -v "$SPU_BAZEL_BIN" || true)"
    echo "[preflight] cmake_bin=$SPU_CMAKE_BIN"
    echo "[preflight] cmake_path=$(command -v cmake || true)"
    echo "[preflight] cmake_env=$CMAKE"
    echo "[preflight] cmake_command_env=$CMAKE_COMMAND"
    echo "[preflight] cmake_version_begin"
    "$SPU_CMAKE_BIN" --version || true
    echo "[preflight] cmake_version_end"
    echo "[preflight] cc=$CC"
    echo "[preflight] cxx=$CXX"
    echo "[preflight] ld=$LD"
    echo "[preflight] distdir=$SPU_BAZEL_DISTDIR"
    echo "[preflight] output_user_root=$SPU_BAZEL_OUTPUT_USER_ROOT"
    echo "[preflight] repository_cache=$SPU_BAZEL_REPOSITORY_CACHE"
    echo "[preflight] tmpdir=$SPU_TMPDIR"
    echo "[preflight] test_tmpdir=${TEST_TMPDIR:-}"
    echo "[preflight] report_stem=$SPU_REPORT_STEM"
    echo "[preflight] targets=$SPU_BAZEL_TARGETS"
    echo "[preflight] extra_args=$SPU_BAZEL_EXTRA_ARGS"
    echo "[preflight] git_head=$(git rev-parse --short HEAD 2>/dev/null || true)"
    echo "[preflight] io_cc_candidates:"
    find "$SPU_SRC_ROOT" -path '*libspu/device/io.cc' 2>/dev/null | sed 's/^/  /'
    echo "[preflight] type_inference_candidates:"
    find "$SPU_SRC_ROOT" -path '*type_inference.cc' 2>/dev/null | sed 's/^/  /'
    echo "[preflight] beaver_ttp_candidates:"
    find "$SPU_SRC_ROOT" -path '*beaver_ttp.cc' 2>/dev/null | sed 's/^/  /'
    echo "[preflight] bazel_version_begin"
    "$SPU_BAZEL_BIN" --output_user_root="$SPU_BAZEL_OUTPUT_USER_ROOT" version || true
    echo "[preflight] bazel_version_end"
    echo
  } > "$SPU_BUILD_LOG"
}

prepare_build_args() {
  read -r -a SPU_TARGET_ARGS <<< "$SPU_BAZEL_TARGETS"
  EXTRA_ARG_FILE="$SPU_REPORT_DIR/${SPU_REPORT_STEM}_extra_args.txt"
  printf '%s\n' "$SPU_BAZEL_EXTRA_ARGS" > "$EXTRA_ARG_FILE"
  read -r -a SPU_EXTRA_ARG_ARRAY <<< "$SPU_BAZEL_EXTRA_ARGS"

  BAZEL_BUILD_ARGS=(
    "--output_user_root=$SPU_BAZEL_OUTPUT_USER_ROOT"
    "--host_jvm_args=-Dhttps.proxyHost=$SPU_PROXY_HOST"
    "--host_jvm_args=-Dhttps.proxyPort=$SPU_PROXY_PORT"
    "--host_jvm_args=-Dhttp.proxyHost=$SPU_PROXY_HOST"
    "--host_jvm_args=-Dhttp.proxyPort=$SPU_PROXY_PORT"
    build
    "${SPU_TARGET_ARGS[@]}"
    -c
    opt
    "--distdir=$SPU_BAZEL_DISTDIR"
    "--repository_cache=$SPU_BAZEL_REPOSITORY_CACHE"
    "--repo_env=HTTPS_PROXY=$HTTPS_PROXY"
    "--repo_env=HTTP_PROXY=$HTTP_PROXY"
    "--repo_env=ALL_PROXY=$ALL_PROXY"
    "--repo_env=CMAKE=$CMAKE"
    "--repo_env=CMAKE_COMMAND=$CMAKE_COMMAND"
    "--repo_env=CC=$CC"
    "--repo_env=CXX=$CXX"
    "--repo_env=LD=$LD"
    "--repo_env=ASM_NASM=$ASM_NASM"
    "--repo_env=CFLAGS=$CFLAGS"
    "--repo_env=CXXFLAGS=$CXXFLAGS"
    "--repo_env=LDFLAGS=$LDFLAGS"
    "--action_env=PATH=$PATH"
    "--action_env=CMAKE=$CMAKE"
    "--action_env=CMAKE_COMMAND=$CMAKE_COMMAND"
    "--action_env=CC=$CC"
    "--action_env=CXX=$CXX"
    "--action_env=LD=$LD"
    "--action_env=ASM_NASM=$ASM_NASM"
    "--action_env=CFLAGS=$CFLAGS"
    "--action_env=CXXFLAGS=$CXXFLAGS"
    "--action_env=LDFLAGS=$LDFLAGS"
    --copt=-Wno-error
    --host_copt=-Wno-error
    --verbose_failures
    -s
    "--jobs=$SPU_JOBS"
    "--local_cpu_resources=$SPU_LOCAL_CPU_RESOURCES"
    "${SPU_EXTRA_ARG_ARRAY[@]}"
  )
}

run_bazel_build() {
  set +e
  {
    echo "[build] command_begin"
    printf '%q ' "$SPU_BAZEL_BIN" "${BAZEL_BUILD_ARGS[@]}"
    echo
    echo "[build] command_end"
    set -x
    "$SPU_BAZEL_BIN" "${BAZEL_BUILD_ARGS[@]}"
  } 2>&1 | tee -a "$SPU_BUILD_LOG"
  BUILD_STATUS=${PIPESTATUS[0]}
  set -e

  printf '%s\n' "$BUILD_STATUS" > "$SPU_REPORT_DIR/${SPU_REPORT_STEM}_build_exit_code.txt"
}

extract_build_report() {
  PATTERN_ARGS=()
  if [[ -n "$SPU_EXTRA_PATTERNS" ]]; then
    while IFS= read -r pattern; do
      [[ -n "$pattern" ]] && PATTERN_ARGS+=(--pattern "$pattern")
    done < <(printf '%s\n' "$SPU_EXTRA_PATTERNS")
  fi

  "$PYTHON_BIN" "$REPO_ROOT/tools/transshield_extract_spu_build_errors.py" \
    "$SPU_BUILD_LOG" \
    "${PATTERN_ARGS[@]}" \
    --output-json "$SPU_REPORT_DIR/${SPU_REPORT_STEM}_build_errors.json" \
    --output-md "$SPU_REPORT_DIR/${SPU_REPORT_STEM}_build_errors.md"
}

main() {
  setup_defaults
  ensure_prereqs
  prepare_wrappers_and_patch
  export_build_env

  cd "$SPU_SRC_ROOT"
  "$SPU_BAZEL_BIN" --output_user_root="$SPU_BAZEL_OUTPUT_USER_ROOT" shutdown || true

  write_preflight_log
  prepare_build_args
  run_bazel_build
  extract_build_report

  echo
  echo "[info] build log: $SPU_BUILD_LOG"
  echo "[info] build report dir: $SPU_REPORT_DIR"
  echo "[info] build report stem: $SPU_REPORT_STEM"

  exit "$BUILD_STATUS"
}

main "$@"
