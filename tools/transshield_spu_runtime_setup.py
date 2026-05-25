import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_dependencies():
    required_modules = [
        "jax",
        "spu.utils.distributed",
    ]
    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            missing.append((module_name, f"{type(error).__name__}: {error}"))

    if not missing:
        return

    missing_text = ", ".join(f"{name} ({detail})" for name, detail in missing)
    raise RuntimeError(
        "Current Python runtime cannot start the SPU backend because required modules are missing: "
        f"{missing_text}. "
        "Use the Python environment that can import both `jax` and `spu.utils.distributed`."
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reserve_free_ports(count: int):
    sockets = []
    try:
        while len(sockets) < count:
            candidate_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            candidate_socket.bind(("127.0.0.1", 0))
            used_ports = {reserved_socket.getsockname()[1] for reserved_socket in sockets}
            if candidate_socket.getsockname()[1] in used_ports:
                candidate_socket.close()
                continue
            sockets.append(candidate_socket)
        return [reserved_socket.getsockname()[1] for reserved_socket in sockets]
    finally:
        for reserved_socket in sockets:
            reserved_socket.close()


def backup_file(path: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup_path = path.with_name(path.name + f".bak.{timestamp}")
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def rotate_log_file(path: Path):
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    rotated_path = path.with_name(path.name + f".prev.{timestamp}")
    path.replace(rotated_path)
    return rotated_path


def update_config(
    path: Path,
    ports,
    remove_unsupported_cheetah_fields: bool,
    make_backup: bool,
    disable_colocated_optimization: bool,
    allow_cheetah_mul_lsb_error: bool,
    cheetah_disable_matmul_pack: bool,
    cheetah_mixed_compare_mode: int,
):
    if make_backup and path.exists():
        backup_file(path)

    payload = load_json(path)
    node0_port, node1_port, internal0_port, internal1_port = ports

    payload["nodes"]["node:0"] = f"127.0.0.1:{node0_port}"
    payload["nodes"]["node:1"] = f"127.0.0.1:{node1_port}"

    spu_config = payload["devices"]["SPU"]["config"]
    spu_config["spu_internal_addrs"] = [
        f"127.0.0.1:{internal0_port}",
        f"127.0.0.1:{internal1_port}",
    ]

    if remove_unsupported_cheetah_fields:
        cheetah_config = spu_config.get("runtime_config", {}).get("cheetah_2pc_config", {})
        cheetah_config.pop("approx_less_precision", None)

    runtime_config = spu_config.get("runtime_config", {})
    cheetah_config = runtime_config.setdefault("cheetah_2pc_config", {})
    if not allow_cheetah_mul_lsb_error:
        cheetah_config["enable_mul_lsb_error"] = False
    if cheetah_disable_matmul_pack:
        cheetah_config["disable_matmul_pack"] = True
    else:
        cheetah_config.pop("disable_matmul_pack", None)
    if int(cheetah_mixed_compare_mode) > 0:
        cheetah_config["mixed_compare_mode"] = int(cheetah_mixed_compare_mode)
    else:
        cheetah_config.pop("mixed_compare_mode", None)

    if disable_colocated_optimization:
        runtime_config["experimental_enable_colocated_optimization"] = False

    write_json(path, payload)
    return payload


def parse_node_ports(config_payload):
    node_ports = []
    for node_id in sorted(config_payload["nodes"]):
        host, port_text = config_payload["nodes"][node_id].rsplit(":", 1)
        node_ports.append((node_id, host, int(port_text)))
    return node_ports


def wait_for_addresses(addresses, timeout_sec: float, label: str):
    deadline = time.time() + timeout_sec
    pending = {(name, host, port) for name, host, port in addresses}
    last_error = {}

    while pending and time.time() < deadline:
        for name, host, port in list(pending):
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    pending.remove((name, host, port))
            except OSError as error:
                last_error[name] = str(error)
        if pending:
            time.sleep(0.5)

    if pending:
        details = {name: last_error.get(name, "not checked") for name, _, _ in sorted(pending)}
        raise RuntimeError(f"{label} did not become reachable before timeout: {details}")


def iter_state_pids(state_payload):
    for process_info in state_payload.get("node_processes", []):
        pid = int(process_info.get("pid", 0) or 0)
        if pid > 0:
            yield pid


def terminate_pid(pid: int):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        pass

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        pass


def stop_existing_nodes(config_path: Path, config_arg: str, state_json: Path):
    if state_json.exists():
        try:
            state_payload = load_json(state_json)
        except Exception:
            state_payload = {}
        for pid in iter_state_pids(state_payload):
            terminate_pid(pid)
        time.sleep(1.0)

    patterns = [
        f"spu.utils.distributed -c {config_arg}",
        f"spu.utils.distributed -c {config_path}",
        f"spu.utils.distributed -c {config_path.resolve()}",
        "spu.utils.distributed -c configs/openbumblebee/2pc.json",
    ]
    for pattern in dict.fromkeys(patterns):
        subprocess.run(["pkill", "-f", pattern], check=False)

    time.sleep(1.0)


def start_node_process(config_path: Path, node_id: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    node_name = node_id.replace(":", "_")
    log_path = log_dir / f"{node_name}.log"
    rotate_log_file(log_path)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spu.utils.distributed",
                "-c",
                str(config_path),
                "start",
                "-n",
                node_id,
            ],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"node_id": node_id, "pid": process.pid, "log_path": str(log_path)}


def warmup_spu_runtime(config_path: Path):
    import numpy as np
    import spu.utils.distributed as ppd

    config_payload = load_json(config_path)
    ppd.init(config_payload["nodes"], config_payload["devices"])
    input_ref = ppd.device("P1")(lambda value: value)(np.asarray([1.0], dtype=np.float32))
    output_ref = ppd.device("SPU")(lambda value: value + 1.0)(input_ref)
    result = ppd.get(output_ref)
    return [float(item) for item in result.tolist()]


def wait_for_warmup(config_path: Path, timeout_sec: float):
    deadline = time.time() + timeout_sec
    last_error = ""

    while time.time() < deadline:
        try:
            return warmup_spu_runtime(config_path)
        except Exception as error:
            last_error = str(error)
            time.sleep(1.0)

    raise RuntimeError(f"SPU runtime warmup did not succeed before timeout: {last_error}")


def configure(args):
    config_path = Path(args.config).resolve()
    template_path = Path(args.template).resolve() if args.template else None
    ports = reserve_free_ports(4)
    updated_config = update_config(
        config_path,
        ports,
        remove_unsupported_cheetah_fields=args.remove_unsupported_cheetah_fields,
        make_backup=args.backup,
        disable_colocated_optimization=args.disable_colocated_optimization,
        allow_cheetah_mul_lsb_error=args.allow_cheetah_mul_lsb_error,
        cheetah_disable_matmul_pack=args.cheetah_disable_matmul_pack,
        cheetah_mixed_compare_mode=args.cheetah_mixed_compare_mode,
    )
    if template_path and template_path.exists() and template_path != config_path:
        update_config(
            template_path,
            ports,
            remove_unsupported_cheetah_fields=args.remove_unsupported_cheetah_fields,
            make_backup=args.backup,
            disable_colocated_optimization=args.disable_colocated_optimization,
            allow_cheetah_mul_lsb_error=args.allow_cheetah_mul_lsb_error,
            cheetah_disable_matmul_pack=args.cheetah_disable_matmul_pack,
            cheetah_mixed_compare_mode=args.cheetah_mixed_compare_mode,
        )

    state = {
        "config_path": str(config_path),
        "template_path": str(template_path) if template_path else "",
        "nodes": updated_config["nodes"],
        "spu_internal_addrs": updated_config["devices"]["SPU"]["config"]["spu_internal_addrs"],
        "remove_unsupported_cheetah_fields": args.remove_unsupported_cheetah_fields,
        "disable_colocated_optimization": args.disable_colocated_optimization,
        "allow_cheetah_mul_lsb_error": args.allow_cheetah_mul_lsb_error,
        "cheetah_disable_matmul_pack": args.cheetah_disable_matmul_pack,
        "cheetah_mixed_compare_mode": int(args.cheetah_mixed_compare_mode),
        "enable_mul_lsb_error": updated_config["devices"]["SPU"]["config"]["runtime_config"]
        .get("cheetah_2pc_config", {})
        .get("enable_mul_lsb_error"),
        "disable_matmul_pack": updated_config["devices"]["SPU"]["config"]["runtime_config"]
        .get("cheetah_2pc_config", {})
        .get("disable_matmul_pack", False),
        "mixed_compare_mode": updated_config["devices"]["SPU"]["config"]["runtime_config"]
        .get("cheetah_2pc_config", {})
        .get("mixed_compare_mode", 0),
        "node_processes": [],
    }
    if args.state_json:
        write_json(Path(args.state_json).resolve(), state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return updated_config, state


def start(args):
    ensure_runtime_dependencies()
    config_path = Path(args.config).resolve()
    state_json = Path(args.state_json).resolve()
    log_dir = Path(args.log_dir).resolve()
    if args.restart:
        stop_existing_nodes(config_path, args.config, state_json)
        time.sleep(args.stop_wait_sec)

    last_error = ""
    for attempt_index in range(args.warmup_attempts):
        if attempt_index > 0:
            stop_existing_nodes(config_path, args.config, state_json)
            time.sleep(args.stop_wait_sec)

        config_payload, state = configure(args)
        state["warmup_attempt"] = attempt_index + 1

        node_processes = [
            start_node_process(config_path, node_id, log_dir)
            for node_id in sorted(config_payload["nodes"])
        ]
        state["node_processes"] = node_processes
        write_json(state_json, state)

        try:
            wait_for_addresses(
                parse_node_ports(config_payload),
                args.startup_timeout_sec,
                "SPU node gRPC ports",
            )
            warmup_result = wait_for_warmup(config_path, args.startup_timeout_sec)
            state["warmup_result"] = warmup_result
            write_json(state_json, state)
            print(f"SPU nodes started and warmed up; state={state_json}")
            return
        except Exception as error:
            last_error = str(error)
            state["warmup_error"] = last_error
            write_json(state_json, state)
            stop_existing_nodes(config_path, args.config, state_json)

    raise RuntimeError(f"SPU runtime setup failed after {args.warmup_attempts} attempts: {last_error}")


def stop(args):
    stop_existing_nodes(Path(args.config).resolve(), args.config, Path(args.state_json).resolve())


def check(args):
    config_payload = load_json(Path(args.config).resolve())
    wait_for_addresses(parse_node_ports(config_payload), args.startup_timeout_sec, "SPU node gRPC ports")
    print("SPU node ports are reachable.")


def add_common_args(parser):
    parser.add_argument("--config", default="configs/openbumblebee/2pc.json")
    parser.add_argument("--template", default="configs/openbumblebee/2pc.template.json")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--state-json", default="logs/spu_runtime_ports.json")
    parser.add_argument("--remove-unsupported-cheetah-fields", action="store_true")
    parser.add_argument("--disable-colocated-optimization", action="store_true")
    parser.add_argument(
        "--allow-cheetah-mul-lsb-error",
        action="store_true",
        help="keep Cheetah enable_mul_lsb_error enabled instead of forcing the stable default false",
    )
    parser.add_argument(
        "--cheetah-disable-matmul-pack",
        action="store_true",
        help="force Cheetah matmul to bypass ciphertext packing for current runtime validation",
    )
    parser.add_argument(
        "--cheetah-mixed-compare-mode",
        type=int,
        default=0,
        help="set Cheetah mixed_compare_mode for current runtime validation; 0 keeps native path",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Configure and start Transshield colocated SPU runtime nodes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_configure = subparsers.add_parser("configure", help="rewrite SPU config with current free localhost ports")
    add_common_args(parser_configure)

    parser_start = subparsers.add_parser("start", help="configure ports, restart SPU nodes, and wait for readiness")
    add_common_args(parser_start)
    parser_start.add_argument("--restart", action="store_true")
    parser_start.add_argument("--log-dir", default="logs/spu_nodes")
    parser_start.add_argument("--stop-wait-sec", type=float, default=1.0)
    parser_start.add_argument("--startup-timeout-sec", type=float, default=30.0)
    parser_start.add_argument("--warmup-attempts", type=int, default=2)

    parser_stop = subparsers.add_parser("stop", help="stop SPU nodes recorded in the runtime state")
    add_common_args(parser_stop)

    parser_check = subparsers.add_parser("check", help="wait until configured SPU node ports are reachable")
    parser_check.add_argument("--config", default="configs/openbumblebee/2pc.json")
    parser_check.add_argument("--startup-timeout-sec", type=float, default=20.0)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "configure":
        configure(args)
    elif args.command == "start":
        start(args)
    elif args.command == "stop":
        stop(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
