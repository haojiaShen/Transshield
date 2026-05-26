import argparse
import importlib
import ipaddress
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARTY_NODE_ALIASES = {
    "hospital": "node:0",
    "p1": "node:0",
    "node:0": "node:0",
    "ai": "node:1",
    "ai-provider": "node:1",
    "provider": "node:1",
    "p2": "node:1",
    "node:1": "node:1",
}


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


def shell_quote(value: str):
    return shlex.quote(str(value))


def split_host_port(raw: str, label: str):
    value = str(raw or "").strip()
    if not value or ":" not in value:
        raise ValueError(f"{label} must use host:port, got {raw!r}")

    if value.startswith("["):
        bracket_index = value.find("]")
        if bracket_index < 0 or bracket_index + 1 >= len(value) or value[bracket_index + 1] != ":":
            raise ValueError(f"{label} must use [ipv6]:port for bracketed IPv6, got {raw!r}")
        host = value[1:bracket_index]
        port_text = value[bracket_index + 2 :]
    else:
        host, port_text = value.rsplit(":", 1)
    host = host.strip()
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f"{label} port must be an integer, got {port_text!r}") from error
    if not host:
        raise ValueError(f"{label} host is empty")
    if port <= 0 or port > 65535:
        raise ValueError(f"{label} port out of range: {port}")
    return host, port


def is_loopback_host(host: str):
    normalized = host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}:
        return True
    try:
        parsed = ipaddress.ip_address(normalized)
        return parsed.is_loopback or parsed.is_unspecified
    except ValueError:
        return False


def validate_remote_address(raw: str, label: str, allow_localhost: bool):
    host, _ = split_host_port(raw, label)
    if not allow_localhost and is_loopback_host(host):
        raise ValueError(
            f"{label} points to {host!r}; remote 2PC configs must use reachable party addresses. "
            "Pass --allow-localhost only for dry-run tests."
        )


def validate_distinct_addresses(addresses):
    seen = {}
    for label, raw in addresses:
        host, port = split_host_port(raw, label)
        key = (host.lower(), port)
        if key in seen:
            raise ValueError(f"{label} duplicates {seen[key]} at {host}:{port}")
        seen[key] = label


def normalize_party(raw: str):
    value = str(raw or "").strip().lower()
    try:
        return PARTY_NODE_ALIASES[value]
    except KeyError as error:
        supported = ", ".join(sorted(PARTY_NODE_ALIASES))
        raise ValueError(f"unknown party {raw!r}; expected one of: {supported}") from error


def party_label(node_id: str):
    return "hospital/P1" if node_id == "node:0" else "AI-provider/P2"


def node_index(node_id: str):
    try:
        return int(node_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid node id {node_id!r}") from error


def get_node_address(config_payload, node_id: str):
    nodes = config_payload.get("nodes", {})
    if node_id not in nodes:
        raise ValueError(f"config does not contain nodes.{node_id}")
    host, port = split_host_port(nodes[node_id], f"nodes.{node_id}")
    return node_id, host, port


def get_party_internal_address(config_payload, node_id: str):
    spu_config = config_payload.get("devices", {}).get("SPU", {}).get("config", {})
    internal_addrs = spu_config.get("spu_internal_addrs", [])
    index = node_index(node_id)
    if index >= len(internal_addrs):
        raise ValueError(f"config does not contain SPU internal address for {node_id}")
    return internal_addrs[index]


def validate_party_remote_config(config_payload, node_id: str, allow_localhost: bool):
    _, host, port = get_node_address(config_payload, node_id)
    validate_remote_address(f"{host}:{port}", f"nodes.{node_id}", allow_localhost)
    validate_remote_address(
        get_party_internal_address(config_payload, node_id),
        f"spu_internal_addrs[{node_index(node_id)}]",
        allow_localhost,
    )


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
        node_ports.append(get_node_address(config_payload, node_id))
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


def warmup(args):
    ensure_runtime_dependencies()
    config_path = Path(args.config).resolve()
    config_payload = load_json(config_path)
    wait_for_addresses(parse_node_ports(config_payload), args.startup_timeout_sec, "SPU node gRPC ports")
    warmup_result = wait_for_warmup(config_path, args.warmup_timeout_sec)
    print(json.dumps({"status": "ok", "warmup_result": warmup_result}, indent=2, sort_keys=True))


def resolve_party_state_path(state_json: str, node_id: str):
    if state_json:
        return Path(state_json).resolve()
    safe_node_id = node_id.replace(":", "_")
    return (REPO_ROOT / "logs" / f"spu_party_{safe_node_id}.json").resolve()


def stop_party_process(config_path: Path, config_arg: str, node_id: str, state_json: Path):
    state_payload = {}
    if state_json.exists():
        try:
            state_payload = load_json(state_json)
        except Exception:
            state_payload = {}
        for pid in iter_state_pids(state_payload):
            terminate_pid(pid)

    patterns = [
        f"spu.utils.distributed -c {config_arg} start -n {node_id}",
        f"spu.utils.distributed -c {config_path} start -n {node_id}",
        f"spu.utils.distributed -c {config_path.resolve()} start -n {node_id}",
    ]
    for pattern in dict.fromkeys(patterns):
        subprocess.run(["pkill", "-f", pattern], check=False)

    if state_payload:
        state_payload["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        state_payload["node_processes"] = []
        write_json(state_json, state_payload)


def start_party(args):
    config_path = Path(args.config).resolve()
    config_payload = load_json(config_path)
    node_id = normalize_party(args.party)
    validate_party_remote_config(config_payload, node_id, args.allow_localhost)
    ensure_runtime_dependencies()
    state_json = resolve_party_state_path(args.state_json, node_id)
    log_dir = Path(args.log_dir).resolve()

    if args.restart:
        stop_party_process(config_path, args.config, node_id, state_json)
        time.sleep(args.stop_wait_sec)

    process_info = start_node_process(config_path, node_id, log_dir)
    state = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "party": party_label(node_id),
        "node_id": node_id,
        "node_address": config_payload["nodes"][node_id],
        "spu_internal_address": get_party_internal_address(config_payload, node_id),
        "node_processes": [process_info],
    }
    write_json(state_json, state)

    if not args.skip_reachability_check:
        try:
            wait_for_addresses(
                [get_node_address(config_payload, node_id)],
                args.startup_timeout_sec,
                f"{party_label(node_id)} SPU node gRPC port",
            )
        except Exception as error:
            state["reachable"] = False
            state["startup_error"] = str(error)
            write_json(state_json, state)
            stop_party_process(config_path, args.config, node_id, state_json)
            raise
        state["reachable"] = True
        write_json(state_json, state)

    print(f"{party_label(node_id)} SPU node started; state={state_json}; log={process_info['log_path']}")


def stop_party(args):
    node_id = normalize_party(args.party)
    state_json = resolve_party_state_path(args.state_json, node_id)
    stop_party_process(Path(args.config).resolve(), args.config, node_id, state_json)
    print(f"{party_label(node_id)} SPU node stop requested; state={state_json}")


def check_party(args):
    config_payload = load_json(Path(args.config).resolve())
    node_id = normalize_party(args.party)
    validate_party_remote_config(config_payload, node_id, args.allow_localhost)
    wait_for_addresses(
        [get_node_address(config_payload, node_id)],
        args.startup_timeout_sec,
        f"{party_label(node_id)} SPU node gRPC port",
    )
    print(f"{party_label(node_id)} SPU node port is reachable.")


def build_remote_config_payload(args):
    template_path = Path(args.template).resolve()
    validate_remote_address(args.node0_addr, "node0_addr", args.allow_localhost)
    validate_remote_address(args.node1_addr, "node1_addr", args.allow_localhost)
    validate_remote_address(args.spu_internal0_addr, "spu_internal0_addr", args.allow_localhost)
    validate_remote_address(args.spu_internal1_addr, "spu_internal1_addr", args.allow_localhost)
    validate_distinct_addresses(
        [
            ("node0_addr", args.node0_addr),
            ("node1_addr", args.node1_addr),
            ("spu_internal0_addr", args.spu_internal0_addr),
            ("spu_internal1_addr", args.spu_internal1_addr),
        ]
    )

    payload = load_json(template_path)
    payload["id"] = args.config_id
    payload["nodes"]["node:0"] = args.node0_addr.strip()
    payload["nodes"]["node:1"] = args.node1_addr.strip()

    spu_config = payload["devices"]["SPU"]["config"]
    spu_config["spu_internal_addrs"] = [
        args.spu_internal0_addr.strip(),
        args.spu_internal1_addr.strip(),
    ]
    runtime_config = spu_config.setdefault("runtime_config", {})
    runtime_config["experimental_enable_colocated_optimization"] = bool(args.enable_colocated_optimization)
    cheetah_config = runtime_config.setdefault("cheetah_2pc_config", {})
    cheetah_config.pop("approx_less_precision", None)
    if not args.allow_cheetah_mul_lsb_error:
        cheetah_config["enable_mul_lsb_error"] = False
    if args.cheetah_disable_matmul_pack:
        cheetah_config["disable_matmul_pack"] = True
    else:
        cheetah_config.pop("disable_matmul_pack", None)
    if int(args.cheetah_mixed_compare_mode) > 0:
        cheetah_config["mixed_compare_mode"] = int(args.cheetah_mixed_compare_mode)
    else:
        cheetah_config.pop("mixed_compare_mode", None)
    return payload


def write_remote_commands(path: Path, config_ref: str, python_bin: str):
    config_text = shell_quote(config_ref)
    python_text = shell_quote(python_bin)
    payload = (
        "# TransShield 远程 2PC 启动命令\n\n"
        "将同一份远程配置复制到双方机器后，分别在对应机器执行下面命令。\n"
        "`start-party` 会在后台启动本方 SPU 节点，写入状态文件，并把日志记录到 `logs/`。\n\n"
        "## 医院侧 / P1 / node:0\n\n"
        "```bash\n"
        f"{python_text} tools/transshield_spu_runtime_setup.py start-party --config {config_text} --party hospital --restart\n"
        "```\n\n"
        "## AI 或模型算力方 / P2 / node:1\n\n"
        "```bash\n"
        f"{python_text} tools/transshield_spu_runtime_setup.py start-party --config {config_text} --party ai --restart\n"
        "```\n\n"
        "## 单方节点检查\n\n"
        "```bash\n"
        f"{python_text} tools/transshield_spu_runtime_setup.py check-party --config {config_text} --party hospital\n"
        f"{python_text} tools/transshield_spu_runtime_setup.py check-party --config {config_text} --party ai\n"
        "```\n\n"
        "## 协调侧连通性检查\n\n"
        "```bash\n"
        f"{python_text} tools/transshield_spu_runtime_setup.py check --config {config_text}\n"
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def render_remote(args):
    output_path = Path(args.output).resolve()
    payload = build_remote_config_payload(args)
    spu_config = payload["devices"]["SPU"]["config"]
    runtime_config = spu_config["runtime_config"]

    write_json(output_path, payload)
    commands_path = Path(args.commands_out).resolve() if args.commands_out else output_path.with_suffix(".commands.md")
    write_remote_commands(commands_path, args.output, args.python_bin)
    summary = {
        "status": "written",
        "config": str(output_path),
        "commands": str(commands_path),
        "nodes": payload["nodes"],
        "spu_internal_addrs": spu_config["spu_internal_addrs"],
        "experimental_enable_colocated_optimization": runtime_config["experimental_enable_colocated_optimization"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def write_executable(path: Path, content: str):
    write_text_file(path, content)
    path.chmod(path.stat().st_mode | 0o755)


def party_env_template(
    *,
    repo_root: Path,
    config_path: Path,
    log_dir: Path,
    state_json: Path,
    role: str,
):
    return f"""# 复制为 .env 后在 {role} 机器上按现场环境修改。
TRANSSHIELD_REPO_ROOT={shell_quote(repo_root)}
TRANSSHIELD_PYTHON_BIN=python
TRANSSHIELD_RUNTIME_SETUP_SCRIPT=tools/transshield_spu_runtime_setup.py
TRANSSHIELD_REMOTE_CONFIG={shell_quote(config_path)}
TRANSSHIELD_SPU_LOG_DIR={shell_quote(log_dir)}
TRANSSHIELD_SPU_STATE_JSON={shell_quote(state_json)}
TRANSSHIELD_STARTUP_TIMEOUT_SEC=60
TRANSSHIELD_ALLOW_LOCALHOST=0
"""


def party_script(*, role: str, action: str, repo_root: Path, config_path: Path, log_dir: Path, state_json: Path):
    timeout_line = 'TIMEOUT="${TRANSSHIELD_STARTUP_TIMEOUT_SEC:-60}"'
    common = f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi
PYTHON_BIN="${{TRANSSHIELD_PYTHON_BIN:-python}}"
REPO_ROOT="${{TRANSSHIELD_REPO_ROOT:-{repo_root}}}"
SETUP_SCRIPT="${{TRANSSHIELD_RUNTIME_SETUP_SCRIPT:-tools/transshield_spu_runtime_setup.py}}"
CONFIG="${{TRANSSHIELD_REMOTE_CONFIG:-{config_path}}}"
LOG_DIR="${{TRANSSHIELD_SPU_LOG_DIR:-{log_dir}}}"
STATE_JSON="${{TRANSSHIELD_SPU_STATE_JSON:-{state_json}}}"
{timeout_line}
ALLOW_LOCALHOST="${{TRANSSHIELD_ALLOW_LOCALHOST:-0}}"
ALLOW_LOCALHOST_ARG=()
if [ "$ALLOW_LOCALHOST" = "1" ]; then
  ALLOW_LOCALHOST_ARG=(--allow-localhost)
fi
cd "$REPO_ROOT"
"""
    if action == "start":
        command = (
            'exec "$PYTHON_BIN" "$SETUP_SCRIPT" start-party '
            f'--config "$CONFIG" --party {role} --restart '
            '--log-dir "$LOG_DIR" --state-json "$STATE_JSON" '
            '--startup-timeout-sec "$TIMEOUT" "${ALLOW_LOCALHOST_ARG[@]}"\n'
        )
    elif action == "check":
        command = (
            'exec "$PYTHON_BIN" "$SETUP_SCRIPT" check-party '
            f'--config "$CONFIG" --party {role} --startup-timeout-sec "$TIMEOUT" '
            '"${ALLOW_LOCALHOST_ARG[@]}"\n'
        )
    else:
        command = (
            'exec "$PYTHON_BIN" "$SETUP_SCRIPT" stop-party '
            f'--config "$CONFIG" --party {role} --state-json "$STATE_JSON"\n'
        )
    return common + command


def coordinator_env_template(repo_root: Path, config_path: Path, output_dir: Path):
    bundle_dir = repo_root / "artifacts" / "frozen_bundle_medical_dynamic_mainline"
    return f"""# 复制为 .env 后在协调机器上修改 manifest 路径和运行参数。
TRANSSHIELD_REPO_ROOT={shell_quote(repo_root)}
TRANSSHIELD_PYTHON_BIN=python
TRANSSHIELD_RUNTIME_SETUP_SCRIPT=tools/transshield_spu_runtime_setup.py
TRANSSHIELD_REMOTE_CONFIG={shell_quote(config_path)}
TRANSSHIELD_BUNDLE_DIR={shell_quote(bundle_dir)}
TRANSSHIELD_PUBLIC_MANIFEST=/path/to/public_manifest.json
TRANSSHIELD_P1_MANIFEST=/path/to/p1_share_manifest.json
TRANSSHIELD_P2_MANIFEST=/path/to/p2_share_manifest.json
TRANSSHIELD_OUTPUT_DIR={shell_quote(output_dir / "runs")}
TRANSSHIELD_MAX_SAMPLES=1
TRANSSHIELD_STARTUP_TIMEOUT_SEC=60
TRANSSHIELD_WARMUP_TIMEOUT_SEC=60
"""


def frontend_env_template():
    return """# 当前 showcase 前端仍调用 /api/medical/* 展示站接口。
# 如果前端和 showcase_api.app 或兼容反向代理分开部署，复制到 showcase/.env.local。
# split_gateway 的 /api/hospital/*、/api/split/* 接口不是当前展示站前端的直接替代入口。
VITE_TRANSSHIELD_API_BASE_URL=https://showcase-api-or-reverse-proxy.example.org
"""


def gateway_env_template(*, role: str, repo_root: Path, storage_dir: Path, config_path: Path, port: int):
    return f"""# 复制为 .gateway.env 后按现场环境修改。
TRANSSHIELD_REPO_ROOT={shell_quote(repo_root)}
TRANSSHIELD_SPLIT_ROLE={role}
TRANSSHIELD_SPLIT_STORAGE_DIR={shell_quote(storage_dir)}
TRANSSHIELD_SPLIT_PYTHON_BIN=python
TRANSSHIELD_SPLIT_HOST=127.0.0.1
TRANSSHIELD_SPLIT_PORT={port}
TRANSSHIELD_SPLIT_AUTH_TOKEN=
TRANSSHIELD_SPLIT_RUNTIME_MODE=mock
TRANSSHIELD_SPLIT_AI_GATEWAY_URL=
TRANSSHIELD_SPLIT_AI_GATEWAY_AUTH_TOKEN=
TRANSSHIELD_SPLIT_GATEWAY_FORWARD_TIMEOUT_SEC=5
TRANSSHIELD_SPLIT_MAX_SHARE_BYTES=20971520
TRANSSHIELD_SPLIT_MAX_IMAGE_BYTES=10485760
TRANSSHIELD_SPLIT_MAX_IMAGE_DIMENSION=8192
TRANSSHIELD_SPLIT_INPUT_SIZE=224
TRANSSHIELD_SPLIT_NORM_MEAN=0.485,0.456,0.406
TRANSSHIELD_SPLIT_NORM_STD=0.229,0.224,0.225
TRANSSHIELD_SPLIT_NORM_CLIP_ABS=2.0
TRANSSHIELD_SPLIT_BUNDLE_DIR={shell_quote(repo_root / "artifacts" / "frozen_bundle_medical_dynamic_mainline")}
TRANSSHIELD_SPLIT_SPU_CONFIG={shell_quote(config_path)}
"""


def gateway_script(*, role: str, repo_root: Path, port: int):
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -f "$SCRIPT_DIR/.gateway.env" ]; then
  set -a
  . "$SCRIPT_DIR/.gateway.env"
  set +a
fi
REPO_ROOT="${{TRANSSHIELD_REPO_ROOT:-{repo_root}}}"
PYTHON_BIN="${{TRANSSHIELD_SPLIT_PYTHON_BIN:-python}}"
HOST="${{TRANSSHIELD_SPLIT_HOST:-127.0.0.1}}"
PORT="${{TRANSSHIELD_SPLIT_PORT:-{port}}}"
export TRANSSHIELD_SPLIT_ROLE="${{TRANSSHIELD_SPLIT_ROLE:-{role}}}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m uvicorn showcase_api.split_gateway:app --host "$HOST" --port "$PORT"
"""


def systemd_quote(value: Path):
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def coordinator_check_script(repo_root: Path, config_path: Path):
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi
PYTHON_BIN="${{TRANSSHIELD_PYTHON_BIN:-python}}"
REPO_ROOT="${{TRANSSHIELD_REPO_ROOT:-{repo_root}}}"
SETUP_SCRIPT="${{TRANSSHIELD_RUNTIME_SETUP_SCRIPT:-tools/transshield_spu_runtime_setup.py}}"
CONFIG="${{TRANSSHIELD_REMOTE_CONFIG:-{config_path}}}"
TIMEOUT="${{TRANSSHIELD_STARTUP_TIMEOUT_SEC:-60}}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SETUP_SCRIPT" check --config "$CONFIG" --startup-timeout-sec "$TIMEOUT"
"""


def coordinator_warmup_script(repo_root: Path, config_path: Path):
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi
PYTHON_BIN="${{TRANSSHIELD_PYTHON_BIN:-python}}"
REPO_ROOT="${{TRANSSHIELD_REPO_ROOT:-{repo_root}}}"
SETUP_SCRIPT="${{TRANSSHIELD_RUNTIME_SETUP_SCRIPT:-tools/transshield_spu_runtime_setup.py}}"
CONFIG="${{TRANSSHIELD_REMOTE_CONFIG:-{config_path}}}"
STARTUP_TIMEOUT="${{TRANSSHIELD_STARTUP_TIMEOUT_SEC:-60}}"
WARMUP_TIMEOUT="${{TRANSSHIELD_WARMUP_TIMEOUT_SEC:-60}}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SETUP_SCRIPT" warmup \\
  --config "$CONFIG" \\
  --startup-timeout-sec "$STARTUP_TIMEOUT" \\
  --warmup-timeout-sec "$WARMUP_TIMEOUT"
"""


def coordinator_run_script(repo_root: Path, config_path: Path, output_dir: Path):
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi
PYTHON_BIN="${{TRANSSHIELD_PYTHON_BIN:-python}}"
REPO_ROOT="${{TRANSSHIELD_REPO_ROOT:-{repo_root}}}"
CONFIG="${{TRANSSHIELD_REMOTE_CONFIG:-{config_path}}}"
BUNDLE_DIR="${{TRANSSHIELD_BUNDLE_DIR:-$REPO_ROOT/artifacts/frozen_bundle_medical_dynamic_mainline}}"
PUBLIC_MANIFEST="${{TRANSSHIELD_PUBLIC_MANIFEST:-/path/to/public_manifest.json}}"
P1_MANIFEST="${{TRANSSHIELD_P1_MANIFEST:-/path/to/p1_share_manifest.json}}"
P2_MANIFEST="${{TRANSSHIELD_P2_MANIFEST:-/path/to/p2_share_manifest.json}}"
OUTPUT_DIR="${{TRANSSHIELD_OUTPUT_DIR:-{output_dir / "runs"}}}"
MAX_SAMPLES="${{TRANSSHIELD_MAX_SAMPLES:-1}}"
mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" integrations/transshield_runtime/e2e_secure_vit/transshield_e2e_secure_vit.py run \\
  --runtime spu \\
  --bundle-dir "$BUNDLE_DIR" \\
  --input-share-public-manifest-json "$PUBLIC_MANIFEST" \\
  --input-p1-share-manifest-json "$P1_MANIFEST" \\
  --input-p2-share-manifest-json "$P2_MANIFEST" \\
  --party-local-share-load \\
  --redact-private-input-paths \\
  --config "$CONFIG" \\
  --output-json "$OUTPUT_DIR/candidate.json" \\
  --output-pt "$OUTPUT_DIR/candidate.pt" \\
  --max-samples "$MAX_SAMPLES"
"""


def systemd_unit_template(*, description: str, start_script: Path, stop_script: Path):
    return f"""[Unit]
Description={description}
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash {systemd_quote(start_script)}
ExecStop=/bin/bash {systemd_quote(stop_script)}

[Install]
WantedBy=multi-user.target
"""


def deployment_readme(args, output_dir: Path, config_path: Path):
    return f"""# TransShield 远程 2PC 部署包

本部署包把“医院侧 P1 + 模型算力方 P2”中可以复用的启动脚本和配置先生成出来。
它仍是部署脚手架，不等于生产安全验收通过。

## 生成目录

- `configs/2pc.remote.json`：双方共用的 SPU 节点配置。
- `hospital/`：P1 启动、检查、停止脚本和 `.env.example`。
- `ai_provider/`：P2 启动、检查、停止脚本和 `.env.example`。
- `coordinator/`：检查全部节点和运行 secure ViT 的示例脚本。
- 各角色目录中的 `start_split_gateway.sh`：启动医院/AI/协调三方网关框架。
- `frontend/`：将当前展示站前端指向兼容 `/api/medical/*` API base 的 Vite 环境变量示例。
- `systemd/`：可选的 systemd service 示例。

## 现场需要修改的值

- 医院侧和模型算力方的真实 IP、端口、域名和防火墙规则。
- 每台机器上的 Python/SPU 环境路径。
- TLS、VPN、认证和密钥分发方式。
- 双方各自的 share manifest 路径和数据落盘目录。
- P2 是外部 AI 公司、内部模型团队，还是其他可信算力方。

## 本部署包当前地址

- 医院/P1 节点：`{args.node0_addr}`
- 模型算力方/P2 节点：`{args.node1_addr}`
- 医院/P1 SPU internal：`{args.spu_internal0_addr}`
- 模型算力方/P2 SPU internal：`{args.spu_internal1_addr}`
- 配置路径：`{config_path}`

## 最小运行步骤

在医院/P1 机器上：

```bash
cd {output_dir}/hospital
cp .env.example .env
# 如果部署包被复制到另一台机器或目录，先编辑 .env 中的路径和 Python 环境。
./start_spu_party.sh
./check_spu_party.sh
cp .gateway.env.example .gateway.env
# 另开终端或交给 systemd/进程守护执行。
./start_split_gateway.sh
```

在模型算力方/P2 机器上：

```bash
cd {output_dir}/ai_provider
cp .env.example .env
# 如果部署包被复制到另一台机器或目录，先编辑 .env 中的路径和 Python 环境。
./start_spu_party.sh
./check_spu_party.sh
cp .gateway.env.example .gateway.env
# 另开终端或交给 systemd/进程守护执行。
./start_split_gateway.sh
```

在一台能访问双方端口的协调机器上：

```bash
cd {output_dir}/coordinator
cp .env.example .env
# 编辑 .env 中的 repo、config、manifest 和 Python 环境路径。
./check_all_parties.sh
./warmup_all_parties.sh
cp .gateway.env.example .gateway.env
# 另开终端或交给 systemd/进程守护执行。
./start_split_gateway.sh
```

如果 Vite 前端和 `showcase_api.app` 或兼容反向代理分开部署，复制
`frontend/.env.example` 到 `showcase/.env.local`，设置
`VITE_TRANSSHIELD_API_BASE_URL` 后重新构建前端。注意：当前展示站前端仍调用
`/api/medical/*`；`split_gateway` 的 `/api/hospital/*`、`/api/split/*`
不是它的直接替代入口，若要做医院/AI 双网关 UI，需要另接对应 API。

真实 public/P1/P2 manifests 准备好之后，修改 `coordinator/.env` 并运行：

```bash
./run_secure_vit_example.sh
```

## 仍未生产化的部分

本部署包不会自动创建生产级医院/AI 双网关、签发证书、轮换密钥、接入合规审计系统，
也不会自动让模型参数变成 AI 侧私有。以上部分必须结合目标医院和算力方环境继续集成。
`start_split_gateway.sh` 提供的是最小可运行网关框架：医院侧只接收 P1 share，
也可通过 `/api/hospital/tasks/{{task_id}}/image` 接收 PNG/JPEG 并生成 P1/P2 share；
AI 侧只接收 P2 share 和模型 manifest，协调侧负责异步任务状态、取消和 runner 命令。
真实生产环境仍应把它放在 TLS/VPN、认证、审计和运维守护之后。

医院侧 `.gateway.env` 可设置 `TRANSSHIELD_SPLIT_AI_GATEWAY_URL`，让图片入口生成 P2 share 后自动转发到 AI 网关。
这只是联调便利功能，不包含生产级消息队列、重试和机构审计；转发失败时响应里仍会返回 `p2_share_delivery`，可由上层系统重试。

在没有真实医院/AI 系统时，可先在仓库根目录运行：

```bash
python tools/split_gateway_smoke.py --keep-state
```

该命令会本地模拟医院图片输入、AI 接收 P2 share、AI model manifest 和协调方 mock run。
"""


def render_deployment(args):
    output_dir = Path(args.output_dir).resolve()
    config_path = output_dir / "configs" / "2pc.remote.json"
    payload = build_remote_config_payload(args)
    write_json(config_path, payload)

    logs_dir = output_dir / "logs"
    hospital_dir = output_dir / "hospital"
    ai_dir = output_dir / "ai_provider"
    coordinator_dir = output_dir / "coordinator"
    frontend_dir = output_dir / "frontend"
    systemd_dir = output_dir / "systemd"
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "tools" / "transshield_spu_runtime_setup.py").exists():
        raise ValueError(f"repo root does not look like TransShield: {repo_root}")

    hospital_state = logs_dir / "hospital_spu_state.json"
    ai_state = logs_dir / "ai_provider_spu_state.json"
    hospital_log_dir = logs_dir / "hospital_spu_nodes"
    ai_log_dir = logs_dir / "ai_provider_spu_nodes"
    hospital_gateway_store = output_dir / "gateway_state" / "hospital"
    ai_gateway_store = output_dir / "gateway_state" / "ai_provider"
    coordinator_gateway_store = output_dir / "gateway_state" / "coordinator"

    write_text_file(
        hospital_dir / ".env.example",
        party_env_template(
            repo_root=repo_root,
            config_path=config_path,
            log_dir=hospital_log_dir,
            state_json=hospital_state,
            role="hospital/P1",
        ),
    )
    write_text_file(
        ai_dir / ".env.example",
        party_env_template(
            repo_root=repo_root,
            config_path=config_path,
            log_dir=ai_log_dir,
            state_json=ai_state,
            role="model-provider/P2",
        ),
    )
    for role, role_dir, log_dir, state_json in (
        ("hospital", hospital_dir, hospital_log_dir, hospital_state),
        ("ai", ai_dir, ai_log_dir, ai_state),
    ):
        write_executable(
            role_dir / "start_spu_party.sh",
            party_script(
                role=role,
                action="start",
                repo_root=repo_root,
                config_path=config_path,
                log_dir=log_dir,
                state_json=state_json,
            ),
        )
        write_executable(
            role_dir / "check_spu_party.sh",
            party_script(
                role=role,
                action="check",
                repo_root=repo_root,
                config_path=config_path,
                log_dir=log_dir,
                state_json=state_json,
            ),
        )
        write_executable(
            role_dir / "stop_spu_party.sh",
            party_script(
                role=role,
                action="stop",
                repo_root=repo_root,
                config_path=config_path,
                log_dir=log_dir,
                state_json=state_json,
            ),
        )

    write_text_file(
        hospital_dir / ".gateway.env.example",
        gateway_env_template(
            role="hospital",
            repo_root=repo_root,
            storage_dir=hospital_gateway_store,
            config_path=config_path,
            port=8701,
        ),
    )
    write_executable(hospital_dir / "start_split_gateway.sh", gateway_script(role="hospital", repo_root=repo_root, port=8701))
    write_text_file(
        ai_dir / ".gateway.env.example",
        gateway_env_template(
            role="ai",
            repo_root=repo_root,
            storage_dir=ai_gateway_store,
            config_path=config_path,
            port=8702,
        ),
    )
    write_executable(ai_dir / "start_split_gateway.sh", gateway_script(role="ai", repo_root=repo_root, port=8702))
    write_text_file(
        coordinator_dir / ".gateway.env.example",
        gateway_env_template(
            role="coordinator",
            repo_root=repo_root,
            storage_dir=coordinator_gateway_store,
            config_path=config_path,
            port=8703,
        ),
    )
    write_executable(
        coordinator_dir / "start_split_gateway.sh",
        gateway_script(role="coordinator", repo_root=repo_root, port=8703),
    )
    write_text_file(coordinator_dir / ".env.example", coordinator_env_template(repo_root, config_path, output_dir))
    write_executable(coordinator_dir / "check_all_parties.sh", coordinator_check_script(repo_root, config_path))
    write_executable(coordinator_dir / "warmup_all_parties.sh", coordinator_warmup_script(repo_root, config_path))
    write_executable(coordinator_dir / "run_secure_vit_example.sh", coordinator_run_script(repo_root, config_path, output_dir))
    write_text_file(frontend_dir / ".env.example", frontend_env_template())
    write_remote_commands(output_dir / "COMMANDS.md", str(config_path), args.python_bin)
    write_text_file(output_dir / "README.md", deployment_readme(args, output_dir, config_path))
    write_text_file(
        systemd_dir / "transshield-hospital-spu.service.example",
        systemd_unit_template(
            description="TransShield hospital/P1 SPU party node",
            start_script=hospital_dir / "start_spu_party.sh",
            stop_script=hospital_dir / "stop_spu_party.sh",
        ),
    )
    write_text_file(
        systemd_dir / "transshield-ai-provider-spu.service.example",
        systemd_unit_template(
            description="TransShield model-provider/P2 SPU party node",
            start_script=ai_dir / "start_spu_party.sh",
            stop_script=ai_dir / "stop_spu_party.sh",
        ),
    )

    summary = {
        "status": "written",
        "deployment_dir": str(output_dir),
        "config": str(config_path),
        "hospital_start": str(hospital_dir / "start_spu_party.sh"),
        "ai_provider_start": str(ai_dir / "start_spu_party.sh"),
        "hospital_gateway": str(hospital_dir / "start_split_gateway.sh"),
        "ai_provider_gateway": str(ai_dir / "start_split_gateway.sh"),
        "coordinator_gateway": str(coordinator_dir / "start_split_gateway.sh"),
        "coordinator_check": str(coordinator_dir / "check_all_parties.sh"),
        "coordinator_warmup": str(coordinator_dir / "warmup_all_parties.sh"),
        "readme": str(output_dir / "README.md"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


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
    parser = argparse.ArgumentParser(description="Configure Transshield 2PC/SPU runtime nodes.")
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

    parser_warmup = subparsers.add_parser("warmup", help="run a minimal SPU runtime warmup against configured nodes")
    parser_warmup.add_argument("--config", default="configs/openbumblebee/2pc.json")
    parser_warmup.add_argument("--startup-timeout-sec", type=float, default=20.0)
    parser_warmup.add_argument("--warmup-timeout-sec", type=float, default=60.0)

    parser_start_party = subparsers.add_parser(
        "start-party",
        help="start one remote 2PC party node in the background",
    )
    parser_start_party.add_argument("--config", default="configs/transshield_runtime/2pc.remote.json")
    parser_start_party.add_argument("--party", required=True, help="hospital/P1/node:0 or ai/P2/node:1")
    parser_start_party.add_argument("--restart", action="store_true")
    parser_start_party.add_argument("--log-dir", default="logs/spu_party_nodes")
    parser_start_party.add_argument("--state-json", default="")
    parser_start_party.add_argument("--stop-wait-sec", type=float, default=1.0)
    parser_start_party.add_argument("--startup-timeout-sec", type=float, default=30.0)
    parser_start_party.add_argument("--skip-reachability-check", action="store_true")
    parser_start_party.add_argument("--allow-localhost", action="store_true", help="allow loopback addresses for dry-run tests")

    parser_stop_party = subparsers.add_parser("stop-party", help="stop one remote 2PC party node")
    parser_stop_party.add_argument("--config", default="configs/transshield_runtime/2pc.remote.json")
    parser_stop_party.add_argument("--party", required=True, help="hospital/P1/node:0 or ai/P2/node:1")
    parser_stop_party.add_argument("--state-json", default="")

    parser_check_party = subparsers.add_parser("check-party", help="wait until one configured party node is reachable")
    parser_check_party.add_argument("--config", default="configs/transshield_runtime/2pc.remote.json")
    parser_check_party.add_argument("--party", required=True, help="hospital/P1/node:0 or ai/P2/node:1")
    parser_check_party.add_argument("--startup-timeout-sec", type=float, default=20.0)
    parser_check_party.add_argument("--allow-localhost", action="store_true", help="allow loopback addresses for dry-run tests")

    parser_remote = subparsers.add_parser(
        "render-remote",
        help="render a cross-machine 2PC config and party launch commands",
    )
    parser_remote.add_argument("--template", default="configs/transshield_runtime/2pc.template.json")
    parser_remote.add_argument("--output", default="configs/transshield_runtime/2pc.remote.json")
    parser_remote.add_argument("--commands-out", default="")
    parser_remote.add_argument("--config-id", default="remote.2pc")
    parser_remote.add_argument("--node0-addr", required=True, help="hospital/P1 node address, host:port")
    parser_remote.add_argument("--node1-addr", required=True, help="AI-provider/P2 node address, host:port")
    parser_remote.add_argument("--spu-internal0-addr", required=True, help="hospital/P1 SPU internal address, host:port")
    parser_remote.add_argument("--spu-internal1-addr", required=True, help="AI-provider/P2 SPU internal address, host:port")
    parser_remote.add_argument("--python-bin", default="python")
    parser_remote.add_argument("--allow-localhost", action="store_true", help="allow loopback addresses for dry-run tests")
    parser_remote.add_argument(
        "--enable-colocated-optimization",
        action="store_true",
        help="keep SPU colocated optimization enabled; remote 2PC defaults to disabled",
    )
    parser_remote.add_argument(
        "--allow-cheetah-mul-lsb-error",
        action="store_true",
        help="keep Cheetah enable_mul_lsb_error enabled instead of forcing the stable default false",
    )
    parser_remote.add_argument(
        "--cheetah-disable-matmul-pack",
        action="store_true",
        help="force Cheetah matmul to bypass ciphertext packing for current runtime validation",
    )
    parser_remote.add_argument(
        "--cheetah-mixed-compare-mode",
        type=int,
        default=0,
        help="set Cheetah mixed_compare_mode for current runtime validation; 0 keeps native path",
    )

    parser_deployment = subparsers.add_parser(
        "render-deployment",
        help="render a reusable hospital/P1 and model-provider/P2 deployment bundle",
    )
    parser_deployment.add_argument("--template", default="configs/transshield_runtime/2pc.template.json")
    parser_deployment.add_argument("--output-dir", default="deploy/transshield_remote_2pc")
    parser_deployment.add_argument("--repo-root", default=".", help="TransShield repo root used by generated scripts")
    parser_deployment.add_argument("--config-id", default="remote.2pc")
    parser_deployment.add_argument("--node0-addr", required=True, help="hospital/P1 node address, host:port")
    parser_deployment.add_argument("--node1-addr", required=True, help="model-provider/P2 node address, host:port")
    parser_deployment.add_argument("--spu-internal0-addr", required=True, help="hospital/P1 SPU internal address, host:port")
    parser_deployment.add_argument("--spu-internal1-addr", required=True, help="model-provider/P2 SPU internal address, host:port")
    parser_deployment.add_argument("--python-bin", default="python")
    parser_deployment.add_argument("--allow-localhost", action="store_true", help="allow loopback addresses for dry-run tests")
    parser_deployment.add_argument(
        "--enable-colocated-optimization",
        action="store_true",
        help="keep SPU colocated optimization enabled; remote 2PC defaults to disabled",
    )
    parser_deployment.add_argument(
        "--allow-cheetah-mul-lsb-error",
        action="store_true",
        help="keep Cheetah enable_mul_lsb_error enabled instead of forcing the stable default false",
    )
    parser_deployment.add_argument(
        "--cheetah-disable-matmul-pack",
        action="store_true",
        help="force Cheetah matmul to bypass ciphertext packing for current runtime validation",
    )
    parser_deployment.add_argument(
        "--cheetah-mixed-compare-mode",
        type=int,
        default=0,
        help="set Cheetah mixed_compare_mode for current runtime validation; 0 keeps native path",
    )

    return parser


def main():
    args = build_parser().parse_args()
    try:
        if args.command == "configure":
            configure(args)
        elif args.command == "start":
            start(args)
        elif args.command == "stop":
            stop(args)
        elif args.command == "check":
            check(args)
        elif args.command == "warmup":
            warmup(args)
        elif args.command == "start-party":
            start_party(args)
        elif args.command == "stop-party":
            stop_party(args)
        elif args.command == "check-party":
            check_party(args)
        elif args.command == "render-deployment":
            render_deployment(args)
        else:
            render_remote(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
