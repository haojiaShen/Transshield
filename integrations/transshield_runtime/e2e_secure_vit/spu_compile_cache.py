"""Content-addressed SPU JAX compilation cache.

SPU 0.9.3's distributed JAX frontend recompiles a function on every device
call.  The showcase starts a fresh runner process for every request, so an
in-memory JAX cache cannot help.  This module installs a narrow wrapper around
``spu.utils.frontend.compile`` and persists only the compiled executable and
the public output shape tree.  Runtime values and secret shares are never
written to the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from functools import wraps
from importlib import metadata as importlib_metadata
from pathlib import Path


_ORIGINAL_COMPILE = None
_CACHE_DIR: Path | None = None
_CACHE_NAMESPACE = ""
_STATS = {
    "enabled": False,
    "hits": 0,
    "misses": 0,
    "compile_sec": 0.0,
    "cache_read_sec": 0.0,
    "cache_write_sec": 0.0,
    "errors": 0,
}


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _abstract_tree(value):
    import jax
    import numpy as np

    leaves, tree = jax.tree_util.tree_flatten(value)
    abstract_leaves = []
    for leaf in leaves:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            abstract_leaves.append(
                {
                    "kind": "array",
                    "shape": [int(dim) for dim in leaf.shape],
                    "dtype": str(np.dtype(leaf.dtype)),
                }
            )
        elif leaf is None or isinstance(leaf, (str, int, float, bool)):
            abstract_leaves.append(
                {
                    "kind": "scalar",
                    "type": type(leaf).__qualname__,
                    "value": leaf,
                }
            )
        else:
            raise TypeError(f"unsupported compile-cache leaf: {type(leaf)!r}")
    return {"tree": str(tree), "leaves": abstract_leaves}


def _cache_key(
    kind,
    fn,
    m_args,
    m_kwargs,
    input_vis,
    static_argnums,
    static_argnames,
    copts,
) -> str:
    import cloudpickle
    import jax

    function_digest = hashlib.sha256(cloudpickle.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    payload = {
        "cache_format": 1,
        "namespace": _CACHE_NAMESPACE,
        "python": list(sys.version_info[:3]),
        "jax": str(jax.__version__),
        "spu": _package_version("spu"),
        "kind": str(kind),
        "function_sha256": function_digest,
        "arguments": _abstract_tree((m_args, m_kwargs)),
        "input_visibility": [int(value) for value in input_vis],
        "static_argnums": list(static_argnums or ()),
        "static_argnames": static_argnames,
        "compiler_options_sha256": hashlib.sha256(copts.SerializeToString()).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_entry(path: Path):
    from spu import spu_pb2

    payload = pickle.loads(path.read_bytes())
    if payload.get("cache_format") != 1:
        raise ValueError("unsupported SPU compile-cache format")
    executable = spu_pb2.ExecutableProto()
    executable.ParseFromString(payload["executable"])
    return executable, payload["output"]


def _write_entry(path: Path, executable, output):
    payload = {
        "cache_format": 1,
        "executable": executable.SerializeToString(),
        "output": output,
    }
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def install_spu_compile_cache(cache_dir: str | Path | None, *, namespace: str = "") -> bool:
    """Install the cache wrapper once and return whether caching is enabled."""
    global _ORIGINAL_COMPILE, _CACHE_DIR, _CACHE_NAMESPACE

    if not cache_dir:
        return False
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR = cache_path
    _CACHE_NAMESPACE = str(namespace)
    _STATS["enabled"] = True

    if _ORIGINAL_COMPILE is not None:
        return True

    import spu.utils.frontend as spu_frontend

    _ORIGINAL_COMPILE = spu_frontend.compile

    @wraps(_ORIGINAL_COMPILE)
    def cached_compile(
        kind,
        fn,
        m_args,
        m_kwargs,
        input_names,
        input_vis,
        outputNameGen,
        static_argnums=(),
        static_argnames=None,
        copts=None,
    ):
        if copts is None:
            from spu import spu_pb2

            copts = spu_pb2.CompilerOptions()
        try:
            key = _cache_key(
                kind,
                fn,
                m_args,
                m_kwargs,
                input_vis,
                static_argnums,
                static_argnames,
                copts,
            )
            entry_path = _CACHE_DIR / f"{key}.spucache"
        except Exception:
            _STATS["errors"] += 1
            return _ORIGINAL_COMPILE(
                kind,
                fn,
                m_args,
                m_kwargs,
                input_names,
                input_vis,
                outputNameGen,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                copts=copts,
            )

        if entry_path.is_file():
            started = time.perf_counter()
            try:
                result = _load_entry(entry_path)
                _STATS["hits"] += 1
                _STATS["cache_read_sec"] += time.perf_counter() - started
                return result
            except Exception:
                _STATS["errors"] += 1

        _STATS["misses"] += 1
        started = time.perf_counter()
        result = _ORIGINAL_COMPILE(
            kind,
            fn,
            m_args,
            m_kwargs,
            input_names,
            input_vis,
            outputNameGen,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            copts=copts,
        )
        _STATS["compile_sec"] += time.perf_counter() - started
        started = time.perf_counter()
        try:
            _write_entry(entry_path, *result)
            _STATS["cache_write_sec"] += time.perf_counter() - started
        except Exception:
            _STATS["errors"] += 1
        return result

    spu_frontend.compile = cached_compile
    return True


def reset_spu_compile_cache_stats():
    for key in ("hits", "misses", "compile_sec", "cache_read_sec", "cache_write_sec", "errors"):
        _STATS[key] = 0 if key in {"hits", "misses", "errors"} else 0.0


def get_spu_compile_cache_stats() -> dict:
    payload = dict(_STATS)
    payload["cache_dir"] = None if _CACHE_DIR is None else str(_CACHE_DIR)
    payload["namespace"] = _CACHE_NAMESPACE
    return payload
