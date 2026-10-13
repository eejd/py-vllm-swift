# SPDX-License-Identifier: Apache-2.0
"""SwiftMetalWorker against the real vLLM engine-core worker contract (>= 0.31).

conftest.py stubs ``vllm`` for the unit tests, so these checks run in a fresh
interpreter that imports the installed vLLM. The engine core resolves worker
RPCs by name through ``vllm.v1.serial_utils.run_method``: a missing
``get_supported_kv_cache_layouts`` / ``set_kv_cache_layout`` aborts engine
start-up with ``NotImplementedError: Method ... is not implemented``.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_with_real_vllm(script: str, **env: str) -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import vllm.v1.attention.backends.utils as u; "
         "u.resolve_kv_cache_layout"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("installed vLLM predates KV cache layout negotiation (or is absent)")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": REPO, **env},
    )
    assert result.returncode == 0, result.stdout + result.stderr


_PRELUDE = """
from types import SimpleNamespace
from vllm.config import CacheConfig
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.serial_utils import run_method
from vllm_swift.worker import SwiftMetalWorker

w = object.__new__(SwiftMetalWorker)
w.cache_config = CacheConfig()
w.vllm_config = SimpleNamespace(cache_config=w.cache_config, kv_transfer_config=None)
rpc = lambda name, *args: run_method(w, name, args, {})
"""


def test_engine_core_layout_negotiation_resolves_and_publishes():
    _run_with_real_vllm(
        _PRELUDE
        + """
supported = rpc("get_supported_kv_cache_layouts")
assert supported  # an empty answer is a hard error in resolve_kv_cache_layout
layout = resolve_kv_cache_layout(w.vllm_config, [supported])
rpc("set_kv_cache_layout", layout.name)
assert w.cache_config.kv_cache_layout == layout.name
assert layout.name in supported
"""
    )


def test_every_layout_is_acceptable_to_the_user_override():
    _run_with_real_vllm(
        _PRELUDE
        + """
from vllm.v1.attention.backends.utils import KVCacheLayout
supported = rpc("get_supported_kv_cache_layouts")
assert set(supported) == {layout.name for layout in KVCacheLayout}, supported
assert resolve_kv_cache_layout(w.vllm_config, [supported]).name == "BHLNC"
""",
        VLLM_KV_CACHE_LAYOUT="BHLNC",
    )


def test_initialize_from_config_adopts_the_resolved_layout():
    _run_with_real_vllm(
        _PRELUDE
        + """
rpc("initialize_from_config", SimpleNamespace(num_blocks=7, kv_cache_layout="BLNHC"))
assert w.cache_config.kv_cache_layout == "BLNHC"
"""
    )


def test_profile_takes_the_prefix_the_executor_sends():
    _run_with_real_vllm(
        _PRELUDE
        + """
assert rpc("profile", True, "prefix") is None
"""
    )
