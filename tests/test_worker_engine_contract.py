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


def test_synchronize_device_is_the_workers_own_not_a_torch_accelerator_wait():
    _run_with_real_vllm(
        """
import torch
from vllm.v1.worker.worker_base import WorkerBase
from vllm_swift.worker import SwiftMetalWorker

assert SwiftMetalWorker.synchronize_device is not WorkerBase.synchronize_device

def torch_wait():
    raise AssertionError("waited on torch.accelerator")

torch.accelerator.synchronize = torch_wait
w = object.__new__(SwiftMetalWorker)
from vllm.v1.serial_utils import run_method
assert run_method(w, "synchronize_device", (), {}) is None
"""
    )


def test_constructor_runs_workerbase_init_and_keeps_the_ranks():
    """SwiftMetalWorker(...) as ``WorkerWrapperBase.init_worker`` builds it."""
    _run_with_real_vllm(
        """
import json, os, tempfile
from vllm.config import ModelConfig, VllmConfig
from vllm_swift.worker import SwiftMetalWorker

with tempfile.TemporaryDirectory() as model_dir:
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
            "max_position_embeddings": 256, "torch_dtype": "float16",
        }, f)
    model_config = ModelConfig(
        model=model_dir, tokenizer=model_dir, skip_tokenizer_init=True
    )
    vllm_config = VllmConfig(model_config=model_config)
    w = SwiftMetalWorker(
        vllm_config, local_rank=1, rank=3,
        distributed_init_method="file:///unused", is_driver_worker=False,
    )

assert (w.rank, w.local_rank, w.is_driver_worker) == (3, 1, False)
assert w.model_config is model_config
assert w.cache_config is vllm_config.cache_config
# WorkerBase.__init__ records the rank on the shared parallel config.
assert w.parallel_config.rank == 3
assert w.engine is None
"""
    )
