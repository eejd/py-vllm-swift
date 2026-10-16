# UPSTREAM.md - what `local` carries over upstream

Primary upstream: [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) (this repo's GitHub fork parent).
Branch rules: ADR-0008 / `docs/development-model.md` ("Forks, upstreams and port tiers") in
[eejd/macports-ports-local](https://github.com/eejd/macports-ports-local).

## Why `local` is based on `v0.6.3`, not on `main`

`local` = the last upstream release tag (`v0.6.3`) + the `topic/*` branches below. Upstream `main`
(`v0.6.3-16-gd1dbba0` when this was written) is unreleased and cannot be built outside the author's
machine: `swift/Package.swift` points mlx-swift-lm at `/Users/tom/dev/mlx-swift-lm`, and
`Bridge.swift` uses `RetrievalAttentionKVCache` / `BatchedRetrievalAttentionKVCache`, which exist
in `TheTom/mlx-swift-lm` only on `feat/batched-sparse-attention`, not on `vllm-swift-stable`
(`fd16163`) or `alpha`. Rebase `local` onto the next upstream release tag that builds from
published sources, then cut `v<base>-local<NN>` again (archive the old tip first).

## Behaviour rows (patches carried; one `topic/*` branch each, on top of `main`)

| Topic branch | Change | Existing tests covering it | Upstream PR |
|---|---|---|---|
| `topic/fail-closed-bridge` | `register()` returns `None` when the Swift bridge library cannot be located (vLLM permits one active platform plugin; "assume available" collided with every other backend) | `tests/test_platform.py` | not submitted |
| `topic/manual-seed-all` | `SwiftMetalPlatform.manual_seed_all` (called in vLLM >= 0.27 worker init) | `tests/test_platform.py` | not submitted |
| `topic/compilation-times` | `compile_or_warm_up_model` returns `CompilationTimes` (vLLM >= 0.27 executor contract) | `tests/test_worker.py` | not submitted |
| `topic/vllm-031-worker-contract` | vLLM 0.31 worker contract: `SwiftMetalWorker(WorkerBase)`, KV cache layout RPCs (`get_supported_kv_cache_layouts`, `set_kv_cache_layout`), `profile(prefix)`, explicit `synchronize_device`, `validate_request` rejects structured outputs with a 400 (eejd/py-vllm-swift#1) | `tests/test_worker_engine_contract.py`, `tests/conftest.py` | not submitted |

Adopted work from other repos: none.

The three small topics add no tests of their own (their commits touch only `vllm_swift/`); the listed tests are the existing suite that covers them. Tracking issue for the vLLM 0.31 contract: [eejd/macports-ports-local#1243](https://github.com/eejd/macports-ports-local/issues/1243).

## Tags

`v0.6.3-local01`: `v0.6.3` + the four topics above (`cd4ce07`). Later commits on `local` that only change this file are not retagged.
