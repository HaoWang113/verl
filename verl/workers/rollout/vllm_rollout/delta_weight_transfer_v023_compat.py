# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vllm 0.23.x compatibility shim for the VERL delta weight-transfer backends.

**NOT FOR UPSTREAM** -- deployment/testing-only layer that lets the
``verl_delta_ipc`` / ``verl_delta_hccl`` backends run on a vLLM 0.23.x host
with vllm-ascend 0.23.x (the deployed stack of the Ascend NPU test
environment). Everything here is inert on hosts that provide the #44353
engine-owned WeightTransferEngine API.

Why a shim is needed
--------------------
vLLM 0.23.x already ships the weight-transfer framework base (#31943): the
factory registry, the worker-side RPC methods, and the layerwise reload API
are all present. What differs from #44353 hosts is the engine contract:

* engines are constructed as ``(config, parallel_config, model)`` -- no
  ``vllm_config`` / ``device`` -- and the base class stores only those three;
* the worker owns the update lifecycle: ``start_weight_update`` runs the
  layerwise-reload setup itself and ``update_weights`` calls the engine in
  callback style (``receive_weights(update_info, load_weights)``), a call
  shape a self-applying engine never fits;
* the worker's ``update_weights`` does not index a list of per-worker update
  infos.

The vllm-ascend 0.23.x NPUWorker mirrors this exact surface (including the
3-arg factory call in ``load_model``).

What this module does
---------------------
* :func:`classify_weight_transfer_host` / :func:`classify_installed_host`
  detect the installed vLLM's engine API shape. Only the 0.23.x series is
  accepted in compat mode; other pre-#44353 versions (0.24/0.25) fail loud
  instead of guessing, because their worker driving surface is unverified.
* ``VerlDelta*WeightTransferEngineV023`` subclasses construct under the
  3-arg factory call, receive the missing context through
  ``hydrate_v023`` (the worker extension calls it before the engine's first
  RPC), and add the concrete ``update_weights`` entry the 0.23.x base lacks
  (parse -> receive -> synchronize).
* The ``worker_*`` helpers drive the engine lifecycle from verl's worker
  extension, mirroring the #44353 worker semantics (including per-worker
  list indexing). They are invoked from the extension's ``verl_delta_*``
  methods: vLLM appends extension classes AFTER the worker class in the MRO
  and asserts the extension defines no attribute the worker already has, so
  the extension cannot override the host lifecycle method names and must use
  VERL-specific ones, delegating back to the host methods whenever the
  worker's engine is not a VERL compat engine.
* :func:`compat_engine_cls` swaps the feature engine classes for their V023
  subclasses at registration time (identity on non-0.23.x hosts).

The checkpoint patch API (#50723) is deliberately NOT part of this shim:
the upstream file is vendored under ``_vendored/`` and
``delta_weight_transfer._checkpoint_patch_api`` prefers the native module
whenever the installed vLLM provides it.
"""

from __future__ import annotations

import inspect

import torch
import vllm
from vllm.distributed.weight_transfer.base import WeightTransferEngine

from verl.workers.rollout.vllm_rollout.delta_weight_transfer import (
    VerlDeltaHCCLWeightTransferEngine,
    VerlDeltaIPCWeightTransferEngine,
)

HOST_ANCHORED = "anchored"
"""Host provides the #44353 engine-owned WeightTransferEngine API."""

HOST_V023 = "vllm-0.23"
"""vLLM 0.23.x host, driven through this compat shim."""

HOST_UNSUPPORTED = "unsupported"
"""Any other engine API shape, including unverified pre-#44353 versions."""

_ANCHORED_INIT_PARAMS = frozenset({"config", "vllm_config", "device", "model"})
_V023_INIT_PARAMS = frozenset({"config", "parallel_config", "model"})


def _is_v023(vllm_version: str) -> bool:
    """Whether a version string is a vLLM 0.23.x release (incl. rc/dev)."""
    from packaging.version import InvalidVersion, Version

    try:
        release = Version(str(vllm_version)).release
    except InvalidVersion:
        return False
    return len(release) >= 2 and release[0] == 0 and release[1] == 23


def classify_weight_transfer_host(init_param_names, vllm_version: str) -> str:
    """Classify a vLLM install by its WeightTransferEngine API shape.

    Pure function over (init parameter names, vLLM version) so it stays unit
    testable without swapping the installed host.
    """
    names = frozenset(init_param_names) - {"self"}
    if _ANCHORED_INIT_PARAMS <= names:
        return HOST_ANCHORED
    if _V023_INIT_PARAMS <= names and _is_v023(vllm_version):
        return HOST_V023
    return HOST_UNSUPPORTED


def classify_installed_host() -> str:
    """Classify the installed vLLM (engine init signature + version)."""
    init_params = inspect.signature(WeightTransferEngine.__init__).parameters
    return classify_weight_transfer_host(init_params, vllm.__version__)


def compat_engine_cls(engine_cls: type) -> type:
    """Swap a feature engine class for its 0.23.x subclass when needed.

    Identity on non-0.23.x hosts, so registration behavior there is
    unchanged.
    """
    if classify_installed_host() != HOST_V023:
        return engine_cls
    if engine_cls is VerlDeltaIPCWeightTransferEngine:
        return VerlDeltaIPCWeightTransferEngineV023
    if engine_cls is VerlDeltaHCCLWeightTransferEngine:
        return VerlDeltaHCCLWeightTransferEngineV023
    raise RuntimeError(f"no vllm 0.23.x compat engine for {engine_cls.__name__}")


def _synchronize(device: torch.device | None) -> None:
    """Accelerator sync mirroring the #44353 base's update_weights tail."""
    if device is None or device.type == "cpu":
        return
    if device.type == "npu":
        torch.npu.synchronize()
    else:
        torch.accelerator.synchronize()


class VerlDeltaIPCWeightTransferEngineV023(VerlDeltaIPCWeightTransferEngine):
    """``VerlDeltaIPCWeightTransferEngine`` for vllm 0.23.x hosts.

    The host factory constructs engines as ``(config, parallel_config,
    model)``; ``vllm_config`` / ``device`` / ``model_config`` are injected
    later through ``hydrate_v023`` (called by the worker extension before the
    engine's first RPC), after which every inherited feature method works
    unchanged.
    """

    def __init__(self, config, parallel_config, model) -> None:
        # The 0.23.x base __init__ stores (config, parallel_config, model)
        # only; the feature __init__ cannot be reused here because its 4-arg
        # super() chain targets #44353 hosts.
        WeightTransferEngine.__init__(self, config, parallel_config, model)
        self.vllm_config = None
        self.device = None
        self.model_config = None
        self._session_encoding = None
        self._update_failed = False

    def hydrate_v023(self, *, vllm_config, device) -> None:
        """Provide the context a #44353 host would pass at construction."""
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.device = device

    def update_weights(self, update_info: dict) -> None:
        """Concrete update entry the 0.23.x base class lacks.

        Mirrors the #44353 base: parse, receive, then synchronize so the
        next step uses the new weights.
        """
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)
        _synchronize(self.device)


class VerlDeltaHCCLWeightTransferEngineV023(
    VerlDeltaHCCLWeightTransferEngine,
    VerlDeltaIPCWeightTransferEngineV023,
):
    """``VerlDeltaHCCLWeightTransferEngine`` for vllm 0.23.x hosts."""

    def __init__(self, config, parallel_config, model) -> None:
        VerlDeltaIPCWeightTransferEngineV023.__init__(self, config, parallel_config, model)
        # Mirror the feature __init__ state.
        self._rendezvous = None
        self._manifest_sub = None
        self._comm = None
        self._group_rank = None


def _verl_delta_engine(worker):
    """The worker's engine if it is a VERL 0.23.x compat engine, else None."""
    engine = getattr(worker, "weight_transfer_engine", None)
    return engine if isinstance(engine, VerlDeltaIPCWeightTransferEngineV023) else None


def _local_update_info(update_info, *, data_parallel_rank: int, world_size: int, rank: int):
    """Select this worker's entry, mirroring the #44353 worker's indexing."""
    if isinstance(update_info, list):
        return update_info[data_parallel_rank * world_size + rank]
    return update_info


def worker_init_weight_transfer_engine(worker, init_info: dict) -> bool:
    """vllm 0.23.x leg of the extension's verl_delta_init RPC.

    Returns False when the worker's engine is not a VERL compat engine, so
    the caller falls back to the host worker method.
    """
    engine = _verl_delta_engine(worker)
    if engine is None:
        return False
    engine.hydrate_v023(vllm_config=worker.vllm_config, device=worker.device)
    engine.init_transfer_engine(engine.parse_init_info(init_info))
    return True


def worker_start_weight_update(worker) -> bool:
    """vllm 0.23.x leg of the extension's verl_delta_start RPC."""
    engine = _verl_delta_engine(worker)
    if engine is None:
        return False
    if getattr(worker, "_weight_update_active", False):
        raise RuntimeError(
            "start_weight_update called while a weight update is already active. "
            "Call finish_weight_update first."
        )
    engine.start_weight_update()
    worker._weight_update_active = True
    return True


def worker_update_weights(worker, update_info) -> bool:
    """vllm 0.23.x leg of the extension's delta update RPC."""
    engine = _verl_delta_engine(worker)
    if engine is None:
        return False
    if not getattr(worker, "_weight_update_active", False):
        raise RuntimeError("start_weight_update must be called before update_weights.")
    parallel_config = worker.vllm_config.parallel_config
    local_update_info = _local_update_info(
        update_info,
        data_parallel_rank=parallel_config.data_parallel_rank,
        world_size=parallel_config.world_size,
        rank=worker.rank,
    )
    try:
        engine.update_weights(local_update_info)
    except BaseException:
        worker._weight_update_active = False
        raise
    return True


def worker_finish_weight_update(worker) -> bool:
    """vllm 0.23.x leg of the extension's verl_delta_finish RPC."""
    engine = _verl_delta_engine(worker)
    if engine is None:
        return False
    if not getattr(worker, "_weight_update_active", False):
        raise RuntimeError("finish_weight_update called without a matching start_weight_update.")
    engine.finish_weight_update()
    worker._weight_update_active = False
    # #44353 workers reset LoRA state after a non-draft update; guard for
    # hosts whose model runner lacks the method.
    reset_lora_state = getattr(worker.model_runner, "reset_lora_state", None)
    if callable(reset_lora_state):
        reset_lora_state()
    return True
