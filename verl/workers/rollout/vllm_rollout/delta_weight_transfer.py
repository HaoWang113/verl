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
"""Receive ``DeltaFlush`` payloads inside vLLM workers.

VERL sends each payload through its existing ZMQ/CUDA-IPC channel. This
adapter decodes it and calls vLLM's checkpoint patch loader, which handles
checkpoint names, packed weights, and TP slicing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import vllm

from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)

try:
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        WeightTransferEngine,
        WeightTransferInitInfo,
        WeightTransferUpdateInfo,
    )
except ImportError as exc:
    raise RuntimeError(
        "vLLM delta_sharded support requires the vllm.config.weight_transfer "
        "and vllm.distributed.weight_transfer modules"
    ) from exc

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.checkpoint_weight_patch import (
        CheckpointWeightPatch,
    )
    from verl.workers.rollout.vllm_rollout.moe_online_update import ExpertLayoutWindow

VERL_DELTA_WEIGHT_TRANSFER_BACKEND = "verl_delta_ipc"
VERL_DELTA_HCCL_WEIGHT_TRANSFER_BACKEND = "verl_delta_hccl"
_REGISTERED = False
_HCCL_REGISTERED = False


def _checkpoint_patch_api():
    try:
        from vllm.model_executor.model_loader.checkpoint_weight_patch import (
            CheckpointWeightPatch,
            load_checkpoint_weight_patches,
        )
    except ImportError as exc:
        if exc.name != "vllm.model_executor.model_loader.checkpoint_weight_patch":
            raise
        # vllm 0.23.x compat (NOT FOR UPSTREAM): fall back to the vendored copy
        # of vLLM #50723; see delta_weight_transfer_v023_compat.
        from verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch import (
            CheckpointWeightPatch,
            load_checkpoint_weight_patches,
        )
        return CheckpointWeightPatch, load_checkpoint_weight_patches
    return CheckpointWeightPatch, load_checkpoint_weight_patches


def require_vllm_delta_support() -> None:
    """Verify that vLLM provides the required weight transfer and checkpoint patch APIs.

    Accepts either a #44353 host (engine-owned WeightTransferEngine API) or,
    through the vllm 0.23.x compat shim, a vLLM 0.23.x host; anything else
    fails loud. NOT FOR UPSTREAM: the 0.23.x leg is deployment/test-only
    (see delta_weight_transfer_v023_compat).
    """
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer_v023_compat import (
        HOST_UNSUPPORTED,
        classify_installed_host,
    )

    if classify_installed_host() == HOST_UNSUPPORTED:
        raise RuntimeError(
            "checkpoint_engine.backend='delta_sharded' with vLLM requires the engine-owned WeightTransferEngine "
            "API from vLLM #44353 and the checkpoint patch API from vLLM #50723, or a vLLM 0.23.x host through "
            "verl's compat shim (other pre-#44353 versions are not covered); installed vLLM "
            f"{getattr(vllm, '__version__', 'unknown')} provides neither. "
            "Install a compatible vLLM build or select a different checkpoint engine backend."
        )
    _checkpoint_patch_api()


def resolve_moe_delta_settings(*, is_npu: bool, moe_backend: Any, enable_eplb: bool) -> str | None:
    """Validate MoE delta settings per platform; return the backend to force, if any.

    CUDA keeps the historical triton-only contract (identical raise order and
    messages). NPU no longer rejects MoE outright: the sparse layout window
    (moe_online_update) replaces the kernel-backend requirement, so backend
    pins are refused and the vllm-ascend default stays untouched. EPLB is
    refused on both platforms -- dynamic_eplb splits expert weights into
    per-expert lists with no in-place write target.
    """
    if is_npu:
        if moe_backend not in {None, "auto"}:
            raise NotImplementedError(
                "delta_sharded with vLLM MoE on Ascend NPU relies on the checkpoint-layout window, not a kernel "
                f"backend; leave moe_backend unset or 'auto' (got {moe_backend!r})"
            )
        if enable_eplb:
            raise NotImplementedError(
                "delta_sharded with vLLM MoE does not support EPLB (dynamic_eplb splits expert weights into "
                "per-expert lists)"
            )
        return None
    if moe_backend not in {None, "auto", "triton"}:
        raise NotImplementedError(f"delta_sharded with vLLM MoE requires moe_backend='triton'; got {moe_backend!r}")
    if enable_eplb:
        raise NotImplementedError("delta_sharded with vLLM MoE does not support EPLB")
    return "triton"


def is_moe_model(hf_config) -> bool:
    """Whether vLLM will treat this HF config as a MoE model.

    Delta updates require the Triton MoE backend, so this check must agree with
    the engine's own MoE detection: expert counts may live under a different
    attribute (Dbrx ``moe_num_experts``, DeepSeek ``n_routed_experts``, Mixtral
    ``num_local_experts``) or on a nested text config (multimodal MoE). The
    fallback mirrors vLLM's attribute list for configs its convertor cannot
    process.
    """
    try:
        from vllm.transformers_utils.config import get_hf_text_config
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            ModelArchConfigConvertorBase,
        )

        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            getattr(hf_config, "model_type", ""), ModelArchConfigConvertorBase
        )
        return int(convertor_cls(hf_config, get_hf_text_config(hf_config)).get_num_experts()) > 0
    except Exception:
        text_config = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        for name in ("num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"):
            num_experts = getattr(text_config, name, 0)
            if isinstance(num_experts, list):
                num_experts = num_experts[0] if num_experts else 0
            if num_experts and int(num_experts) > 0:
                return True
        return False


@dataclass
class VerlDeltaIPCInitInfo(WeightTransferInitInfo):
    """No initialization data is needed for the local IPC transport."""


@dataclass
class VerlDeltaIPCUpdateInfo(WeightTransferUpdateInfo):
    """Connection details for one flush."""

    zmq_handle: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.zmq_handle, str) or not self.zmq_handle:
            raise ValueError("VERL delta IPC updates require a worker-local zmq_handle")


def decode_delta_payload(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[str, list[CheckpointWeightPatch]]:
    """Validate and decode one DeltaFlush into vLLM checkpoint patches.

    The wire validation and per-parameter slicing live in the device-agnostic
    :func:`verl.checkpoint_engine.delta_sync.decode_delta_wire` (shared with the
    NPU HCCL receiver and the tests); this wrapper only converts the decoded
    patches into vLLM's ``CheckpointWeightPatch``.
    """
    from verl.checkpoint_engine.delta_sync.decode import decode_delta_wire

    CheckpointWeightPatch, _ = _checkpoint_patch_api()
    encoding, patches = decode_delta_wire(named_tensors)
    return encoding, [
        CheckpointWeightPatch(
            name=p.name,
            shape=p.shape,
            dtype=p.dtype,
            values=p.values,
            indices=p.positions,
        )
        for p in patches
    ]


class VerlDeltaIPCWeightTransferEngine(WeightTransferEngine):
    """Consume VERL DeltaFlush payloads over the colocated CUDA-IPC channel."""

    init_info_cls = VerlDeltaIPCInitInfo
    update_info_cls = VerlDeltaIPCUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._session_encoding: str | None = None
        self._update_failed = False
        self._sparse_session_begun = False

    def _validate_configuration(self) -> None:
        require_vllm_delta_support()
        if self.device.type != get_device_name():
            raise NotImplementedError(f"VERL delta weight transfer requires {get_device_name()!r} devices")
        if self.parallel_config.data_parallel_size != 1:
            raise NotImplementedError("VERL delta weight transfer requires data_parallel_size=1")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("VERL delta weight transfer requires pipeline_parallel_size=1")
        if self.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("VERL delta weight transfer supports only BF16 vLLM models")
        if getattr(self.vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("VERL delta weight transfer does not support quantized vLLM models")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("VERL delta weight transfer does not support speculative decoding")

    def init_transfer_engine(self, init_info: VerlDeltaIPCInitInfo) -> None:
        self._validate_configuration()

    def start_weight_update(self) -> None:
        if self._update_failed:
            raise RuntimeError(
                "A previous delta update did not complete; restart the full job so every rollout worker receives a "
                "fresh dense seed. Restarting only the vLLM workers is unsafe because the producer snapshot may "
                "already have advanced."
            )
        if self._sparse_session_begun:
            raise RuntimeError(
                "A previous sparse weight-update session never finished cleanly; restart the full job so every "
                "rollout worker receives a fresh dense seed."
            )
        self._session_encoding = None

    def _on_sparse_session_begin(self) -> None:
        """Prepare the model for in-place checkpoint-coordinate patches.

        Invoked once, right before the first ``indices``-encoded flush of an
        update session is applied. Default no-op: the reference engines run on
        platforms whose live parameters already use checkpoint layout, so
        sparse patches land correctly as-is. Platform engines override this to
        open a layout window (see VerlDeltaHCCLWeightTransferEngine on Ascend
        NPU, where MoE expert weights are stored transposed after load).
        """

    def _on_sparse_session_end(self) -> None:
        """Close the layout window opened by :meth:`_on_sparse_session_begin`.

        Invoked from ``finish_weight_update`` (and once, best effort, from the
        failure path). Default no-op for the same reason as above.
        """

    def _receive_payload(self, *, zmq_handle: str) -> list[tuple[str, torch.Tensor]]:
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        receiver = BucketedWeightReceiver(
            zmq_handle=zmq_handle,
            device=self.device,
            use_shm=False,
        )
        payload: list[tuple[str, torch.Tensor]] = []
        callback_error: BaseException | None = None

        def retain_bucket(weights: list[tuple[str, torch.Tensor]], _is_last: bool) -> None:
            nonlocal callback_error
            if callback_error is not None:
                return
            try:
                # The receiver reuses its IPC buffer for the next bucket, so
                # retain a private copy until this flush is decoded and applied.
                payload.extend((name, tensor.clone()) for name, tensor in weights)
            except BaseException as exc:
                # Finish the receiver handshake before reporting a callback error.
                callback_error = exc

        receiver.receive_weights(on_bucket_received=retain_bucket)
        if callback_error is not None:
            raise callback_error
        return payload

    def _decode_and_apply(self, payload: list[tuple[str, torch.Tensor]]) -> None:
        """Decode one flush's named tensors into checkpoint patches and apply them."""
        encoding, patches = decode_delta_payload(payload)

        if self._session_encoding is None:
            self._session_encoding = encoding
            if encoding == "dense":
                from vllm.model_executor.model_loader.reload import (
                    initialize_layerwise_reload,
                )

                initialize_layerwise_reload(self.model)
            else:
                # Sparse patches are written in place at checkpoint
                # coordinates: open the layout window before the first flush
                # lands (no-op on reference engines).
                self._sparse_session_begun = True
                self._on_sparse_session_begin()
        elif encoding != self._session_encoding:
            raise ValueError(
                "One weight update cannot mix dense and sparse DeltaFlush payloads "
                f"({self._session_encoding!r} then {encoding!r})"
            )

        _, load_checkpoint_weight_patches = _checkpoint_patch_api()
        # The trusted producer emits every changed position at most once;
        # skip vLLM's sort-based duplicate check on the rollout GPU.
        load_checkpoint_weight_patches(
            self.model,
            patches,
            validate_unique_indices=False,
        )

    def receive_weights(self, update_info: VerlDeltaIPCUpdateInfo) -> None:
        assert update_info.zmq_handle is not None
        try:
            payload = self._receive_payload(zmq_handle=update_info.zmq_handle)
            self._decode_and_apply(payload)
        except BaseException:
            # An earlier flush may already have changed runtime weights, so a
            # later update would start from an unknown partial state.
            if self._sparse_session_begun:
                # Best effort: close the layout window so the poisoned-worker
                # restart contract is the only remaining failure mode. Exactly
                # one end attempt (flag cleared first) -- never re-enter a
                # partially restored model.
                self._sparse_session_begun = False
                try:
                    self._on_sparse_session_end()
                except BaseException:
                    logger.exception(
                        "best-effort sparse session restore failed; restart the full job so every rollout worker "
                        "receives a fresh dense seed"
                    )
            self._update_failed = True
            raise

    def finish_weight_update(self) -> None:
        if self._session_encoding == "dense":
            # Dense updates use vLLM's layerwise reload lifecycle. Finalize it
            # once after the last flush to run model-specific post-load work.
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            try:
                finalize_layerwise_reload(self.model, self.model_config)
            except BaseException:
                self._update_failed = True
                raise
        elif self._session_encoding == "indices":
            # Sparse updates replay platform load-time transforms to return
            # live parameters to runtime layout (no-op on reference engines).
            try:
                self._on_sparse_session_end()
            except BaseException:
                self._update_failed = True
                raise
            finally:
                # Exactly one end attempt: never re-enter a partially restored
                # model; the poisoned-worker contract takes over.
                self._sparse_session_begun = False
        self._session_encoding = None

    def shutdown(self) -> None:
        self._session_encoding = None
        self._sparse_session_begun = False

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError("VERL sends DeltaFlush payloads; this adapter only receives them")


def _compat_engine_cls(engine_cls: type) -> type:
    """vllm 0.23.x compat hook (NOT FOR UPSTREAM): swap the feature engine
    class for its 0.23.x subclass when the installed vLLM predates #44353;
    identity on #44353 hosts. See delta_weight_transfer_v023_compat."""
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer_v023_compat import compat_engine_cls

    return compat_engine_cls(engine_cls)


def register_verl_delta_weight_transfer_engine() -> None:
    """Register the adapter before vLLM creates its worker-side engine."""

    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(
        VERL_DELTA_WEIGHT_TRANSFER_BACKEND,
        _compat_engine_cls(VerlDeltaIPCWeightTransferEngine),
    )
    _REGISTERED = True


@dataclass
class VerlDeltaHCCLInitInfo(WeightTransferInitInfo):
    """Rendezvous endpoints of the trainer's HCCL delta broadcast group."""

    dist_ip: str
    dist_port: int
    zmq_ip: str
    zmq_port: int
    topic: str = "bucket_metadata"
    world_size: int = 1


@dataclass
class VerlDeltaHCCLUpdateInfo(WeightTransferUpdateInfo):
    """This worker's rank in the trainer's PyHccl delta broadcast group."""

    group_rank: int


class VerlDeltaHCCLWeightTransferEngine(VerlDeltaIPCWeightTransferEngine):
    """Consume VERL DeltaFlush payloads directly from the trainer's HCCL broadcasts.

    C1 receiver: instead of pulling each flush over the colocated zmq/IPC
    channel, the vLLM worker process joins the trainer's PyHccl broadcast
    group itself -- one manifest subscription plus two broadcasts per flush,
    no payload forwarding hop and no IPC handle negotiation. The zmq SUB is
    connected before the communicator is built (see
    :mod:`verl.checkpoint_engine.delta_sync.hccl_receiver`); the trainer
    publishes manifests only once the group is complete, so no manifest can be
    lost. Both the SUB and the communicator are created once and reused across
    weight-update sessions.
    """

    init_info_cls = VerlDeltaHCCLInitInfo
    update_info_cls = VerlDeltaHCCLUpdateInfo

    def __init__(self, config, vllm_config, device, model) -> None:
        super().__init__(config, vllm_config, device, model)
        self._rendezvous: VerlDeltaHCCLInitInfo | None = None
        self._manifest_sub = None
        self._comm = None
        self._group_rank: int | None = None
        self._expert_layout_window: "ExpertLayoutWindow | None" = None

    def _validate_configuration(self) -> None:
        require_vllm_delta_support()
        if self.device.type != get_device_name():
            raise NotImplementedError(f"VERL delta HCCL weight transfer requires {get_device_name()!r} devices")
        if self.parallel_config.data_parallel_size != 1:
            raise NotImplementedError("VERL delta HCCL weight transfer requires data_parallel_size=1")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("VERL delta HCCL weight transfer requires pipeline_parallel_size=1")
        if self.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("VERL delta HCCL weight transfer supports only BF16 vLLM models")
        if getattr(self.vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("VERL delta HCCL weight transfer does not support quantized vLLM models")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("VERL delta HCCL weight transfer does not support speculative decoding")

    def init_transfer_engine(self, init_info: VerlDeltaHCCLInitInfo) -> None:
        self._validate_configuration()
        self._rendezvous = init_info

    def _ensure_channel(self, group_rank: int):
        """Create (once) and return the (manifest SUB, PyHccl communicator) pair.

        The SUB is connected BEFORE the communicator joins the group: the
        trainer's lazy rendezvous only completes once every member joined, and
        it publishes the first manifest after that -- so subscribing first
        guarantees the SUB is in place before any manifest exists.
        """
        from verl.checkpoint_engine.delta_sync.hccl_receiver import (
            build_delta_hccl_communicator,
            connect_delta_manifest_sub,
        )

        if self._comm is None:
            assert self._rendezvous is not None, "init_weight_transfer_engine must run first"
            self._manifest_sub = connect_delta_manifest_sub(
                self._rendezvous.zmq_ip, self._rendezvous.zmq_port, self._rendezvous.topic
            )
            self._comm = build_delta_hccl_communicator(
                self._rendezvous.dist_ip,
                self._rendezvous.dist_port,
                group_rank,
                self._rendezvous.world_size,
                self.device,
            )
            self._group_rank = group_rank
        else:
            assert self._group_rank == group_rank, (
                f"delta HCCL group rank changed between sessions: {self._group_rank} -> {group_rank}"
            )
        return self._manifest_sub, self._comm

    def receive_weights(self, update_info: VerlDeltaHCCLUpdateInfo) -> None:
        from verl.checkpoint_engine.delta_sync.hccl_receiver import pump_hccl_delta_session

        assert self._rendezvous is not None, "init_weight_transfer_engine must run first"
        try:
            sub, comm = self._ensure_channel(update_info.group_rank)
            pump_hccl_delta_session(sub, comm, self.device, on_flush=self._decode_and_apply)
        except BaseException:
            self._update_failed = True
            raise

    def _on_sparse_session_begin(self) -> None:
        from verl.workers.rollout.vllm_rollout.moe_online_update import open_expert_layout_window

        self._expert_layout_window = open_expert_layout_window(self.model)

    def _on_sparse_session_end(self) -> None:
        if self._expert_layout_window is None:
            return
        from verl.workers.rollout.vllm_rollout.moe_online_update import close_expert_layout_window

        close_expert_layout_window(self._expert_layout_window)
        self._expert_layout_window = None

    def shutdown(self) -> None:
        super().shutdown()
        if self._expert_layout_window is not None:
            from verl.workers.rollout.vllm_rollout.moe_online_update import close_expert_layout_window

            try:
                close_expert_layout_window(self._expert_layout_window)
            except BaseException:
                logger.exception("best-effort expert layout restore during shutdown failed")
            self._expert_layout_window = None


def register_verl_delta_hccl_weight_transfer_engine() -> None:
    """Register the HCCL adapter before vLLM creates its worker-side engine."""

    global _HCCL_REGISTERED
    if _HCCL_REGISTERED:
        return
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(
        VERL_DELTA_HCCL_WEIGHT_TRANSFER_BACKEND,
        _compat_engine_cls(VerlDeltaHCCLWeightTransferEngine),
    )
    _HCCL_REGISTERED = True
