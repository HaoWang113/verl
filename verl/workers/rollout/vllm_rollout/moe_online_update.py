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
"""Checkpoint-layout window for MoE expert weights during sparse delta updates.

On Ascend NPU, vllm-ascend transposes FusedMoE expert weights after load
(``process_weights_after_loading``: ``transpose(1, 2)`` per w13/w2, plus
conditional NZ / per-expert-list variants). Sparse delta patches carry
checkpoint coordinates and are applied in place, so a sparse update window
must expose the live expert weights at checkpoint layout and replay the
load-time transforms when the window closes.

Both certified vllm-ascend shapes (0.23.x and the anchored modular layout)
share the same ND main path and the pad step is a no-op on NPU (vllm core
``_maybe_pad_weight`` is ROCm-only), so the window is a pure shape
permutation. Fingerprint evidence:
``openspec/changes/npu-moe-sparse-delta-sync/notes/0230rc1-transform-fingerprint.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import version as _package_version

import torch

logger = logging.getLogger(__name__)

_VLLM_ASCEND_CERTIFIED_RELEASES = {"0.23.0rc1"}
_VLLM_ASCEND_CERTIFIED_PREFIXES = ("0.23.0",)
_ANCHORED_HOST_MODULE = "vllm_ascend.ops.fused_moe.routed_experts"
_V023_HOST_MODULE = "vllm_ascend.ops.fused_moe.fused_moe"

_LIST_SPLIT_ATTRS = ("w13_weight_list", "w2_weight_list")


@dataclass
class ExpertLayoutWindow:
    """Layers whose expert weights currently sit at checkpoint layout."""

    flipped_layers: list[torch.nn.Module] = field(default_factory=list)


def _installed_vllm_ascend_version() -> str:
    try:
        return _package_version("vllm-ascend")
    except Exception:
        return "unknown"


def assert_online_update_fingerprint() -> str:
    """Fail loud unless the installed vllm-ascend matches a certified layout.

    Two layers: a coarse version gate (certified releases / 0.23.0 series /
    setuptools_scm dev builds) and a structural host-shape probe -- the MoE
    method module that exists tells the two generations apart. Returns the
    detected host module for diagnostics.
    """
    detected = _installed_vllm_ascend_version()
    known = (
        detected in _VLLM_ASCEND_CERTIFIED_RELEASES
        or detected.startswith(_VLLM_ASCEND_CERTIFIED_PREFIXES)
        or "dev" in detected
        or "+" in detected
    )
    if not known:
        raise NotImplementedError(
            f"vllm-ascend {detected!r} is not certified for the MoE sparse layout window; certified: "
            f"{sorted(_VLLM_ASCEND_CERTIFIED_RELEASES)} / {_VLLM_ASCEND_CERTIFIED_PREFIXES} prefixes / dev builds. "
            "Certify the transform sequence against "
            "openspec/changes/npu-moe-sparse-delta-sync/notes/0230rc1-transform-fingerprint.md first."
        )
    for module_name in (_ANCHORED_HOST_MODULE, _V023_HOST_MODULE):
        try:
            import_module(module_name)
        except ImportError:
            continue
        return module_name
    raise NotImplementedError(
        f"vllm-ascend {detected!r} exposes neither {_ANCHORED_HOST_MODULE} nor {_V023_HOST_MODULE}; the MoE "
        "transform sequence is unrecognized and the sparse layout window refuses to guess."
    )


def _assert_nz_inactive() -> None:
    # BF16 expert weights are physically NZ only when weight_nz_mode == 2, on
    # 310P hardware, or under a FORCE_NZ hardware profile; verl's delta session
    # bypasses the host worker's _check_nz_disabled, so this is the enforcing
    # gate (memo: conclusion B).
    try:
        from vllm_ascend.ascend_config import get_ascend_config
    except ImportError as exc:
        raise NotImplementedError("vllm-ascend is required for the NPU sparse expert layout window") from exc

    nz_mode = getattr(get_ascend_config(), "weight_nz_mode", 0)
    if nz_mode:
        # Conservative on purpose: mode 1 leaves BF16 in ND, but the session
        # gate contract (host _check_nz_disabled semantics) rejects any NZ
        # mode so only the certified configuration reaches the window.
        raise NotImplementedError(
            f"weight_nz_mode={nz_mode} fails the NZ session gate; sparse in-place updates require "
            "weight_nz_mode=0 (verified configuration). Set it via --additional-config, or enable "
            "additional_config.rl_config which forces it to 0."
        )
    try:
        from vllm_ascend.utils import is_310p

        if is_310p():
            raise NotImplementedError(
                "310P stores expert weights in FRACTAL_NZ unconditionally; sparse in-place updates are not "
                "supported on this hardware profile."
            )
    except ImportError:
        pass
    try:
        from vllm_ascend.utils import WeightLayoutPolicy, get_current_hardware_profile

        if get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ:
            raise NotImplementedError(
                "The hardware profile forces FRACTAL_NZ expert weight storage; sparse in-place updates are "
                "not supported on this profile."
            )
    except ImportError:
        pass


def _replace_parameter(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    try:
        from vllm.model_executor.utils import replace_parameter
    except ImportError:
        replace_parameter = None

    if replace_parameter is not None:
        replace_parameter(layer, name, tensor)
    else:
        getattr(layer, name).data = tensor


def iter_expert_weight_modules(model: torch.nn.Module):
    """Yield fused-expert modules by structure (survives class renames)."""
    for _, module in model.named_modules():
        if getattr(module, "w13_weight", None) is not None or getattr(module, "w13_weight_list", None) is not None:
            yield module


def _assert_layer_supported(layer: torch.nn.Module) -> None:
    for list_attr in _LIST_SPLIT_ATTRS:
        if getattr(layer, list_attr, None) is not None:
            raise NotImplementedError(
                f"Per-expert weight list variant ({list_attr}; megamoe/dynamic-eplb fused-MC2) deletes the "
                "parameter body, leaving no in-place write target for sparse patches."
            )
    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    if not isinstance(w13, torch.nn.Parameter) or not isinstance(w2, torch.nn.Parameter):
        raise NotImplementedError(
            f"Unexpected expert weight binding on {type(layer).__name__}: expected Parameter-typed "
            "w13_weight/w2_weight."
        )
    if w13.dim() != 3 or w2.dim() != 3:
        raise NotImplementedError(
            f"Unexpected expert weight rank on {type(layer).__name__}: w13{tuple(w13.shape)} / "
            f"w2{tuple(w2.shape)}; the certified transform expects 3D [E, 2N, K] / [E, K, N] tensors."
        )
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"Expert weights must be BF16 for the sparse layout window, got w13={w13.dtype} w2={w2.dtype}."
        )


def prepare_expert_weights_for_online_update(layer: torch.nn.Module, window: ExpertLayoutWindow) -> None:
    """Flip one layer's expert weights: runtime [E, K, 2N] -> checkpoint [E, 2N, K]."""
    _assert_layer_supported(layer)
    _assert_nz_inactive()
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    # Build the checkpoint-layout copies first, then release the runtime
    # storages: transient overhead stays at this single layer's size.
    ckpt_w13 = w13.transpose(1, 2).contiguous()
    ckpt_w2 = w2.transpose(1, 2).contiguous()
    _replace_parameter(layer, "w13_weight", ckpt_w13)
    _replace_parameter(layer, "w2_weight", ckpt_w2)
    del w13, w2
    window.flipped_layers.append(layer)


def restore_expert_weights_after_online_update(layer: torch.nn.Module) -> None:
    """Replay the load-time transform for one flipped layer (ND main path)."""
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    runtime_w13 = w13.transpose(1, 2).contiguous()
    runtime_w2 = w2.transpose(1, 2).contiguous()
    _replace_parameter(layer, "w13_weight", runtime_w13)
    _replace_parameter(layer, "w2_weight", runtime_w2)
    del w13, w2


def open_expert_layout_window(model: torch.nn.Module) -> ExpertLayoutWindow | None:
    """Open the window: flip every fused-expert module to checkpoint layout.

    Returns ``None`` (a no-op window) when the model has no expert weight
    modules -- dense models on any platform pass through untouched, and the
    fingerprint gate is only consulted when there is something to flip.
    """
    modules = list(iter_expert_weight_modules(model))
    if not modules:
        return None
    host_module = assert_online_update_fingerprint()
    logger.info(
        "sparse delta: opening MoE layout window over %d expert module(s) (host shape %s)",
        len(modules),
        host_module,
    )
    window = ExpertLayoutWindow()
    for module in modules:
        prepare_expert_weights_for_online_update(module, window)
    return window


def close_expert_layout_window(window: ExpertLayoutWindow) -> None:
    """Close the window: replay load-time transforms for the flipped layers."""
    for layer in window.flipped_layers:
        restore_expert_weights_after_online_update(layer)
    window.flipped_layers.clear()
