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
load-time transforms when the window closes. The flips rebind through
``param.data`` so the Parameter object -- and the ``weight_loader``
attribute the sparse apply's ``model.load_weights`` re-entry reads --
survives every open/restore cycle.

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
    """Rebind one expert weight through ``param.data`` -- the Parameter object survives.

    Keeping the object preserves every attribute attached at load time, above
    all ``weight_loader``: the sparse apply re-enters ``model.load_weights``,
    whose fused-expert loader reads ``param.weight_loader`` off the live
    parameters. vllm's ``replace_parameter`` carries that attribute over only
    on hosts that fixed the attr carry-over (observed dropping on the 0.23.0
    generation), so the window does not depend on it. The storage swap is
    intended: the transpose must materialize a new storage either way."""
    getattr(layer, name).data = tensor


def iter_expert_weight_modules(model: torch.nn.Module):
    """Yield fused-expert modules by structure (survives class renames)."""
    for _, module in model.named_modules():
        if getattr(module, "w13_weight", None) is not None or getattr(module, "w13_weight_list", None) is not None:
            yield module


def _runtime_host_file() -> str:
    """Best-effort path of the installed host fused-MoE module (diagnostics only)."""
    for module_name in (_V023_HOST_MODULE, _ANCHORED_HOST_MODULE):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        return str(getattr(module, "__file__", module_name))
    return "unknown"


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
    # The apply path densifies sparse patches into NaN-masked checkpoint
    # tensors and re-enters model.load_weights, whose fused-expert loader
    # reads param.weight_loader on the live parameters. Stock vllm-ascend
    # 0.23.0rc1 rebinds transposed experts as fresh Parameters in
    # process_weights_after_loading and drops that attribute, so the first
    # expert flush dies with AttributeError mid-apply. Fail loud at window
    # open instead, with the fix pointer.
    if not hasattr(w13, "weight_loader") or not hasattr(w2, "weight_loader"):
        host_file = _runtime_host_file()
        raise NotImplementedError(
            f"Expert weights on {type(layer).__name__} carry no weight_loader attribute: the host "
            "vllm-ascend rebinds transposed experts as fresh Parameters in process_weights_after_loading. "
            "The sparse apply re-enters model.load_weights, which reads param.weight_loader; apply the "
            "vllm-ascend-0.23.0rc1 delta-repo fix or "
            "move to a host that preserves loader attributes across post-load transforms. "
            f"Runtime host module: {host_file} -- verify the patch is live in THAT file with "
            f"grep -n 'Rebind through' {host_file} (no hit = the runtime install is unpatched; note that "
            "reinstalling vllm-ascend wipes a patch applied inside site-packages)."
        )


def prepare_expert_weights_for_online_update(layer: torch.nn.Module, window: ExpertLayoutWindow) -> None:
    """Flip one layer's expert weights: runtime [E, K, 2N] -> checkpoint [E, 2N, K]."""
    _assert_layer_supported(layer)
    _assert_nz_inactive()
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    # [DIAG-WINDOW 2026-10-08] Log the runtime->checkpoint layout flip per layer.
    # The sparse delta applies patches at checkpoint coordinates; if this flip is
    # a no-op or wrong-orient, copy_non_nan_ writes into runtime-layout storage
    # and the shape mismatch surfaces downstream. Pair with [DIAG] (apply entry)
    # to confirm the live param shape matches the patch shape when the apply runs.
    try:
        logger.warning(
            "[DIAG-WINDOW prepare] layer=%s w13 runtime=%s -> ckpt=%s | w2 runtime=%s -> ckpt=%s",
            type(layer).__name__,
            tuple(w13.shape), tuple(w13.transpose(1, 2).shape),
            tuple(w2.shape), tuple(w2.transpose(1, 2).shape),
        )
    except Exception:
        pass
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
    logger.warning(
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


# ---- per-expert direct-apply (bypass model.load_weights fused-expert loader) ----
# The fused-expert loader (e.g. qwen3_5.py:load_fused_expert_weights) loops
# ``for expert_id in range(num_experts): curr = loaded_weight[expert_id]`` and
# assumes loaded_weight is 3D-stacked [E,2N,K] (expert_id from dim 0, NOT from
# the name). Dense seed (v=0) lands a whole layer's experts in one
# model.load_weights call, so AutoWeightsLoader groups+stacks them into 3D and
# the fused loader slices correctly. Sparse delta (v>=1) ships per-expert 2D
# [N,K] patches per flush; the mapper maps the per-expert name back to the
# fused name, but the 2D tensor cannot be stacked -> the fused loader slices a
# 1D row -> copy_non_nan_ shape mismatch (dest (512,2048) != src (2048,)).
# This path writes per-expert patches directly into the checkpoint-layout live
# parameters (the sparse window already flipped them to [E,2N,K]/[E,K,N]),
# bypassing the fused loader entirely. TP sharding is derived from the live
# parameter shape vs the patch shape, so no vllm_config plumbing is needed.

_EXPERT_PATCH_RE = None


def _expert_patch_name_regex():
    global _EXPERT_PATCH_RE
    if _EXPERT_PATCH_RE is None:
        import re

        _EXPERT_PATCH_RE = re.compile(
            r"^(?P<layer>.+\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)(?:\.weight)?$"
        )
    return _EXPERT_PATCH_RE


def parse_expert_patch_name(name: str):
    """Return ``(layer_prefix, expert_id, proj)`` for a per-expert HF checkpoint
    name, or ``None`` if the name is not a per-expert MoE checkpoint name."""
    m = _expert_patch_name_regex().match(name)
    if m is None:
        return None
    return m.group("layer"), int(m.group("eid")), m.group("proj")


def apply_expert_patch_direct(model: torch.nn.Module, name: str, shape, dtype, values, indices) -> bool:
    """Apply one sparse per-expert patch directly into the checkpoint-layout live
    parameter, bypassing ``model.load_weights``. Returns ``True`` if the patch
    was applied here, ``False`` if it is not a per-expert patch (caller should
    fall back to the normal loader path).

    The sparse window must already be open (parameters at checkpoint layout):
    ``w13_weight`` = [E, 2N, K], ``w2_weight`` = [E, K, N] (TP-narrowed). The
    patch carries per-expert checkpoint coordinates: ``shape`` = [N, K] (gate/up)
    or [K, N] (down), ``indices`` = int32 flat positions into ``shape``, ``values``
    = the changed elements. We densify into a NaN-masked [N,K]/[K,N] tensor,
    TP-narrow it to match the live parameter's per-expert slot, then NaN-masked
    copy into ``w13_weight[e, slot, :]`` / ``w2_weight[e]``."""
    parsed = parse_expert_patch_name(name)
    if parsed is None:
        return False
    layer_prefix, expert_id, proj = parsed

    # Find the fused-expert module matching this layer_prefix. ``layer_prefix``
    # is e.g. "model.layers.0.mlp.experts"; the FusedMoE module is the one whose
    # qualname equals layer_prefix (or starts with it under base_layer wrapping).
    target_module = None
    for mod_qualname, module in model.named_modules():
        if mod_qualname == layer_prefix or mod_qualname.startswith(layer_prefix + "."):
            if getattr(module, "w13_weight", None) is not None or getattr(module, "w13_weight_list", None) is not None:
                target_module = module
                break
    if target_module is None:
        return False
    w13 = getattr(target_module, "w13_weight", None)
    w2 = getattr(target_module, "w2_weight", None)
    if w13 is None or w2 is None:
        return False

    # EP: map the global expert_id to the local physical id. If this expert is
    # not on this rank (EP-sharded), skip it (the HCCL broadcast reaches every
    # worker but each worker only owns its local experts).
    expert_map_manager = getattr(target_module, "expert_map_manager", None)
    if expert_map_manager is not None:
        local_expert_id = expert_map_manager.map_global_to_local(expert_id)
        if local_expert_id == -1:
            return True  # not local; nothing to write on this rank
        expert_id = local_expert_id

    import math

    numel = int(math.prod(shape))
    # Densify: full [N,K]/[K,N] checkpoint tensor, NaN at unchanged positions.
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    if indices is not None and int(indices.numel()):
        flat.index_copy_(
            0,
            indices.to(device=flat.device, dtype=torch.long),
            values.to(dtype=dtype, device=flat.device),
        )
    elif indices is None and int(values.numel()):
        # dense patch (full coverage): values already fill the whole tensor
        flat = values.to(dtype=dtype, device=flat.device)
    checkpoint_tensor = flat.view(shape)

    # w13 checkpoint layout [E, 2N, K]: gate_proj (w1) -> [e, :N, :], up_proj (w3)
    # -> [e, N:, :]; w2 checkpoint layout [E, K, N]: down_proj -> [e, :, :].
    # The live parameter is already TP-sharded, so select the per-expert slot
    # FIRST and then narrow the densified checkpoint tensor to that slot's shape
    # (the slot's shape encodes the TP shard: 2N->2N/TP, K->K/TP, N->N/TP).
    if proj == "gate_proj":
        slot_full_n = w13.shape[1] // 2
        dest = w13.data[expert_id, :slot_full_n, :]
    elif proj == "up_proj":
        slot_full_n = w13.shape[1] // 2
        dest = w13.data[expert_id, slot_full_n:, :]
    else:  # down_proj
        dest = w2.data[expert_id]

    # Narrow the densified checkpoint tensor to the live slot's per-rank shape.
    # checkpoint_tensor dims (before TP): gate/up [N_full, K_full], down [K_full, N_full].
    # dest dims (after TP): gate/up [N_full/TP, K_full/TP], down [K_full/TP, N_full/TP].
    # Each dim where checkpoint_tensor is larger than dest is a TP-sharded dim;
    # pick this rank's slice using the module's tp_rank.
    tp_rank = getattr(target_module, "tp_rank", 0)
    src = checkpoint_tensor
    for d in range(src.ndim):
        if src.shape[d] > dest.shape[d]:
            per_rank = dest.shape[d]
            start = per_rank * tp_rank
            src = src.narrow(d, start, per_rank)
    src = src.to(dtype=dest.dtype, device=dest.device)
    if dest.shape != src.shape:
        return False

    torch.where(torch.isnan(src), dest, src, out=dest)
    return True
