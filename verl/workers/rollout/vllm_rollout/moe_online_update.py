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

RUNTIME-AT-REST CONTRACT (confirmed 2026-10-10, raysubmit_GDrzYzFhu4B7rATx):
the unquantized Ascend kernel path feeds the live expert parameter straight
into ``torch_npu.npu_grouped_matmul`` without transposition
(``fused_moe.py:apply`` -> ``build_fused_experts_input`` leaves ``need_trans``
at its False default), and that op requires ``weight.dim(1) == hidden_size``
-- i.e. the live w13 must rest at vllm-ascend's transposed RUNTIME layout
``[E, K, 2N]`` between weight-update sessions. Leaving it at checkpoint
layout ``[E, 2N, K]`` crashes the first forward with aclnnGroupedMatmulV5
161002 ("Dim 1 value of x[0] should be equal with dim 1 value of weight[0],
but now is 2048 and 1024 respectively" -- observed on a rollout rank whose
dense-bucket seed had left w13 checkpoint-layout, Qwen3.5-35B-A3B:
E_local=128, 2N=1024, K=2048). Therefore every window CLOSE (and the
dense-bucket seed finish) must restore runtime layout, including for params
that already sat at checkpoint layout when the window opened.

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
    """Expert-weight parameters currently held at checkpoint layout.

    Tracks every ``(layer, param_name)`` pair the open pass normalized to
    checkpoint layout -- both the ones it actually flipped from runtime
    layout and the ones that already sat at checkpoint layout. ALL of them
    are transposed back to runtime layout at close time: the Ascend
    unquantized kernel path requires runtime-resting expert weights (see
    the module docstring), so a param that entered the window already at
    checkpoint layout (e.g. left there by the dense-bucket seed's
    weight-loader wrapper transpose) must not stay there after close.
    """

    at_checkpoint: list[tuple[torch.nn.Module, str]] = field(default_factory=list)


_HF_VLLM_PREFIX_PAIRS = (
    # HF checkpoint names vs vLLM module qualnames for the multimodal wrapper
    # models (qwen3_5.py): HF nests ``model.language_model`` / ``model.visual``
    # while vLLM nests ``language_model.model`` / ``visual``. Confirmed on the
    # 4+2 cluster: wire 'model.language_model.layers.0.mlp.experts.{id}...'
    # vs live param 'language_model.model.layers.0.mlp.experts.w13_weight'.
    ("model.language_model.", "language_model.model."),
    ("model.visual.", "visual."),
)


def _layer_prefix_candidates(layer_prefix: str):
    """Yield the wire prefix plus its HF<->vLLM remaps for module matching."""
    yield layer_prefix
    for hf_prefix, vllm_prefix in _HF_VLLM_PREFIX_PAIRS:
        if layer_prefix.startswith(hf_prefix):
            yield vllm_prefix + layer_prefix[len(hf_prefix):]
        elif layer_prefix.startswith(vllm_prefix):
            yield hf_prefix + layer_prefix[len(vllm_prefix):]


def _module_hidden_size(layer: torch.nn.Module) -> int:
    """The module's hidden size, used as the runtime-layout discriminator.

    Mirrors the signature verl's NPU weight-loader wrapper relies on
    (``verl.utils.vllm.npu_vllm_patch.vllm_v013_weight_loader_method_wrapper``):
    a w13 param with ``shape[1] == hidden_size`` (w2 with ``shape[2] ==
    hidden_size``) is in vllm-ascend's transposed RUNTIME layout; otherwise it
    already sits at checkpoint layout.
    """
    hidden = getattr(layer, "hidden_size", None)
    if not isinstance(hidden, int) or hidden <= 0:
        raise NotImplementedError(
            f"{type(layer).__name__} exposes no usable hidden_size attribute; the sparse expert layout window "
            "cannot decide whether expert weights sit at runtime or checkpoint layout. Fail loud instead of "
            "guessing (the NPU weight-loader wrapper requires the same attribute)."
        )
    return hidden


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
    """Normalize this layer's expert weights to checkpoint layout and record
    EVERY normalized param for the close-time runtime restore.

    Runtime layout (transposing hosts, post ``process_weights_after_loading``):
    w13 ``[E, K, 2N]`` / w2 ``[E, N, K]``, detected by dim-1 == hidden_size
    for w13 / dim-2 == hidden_size for w2 (the same signature verl's NPU
    weight-loader wrapper uses). Checkpoint layout: w13 ``[E, 2N, K]`` / w2
    ``[E, K, N]`` (TP-narrowed variants included).

    Both branches record into ``window.at_checkpoint``: the close pass
    restores runtime layout unconditionally, because the Ascend unquantized
    kernel path (``npu_grouped_matmul`` with ``need_trans=False``) crashes on
    checkpoint-resting expert weights (aclnnGroupedMatmulV5 161002, confirmed
    2026-10-10). A param that already sits at checkpoint layout at open time
    is an artifact of the dense-bucket seed's weight-loader wrapper
    transpose, not a native resting state.
    """
    _assert_layer_supported(layer)
    _assert_nz_inactive()
    hidden = _module_hidden_size(layer)
    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_flipped = w13.shape[1] == hidden
    w2_flipped = w2.shape[2] == hidden
    if not w13_flipped and w13.shape[2] != hidden:
        raise NotImplementedError(
            f"{type(layer).__name__} w13 {tuple(w13.shape)} matches neither the runtime signature [E, K, 2N] "
            f"(dim-1 == hidden_size={hidden}) nor the checkpoint signature [E, 2N, K] (dim-2 == hidden_size); "
            "refusing to guess the expert weight layout."
        )
    if not w2_flipped and w2.shape[1] != hidden:
        raise NotImplementedError(
            f"{type(layer).__name__} w2 {tuple(w2.shape)} matches neither the runtime signature [E, N, K] "
            f"(dim-2 == hidden_size={hidden}) nor the checkpoint signature [E, K, N] (dim-1 == hidden_size); "
            "refusing to guess the expert weight layout."
        )
    # [DIAG-WINDOW 2026-10-08] Log the per-param normalization decision at
    # window open. Both branches are restored to runtime layout at close:
    # "already checkpoint" is a seed artifact, not a resting state (see the
    # module docstring). Pair with [DIAG-DELTA]/[DIAG-EXPERT] on the apply
    # path to confirm patch coordinates match the live layout.
    try:
        logger.warning(
            "[DIAG-WINDOW prepare] layer=%s hidden=%d w13=%s -> %s | w2=%s -> %s",
            type(layer).__name__,
            hidden,
            tuple(w13.shape),
            "flip to checkpoint" if w13_flipped else "already checkpoint (restore at close)",
            tuple(w2.shape),
            "flip to checkpoint" if w2_flipped else "already checkpoint (restore at close)",
        )
    except Exception:
        pass
    # Build the checkpoint-layout copy first, then rebind: transient overhead
    # stays at this single layer's size.
    if w13_flipped:
        _replace_parameter(layer, "w13_weight", w13.transpose(1, 2).contiguous())
    if w2_flipped:
        _replace_parameter(layer, "w2_weight", w2.transpose(1, 2).contiguous())
    window.at_checkpoint.append((layer, "w13_weight"))
    window.at_checkpoint.append((layer, "w2_weight"))
    del w13, w2


def restore_expert_param_after_online_update(layer: torch.nn.Module, param_name: str) -> None:
    """Replay the load-time transform for one windowed expert param (checkpoint
    -> runtime, ND main path)."""
    data = getattr(layer, param_name).data
    _replace_parameter(layer, param_name, data.transpose(1, 2).contiguous())
    del data


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
    """Close the window: replay the load-time transform (checkpoint -> runtime)
    for every param the open pass normalized to checkpoint layout.

    The restore is unconditional for all recorded params: the Ascend
    unquantized kernel path requires runtime-resting expert weights, so
    params that entered the window already at checkpoint layout must be
    transposed too (leaving them at checkpoint crashes the next forward with
    aclnnGroupedMatmulV5 161002).
    """
    for layer, param_name in window.at_checkpoint:
        restore_expert_param_after_online_update(layer, param_name)
    window.at_checkpoint.clear()


def restore_expert_runtime_layout(model: torch.nn.Module) -> int:
    """Transpose every checkpoint-resting fused-expert param back to runtime
    layout (ND main path), for session finishes that bypass the sparse window.

    Used by the dense-bucket seed finish: each expert ``weight_loader`` write
    went through verl's NPU wrapper, which transposes runtime params to
    checkpoint layout before writing, so the seed leaves the live expert
    weights at checkpoint layout while the kernel path requires runtime
    (vllm's ``process_weights_after_loading`` util does not reach the
    vllm-ascend fused-MoE weights). Params already at runtime layout are
    left untouched. Returns the number of transposed params.
    """
    n_restored = 0
    for layer in iter_expert_weight_modules(model):
        _assert_layer_supported(layer)
        hidden = _module_hidden_size(layer)
        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        # Checkpoint signatures: w13 [E, 2N, K] (dim-2 == hidden), w2 [E, K, N]
        # (dim-1 == hidden). Runtime: w13 [E, K, 2N] (dim-1 == hidden), w2
        # [E, N, K] (dim-2 == hidden).
        if w13.shape[1] != hidden and w13.shape[2] == hidden:
            _replace_parameter(layer, "w13_weight", w13.transpose(1, 2).contiguous())
            n_restored += 1
        if w2.shape[2] != hidden and w2.shape[1] == hidden:
            _replace_parameter(layer, "w2_weight", w2.transpose(1, 2).contiguous())
            n_restored += 1
        del w13, w2
    return n_restored


# ---- per-expert direct-apply (bypass model.load_weights fused-expert loader) ----
# CONFIRMED on the 4+2 NPU cluster (raysubmit_UMB8tQQKKLLLW1Qf, 2026-10-09):
# the steady delta wire carries PER-EXPERT MoE patches -- single-expert shapes
# like [2N, K] (Qwen3.5-35B-A3B: E_global=256, E_local=128 per rollout rank at
# gen_ep=2, N=512, K=2048, live w13 [E_local, 2N, K] = (128, 1024, 2048)) --
# under per-expert fused names ('...mlp.experts.{gid}.gate_up_proj' [2N, K] /
# '...mlp.experts.{gid}.down_proj' [K, N], global expert ids from the
# trainer-side re-key, commit 07533d4f). Layout reading verified by the
# observed crash arithmetic: the 2D (1024, 2048) fused patch made qwen3_5's
# fused loader chunk(2, dim=-2) -> (512, 2048) w1 half, then
# loaded_weight[expert_id] sliced a single (2048,) row toward the (512, 2048)
# w1 slot -- hence dest (512, 2048) != src (2048,).
#
# CONFIRMED on the 4+2 NPU cluster (raysubmit_GDrzYzFhu4B7rATx, 2026-10-10):
# the module lookup MUST remap the HF wire prefix to the vLLM module tree.
# The wire name 'model.language_model.layers.0.mlp.experts.13.gate_up_proj'
# parses to layer_prefix 'model.language_model.layers.0.mlp.experts', but the
# live module qualname is 'language_model.model.layers.0.mlp.experts' (vllm
# qwen3_5.py nests language_model.model.layers while HF nests
# model.language_model.layers). With the exact-qualname lookup every
# per-expert patch returned False, fell to the model loader, and
# qwen3_5.py:420 silently skipped all of them ("Parameter
# layers.N.mlp.experts.{gid}.down_proj not found in params_dict, skip
# loading" -- 808 logged + 51713 dedup-collapsed across the cluster), so the
# expert weights were never updated while the sync reported success.
#
# qwen3_5.load_weights' fused branch accepts ONLY whole-layer 3D [E, 2N, K]
# tensors under the id-less fused name 'experts.gate_up_proj' (it does
# chunk(2, dim=-2) then loops `loaded_weight[expert_id]`). A 2D per-expert
# patch that reaches it is mis-sliced into a 1D row -> copy_non_nan_ shape
# mismatch (dest (512, 2048) != src (2048,)). This path parses the expert id
# from the name and writes the patch directly into the checkpoint-layout live
# parameters (the sparse window guarantees that layout at apply time),
# bypassing the fused loader entirely. TP sharding is derived from the live
# parameter shape vs the patch shape, so no vllm_config plumbing is needed.
#
# An id-less FUSED name ('...experts.gate_up_proj', no expert id) with a 2D
# shape is a sender contract violation: the expert id (and the gate/up half)
# is not encoded anywhere on the wire, so NO receiver can apply it. The
# receiver fails loud on that case (see _decode_and_apply) instead of
# mis-slicing inside the fused loader.

_EXPERT_PATCH_RE = None


def _expert_patch_name_regex():
    global _EXPERT_PATCH_RE
    if _EXPERT_PATCH_RE is None:
        import re

        _EXPERT_PATCH_RE = re.compile(
            r"^(?P<layer>.+\.experts)\.(?P<eid>\d+)\."
            r"(?P<proj>gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$"
        )
    return _EXPERT_PATCH_RE


def parse_expert_patch_name(name: str):
    """Return ``(layer_prefix, expert_id, proj)`` for a per-expert HF checkpoint
    name, or ``None`` if the name is not a per-expert MoE checkpoint name."""
    m = _expert_patch_name_regex().match(name)
    if m is None:
        return None
    return m.group("layer"), int(m.group("eid")), m.group("proj")


def build_expert_module_index(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """One-walk qualname index of the fused-expert modules for per-patch lookup.

    Keys are every fused-expert module's qualname plus each of its ancestor
    prefixes, so the per-patch lookup condition ``qualname == candidate or
    qualname.startswith(candidate + '.')`` becomes an O(1) dict hit;
    first-module-wins matches the named_modules() scan order (setdefault
    registers in walk order). Round 5 re-ran the full module-tree walk for
    EVERY per-expert patch -- 16k+ patches per flush x ~1-2k modules of Python
    string matching dominated the receiver-side apply (45-48s per flush,
    raysubmit_vm8W34zfzVDfh3aK). Build per flush, not per process: the walk
    costs milliseconds, and rebuilding keeps the index immune to cross-session
    module rebinding (sleep-mode wake-up paths).
    """
    index: dict[str, torch.nn.Module] = {}
    for qualname, module in model.named_modules():
        if getattr(module, "w13_weight", None) is not None or getattr(module, "w13_weight_list", None) is not None:
            parts = qualname.split(".")
            for i in range(len(parts)):
                index.setdefault(".".join(parts[: i + 1]), module)
    return index


def apply_expert_patch_direct(
    model: torch.nn.Module,
    name: str,
    shape,
    dtype,
    values,
    indices,
    module_index: dict | None = None,
) -> bool:
    """Apply one sparse per-expert patch directly into the checkpoint-layout live
    parameter, bypassing ``model.load_weights``. Returns ``True`` if the patch
    was applied here (or intentionally skipped on this rank), ``False`` if the
    caller should fall back to the normal loader path: non-per-expert names,
    or per-expert names on a model with no fused-expert modules (ordinary
    per-expert parameters the loader applies). Raises on a fused-expert model
    when no module matches the (remapped) layer prefix -- the loader would
    silently skip the patch there.

    ``module_index`` (from :func:`build_expert_module_index`, built once per
    flush by the caller) replaces the per-patch full ``named_modules()`` walk
    with O(1) dict lookups; ``None`` keeps the legacy walk (tests, direct
    callers). Both paths are equivalent: the index keys cover exactly the
    walk's ``qualname == cand or qualname.startswith(cand + '.')`` condition.

    The sparse window guarantees the live parameters sit at checkpoint layout
    at apply time: ``w13_weight`` = [E, 2N, K], ``w2_weight`` = [E, K, N]
    (TP-narrowed variants included). The patch carries per-expert checkpoint
    coordinates: ``shape`` = [N, K] (gate_proj/up_proj), [2N, K]
    (gate_up_proj, gate rows first) or [K, N] (down_proj), ``indices`` = int32
    flat positions into ``shape``, ``values`` = the changed elements. We
    densify into a NaN-masked tensor, TP-narrow it to this rank's slice, then
    NaN-masked copy into the per-expert slot of ``w13_weight``/``w2_weight``.
    """
    parsed = parse_expert_patch_name(name)
    if parsed is None:
        return False
    layer_prefix, expert_id, proj = parsed

    # Find the fused-expert module matching this layer_prefix. ``layer_prefix``
    # comes from the HF checkpoint name on the wire, e.g.
    # "model.language_model.layers.0.mlp.experts", while the vLLM module tree
    # for the multimodal wrapper nests the same module as
    # "language_model.model.layers.0.mlp.experts" (confirmed 2026-10-10: the
    # exact-qualname-only lookup below returned None for every per-expert
    # patch, all 9145 v=1 patches fell through to the model loader, and
    # qwen3_5.py:420 silently skipped every one of them -- "not found in
    # params_dict, skip loading"). Match the wire prefix plus its HF<->vLLM
    # remaps; fail loud when nothing matches, because per-expert names have
    # no other apply path.
    candidates = list(_layer_prefix_candidates(layer_prefix))
    target_module = None
    if module_index is not None:
        for cand in candidates:
            target_module = module_index.get(cand)
            if target_module is not None:
                break
    else:
        for mod_qualname, module in model.named_modules():
            if any(mod_qualname == cand or mod_qualname.startswith(cand + ".") for cand in candidates):
                has_expert_weights = (
                    getattr(module, "w13_weight", None) is not None
                    or getattr(module, "w13_weight_list", None) is not None
                )
                if has_expert_weights:
                    target_module = module
                    break
    if target_module is None:
        # A model with NO fused-expert modules holds per-expert weights as
        # ordinary parameters, so the model loader is the legitimate apply
        # path for per-expert names. But on a fused-expert model (qwen3_5:
        # params_dict carries w13_weight/w2_weight, never per-expert names)
        # an unmatched prefix means the loader will silently skip the patch
        # -- fail loud instead (confirmed 2026-10-10: 808 logged + 51713
        # dedup-collapsed "not found in params_dict, skip loading" skips
        # left every expert weight stale while the sync reported success).
        if next(iter_expert_weight_modules(model), None) is None:
            return False
        raise NotImplementedError(
            f"apply_expert_patch_direct({name!r}): no fused-expert module matches the layer prefix "
            f"(tried {candidates}) although the model carries fused-expert modules. Per-expert patches "
            "cannot fall back to the model loader on a fused-expert model -- vllm's fused-expert loader "
            "only accepts whole-layer 3D tensors under the id-less fused name and silently skips "
            "per-expert names ('not found in params_dict, skip loading')."
        )
    w13 = getattr(target_module, "w13_weight", None)
    w2 = getattr(target_module, "w2_weight", None)
    if w13 is None or w2 is None:
        raise NotImplementedError(
            f"apply_expert_patch_direct({name!r}): matched fused-expert module "
            f"{type(target_module).__name__} exposes no Parameter-typed w13_weight/w2_weight (per-expert "
            "weight-list variant?); the model loader cannot apply per-expert names on a fused-expert model."
        )

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
    # Densify: full checkpoint tensor, NaN at unchanged positions.
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

    # Layout guards: the write below assumes checkpoint layout. If the live
    # param still matches the RUNTIME signature (npu_vllm_patch: dim-1 ==
    # hidden for w13, dim-2 == hidden for w2), the window did not normalize it
    # -- fail loud instead of writing transposed garbage.
    hidden = _module_hidden_size(target_module)
    if w13.shape[2] != hidden:
        raise NotImplementedError(
            f"apply_expert_patch_direct({name}): live w13 {tuple(w13.shape)} is not at checkpoint layout "
            f"[E, 2N, K] (dim-2 should be hidden_size={hidden}); refusing to write transposed coordinates."
        )
    if w2.shape[1] != hidden:
        raise NotImplementedError(
            f"apply_expert_patch_direct({name}): live w2 {tuple(w2.shape)} is not at checkpoint layout "
            f"[E, K, N] (dim-1 should be hidden_size={hidden}); refusing to write transposed coordinates."
        )
    n_experts_local = w13.shape[0]
    if not 0 <= expert_id < n_experts_local:
        raise NotImplementedError(
            f"apply_expert_patch_direct({name}): expert_id {expert_id} out of range for the local expert "
            f"stack ({n_experts_local} experts); expert_map_manager is "
            f"{'absent' if expert_map_manager is None else 'inconsistent with the live parameter'}."
        )

    # w13 checkpoint layout [E, 2N, K]: gate (w1) -> [e, :N/TP, :], up (w3) ->
    # [e, N/TP:, :]; w2 checkpoint layout [E, K, N]: down -> [e, :, :]. The
    # live parameter is already TP-sharded, so select the per-expert slot FIRST
    # and then narrow the densified checkpoint tensor to that slot's shape.
    if proj in ("gate_proj", "up_proj", "gate_up_proj"):
        slot_full_n = w13.shape[1] // 2
        if proj == "gate_proj":
            writes = [(checkpoint_tensor, w13.data[expert_id, :slot_full_n, :])]
        elif proj == "up_proj":
            writes = [(checkpoint_tensor, w13.data[expert_id, slot_full_n:, :])]
        else:  # gate_up_proj [2N, K]: gate rows first, then up rows
            half = checkpoint_tensor.shape[0] // 2
            writes = [
                (checkpoint_tensor[:half], w13.data[expert_id, :slot_full_n, :]),
                (checkpoint_tensor[half:], w13.data[expert_id, slot_full_n:, :]),
            ]
    else:  # down_proj
        writes = [(checkpoint_tensor, w2.data[expert_id])]

    # Narrow each densified checkpoint tensor to the live slot's per-rank shape.
    # checkpoint_tensor dims (before TP): gate/up [N_full, K_full], gate_up
    # [2N_full, K_full], down [K_full, N_full]. dest dims (after TP): the same
    # with the sharded dims divided. Each dim where the source is larger than
    # the dest is a TP-sharded dim; pick this rank's slice.
    tp_rank = getattr(target_module, "tp_rank", None)
    if not isinstance(tp_rank, int):
        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()

    for src, dest in writes:
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
