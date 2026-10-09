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
"""Megatron-side delta export machinery, built on Megatron-Bridge param mappings.

The delta engine consumes final HF-coordinate entries; everything mcore-specific
lives here: enumerating parameters through :meth:`AutoBridge.get_conversion_tasks`,
routing each parameter's entries to its wire merge group, and probing the
bridge's own ``megatron_to_hf`` converters with NaN sentinels to translate a
shard-local delta into HF coordinates.

The probe never runs a collective, yet executes the mapping's REAL parallel
code paths: a "group" carries two separable meanings -- its SIZE/RANK (which
value math like Mamba's ``local_dim = global // tp_size`` consumes) and its
COMMUNICATION. The probe copies keep the first faithful and replace only the
second: groups become :class:`_ProbeGroup` (real size/rank, any actual use for
communication raises), and the bridge's comm helpers are stubbed with local
synthesis -- ``gather_from_tp_ranks`` returns this rank's shard at its true
rank index with NaN placeholders for every other rank (their contributions are
exported by those ranks' own probes). Feeding ``megatron_to_hf`` the LOCAL
shard as a NaN buffer with the rank's own delta scattered in yields HF tensors
whose non-NaN survivors are exactly this rank's contributions in final HF
coordinates.

Scope (asserted in the exporter): TP + EP + PP/VPP, no LoRA. Under PP the
bridge's conversion tasks already enumerate the GLOBAL parameter directory
(identical order on every rank, tied embeddings deduped, placeholders for
other stages); non-owner ranks ship zero-count lockstep rows and every
param merges over the WORLD group, so the wire master needs no relay.

Safety boundary: communication must stay confined to the four stubbed helpers
(``gather_from_tp_ranks`` / ``gather_from_ep_ranks[_scale]`` / the PP
broadcasts, whose ``pp_size == 1`` fast path the probe keeps) -- anything else
touching a probe group raises instead of skewing silently. The remaining
assumption is that transforms REARRANGE elements rather than arithmetically
BLEND chunks (blending would eat the NaN sentinels); the TP>1 differential
test (real ``megatron_to_hf`` vs probe assembly, bitwise) is the regression
oracle for both assumptions on every Megatron-Bridge upgrade.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch

from verl.workers.engine.spec import ShardSpec

logger = logging.getLogger(__name__)

# Id-less fused MoE HF names (e.g. qwen3_5's fused checkpoint format
# '...mlp.experts.gate_up_proj'): legitimate for whole-layer 3D slots on the
# dense seed, but a 2D per-expert slot under one carries no expert id on the
# wire. Deliberate twin of delta_weight_transfer._FUSED_EXPERT_NAME_RE (the
# engine layer must not import from the rollout layer; keep the two in sync).
_FUSED_EXPERT_NAME_RE = re.compile(r"\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$")

# Megatron per-expert MoE parameter suffix ('...mlp.experts.linear_fc1.weight0'
# .. '.weight15'): the digit is the EP-LOCAL expert index (GroupedMLP-style
# per-expert flattened weights). Used to recover the GLOBAL expert id that
# fused-name bridge mappings (qwen3_5) leave off the HF name.
_MCORE_EXPERT_IDX_RE = re.compile(r"\.weight(\d+)$")


class _ProbeGroup:
    """Size/rank-faithful stand-in for a process group on probe copies: value
    math inside ``megatron_to_hf`` (e.g. Mamba's per-rank de-interleave dims)
    sees the REAL parallel sizes, while any attempt to actually communicate
    through the group fails loud (the probe stubs the bridge's comm helpers;
    anything else reaching a group is an unstubbed communication pattern)."""

    def __init__(self, size: int, rank: int):
        self._size = int(size)
        self._rank = int(rank)

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def __getattr__(self, name):
        raise RuntimeError(
            f"probe process group asked for {name!r}: this mapping communicates outside "
            "the stubbed helpers (gather_from_tp_ranks / gather_from_ep_ranks[_scale] / "
            "pp broadcasts) -- extend make_probe's comm stubs before trusting its export"
        )


_NAN_POOL: dict = {}


def _nan_block(shape, dtype, device) -> torch.Tensor:
    """Read-only all-NaN placeholder for another rank's gather chunk, pooled by
    (shape, dtype, device): the gather stub hands these out on every probe call
    for every param, and they are only ever READ (the transforms are functional),
    so one block per distinct shape serves the whole model for the run's
    lifetime instead of a fresh cudaMalloc+fill per param per sync."""
    key = (tuple(shape), dtype, str(device))
    t = _NAN_POOL.get(key)
    if t is None:
        t = torch.full(tuple(shape), float("nan"), dtype=dtype, device=device)
        _NAN_POOL[key] = t
    return t


def _warm_lazy_mappings(mapping, module) -> None:
    """Force AutoMapping's lazy concrete delegate into existence: AutoMapping
    resolves its Column/Row/Replicated delegate on first use, snapshotting
    whatever process groups exist at that moment -- if that first use happened
    inside the probe, the delegate would be born with the REAL groups and
    gather for real (the qkv-bias double-size bug). Self-only on purpose:
    ``_inject`` warms every child right before copying it, so each node of the
    mapping tree is warmed exactly once."""
    if hasattr(mapping, "_detect_parallelism_type") and getattr(mapping, "_mapping", None) is None:
        try:
            t = mapping._detect_parallelism_type(module)
            mapping._mapping = mapping._get_or_create_mapping(t)
            mapping._detected_type = t
        except Exception as e:  # pragma: no cover - defensive; probe falls back to real groups
            logger.warning("could not warm lazy mapping %s: %s", type(mapping).__name__, e)


def make_probe(mapping, module):
    """Copy a Megatron-Bridge param mapping tree and turn ``megatron_to_hf``
    into a communication-free LOCAL transform that still runs the REAL
    (tp_size > 1) code paths: every copy's groups become size-faithful
    :class:`_ProbeGroup` stand-ins and the bridge's comm helpers are stubbed
    with local synthesis. The copy is recursive (composite mappings delegate to
    inner mappings -- QKVMapping._tp_mapping is an AutoMapping which itself
    delegates to a lazily-created concrete mapping; none of these receive the
    outer stubbing on their own)."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.core.utils import get_pg_rank, get_pg_size

    _warm_lazy_mappings(mapping, module)

    def _stub(c):
        # size-faithful groups, sizes/ranks read from the ORIGINAL (real) groups
        # BEFORE replacement. The probe only ever runs on the stage that OWNS
        # the param, where a pp broadcast is the identity -- a (1, 0) group
        # keeps the pp helpers on that fast path regardless of the real pp size.
        c.pp_group = _ProbeGroup(1, 0)
        for attr in ("ep_group", "_tp_group", "_etp_group"):
            g = getattr(c, attr, None)
            setattr(c, attr, _ProbeGroup(get_pg_size(g), get_pg_rank(g)))

        def _gather_tp(tensor, _c=c):
            # what a real all_gather over the tp group would produce, minus the
            # other ranks' data: our shard rides at its true rank index, every
            # other chunk is a pooled read-only NaN block (those ranks export
            # their own contributions; transforms never write into their inputs,
            # which the TP>1 bitwise differential revalidates).
            nan = _nan_block(tensor.shape, tensor.dtype, tensor.device)
            out = [nan] * _c.tp_size
            out[_c.tp_rank] = tensor
            return out

        c.gather_from_tp_ranks = _gather_tp
        # each rank's probe emits only its own local experts, under the hf name
        # bound with the GLOBAL expert id at task construction; the engine
        # merges entries over the etp x ep group, so the ep fan-out reduces to
        # the bound name.
        c.gather_from_ep_ranks = lambda w, mod, name: {str(name): w}
        # mirrors the real helper's tail (unsqueeze(0) ... squeeze().unsqueeze(-1))
        c.gather_from_ep_ranks_scale = lambda w, mod, name: {str(name): w.unsqueeze(0).squeeze().unsqueeze(-1)}
        return c

    def _inject(m):
        c = _stub(copy.copy(m))
        for attr, value in list(vars(c).items()):
            if isinstance(value, MegatronParamMapping):
                # warm BEFORE copying the child: the lazy delegate must exist so
                # the child copy's vars() include it and the recursion reaches it.
                _warm_lazy_mappings(value, module)
                setattr(c, attr, _inject(value))
        return c

    return _inject(mapping)


@dataclass
class McoreParamExport:
    """One mcore parameter's export record: geometry + probe + module handle.
    ``param is None`` marks a parameter owned by another pipeline stage: the
    rank ships a zero-count lockstep row for it (probe/module unused)."""

    megatron_name: str
    param: Optional[torch.Tensor]
    spec: ShardSpec
    probe: Any  # comm-stubbed mapping copy (local megatron_to_hf evaluator)
    module: Any  # module handle megatron_to_hf reads config from
    # GLOBAL slot table for this directory row (set by the index-build
    # exchange): identical on every rank of the row's merge group.
    slots: Optional[list] = None
    # Whether the bridge mapping marks this parameter as an expert weight
    # (drives the sparse-delta expert-name contract check at index build).
    is_expert: bool = False
    # GLOBAL expert id for per-expert rows whose bridge mapping emits the
    # id-less fused HF name (qwen3_5's FusedGated/FusedExpertMapping): bound
    # at index build from the megatron param's EP-local 'weight{i}' suffix
    # (see _bind_expert_global_ids); None leaves the probe's names untouched.
    expert_gid: Optional[int] = None


def _bind_expert_global_ids(index: list[McoreParamExport], ep_rank: int) -> None:
    """Resolve GLOBAL expert ids for per-expert rows whose bridge mapping emits
    id-less fused HF names (qwen3_5's FusedGatedExpertMapping / FusedExpertMapping:
    hf_param '...experts.gate_up_proj' / '...experts.down_proj').

    The probe contract (see make_probe) is that each rank emits its own local
    experts under the hf name bound with the GLOBAL expert id -- per-expert
    mappings (glm4.5 and the standard MoE zoo) satisfy it at task construction.
    The fused mappings do not: ``megatron_to_hf`` on one 'linear_fc1.weight{i}'
    shard returns {fused_name: [2N, K]} with correct values/orientation but no
    expert id, so every local expert would collapse into the same slot at the
    etp x ep merge (observed: raysubmit_sHZUTh4setzRJ5GP, guard fired on
    'language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0'). Recover
    the id from the megatron param name: '...weight{i}' carries the EP-LOCAL
    index, and the global id follows the contiguous EP block layout shared by
    the bridge's own rank-major fused export (the dense seed's [E, 2N, K]
    assembly) and verl's veomni expert stack (veomni/utils.py:
    ``ep_rank * n_local_experts``): gid = ep_rank * n_local + i.

    A suffix set matching neither the local run {0..n-1} nor this rank's
    contiguous global block fails loud -- silently routing experts to wrong
    ids is the one failure this must never produce.
    """
    groups: dict[str, list[tuple[int, McoreParamExport]]] = {}
    for rec in index:
        if rec.param is None or not rec.is_expert:
            continue
        m = _MCORE_EXPERT_IDX_RE.search(rec.megatron_name)
        if m is None:
            continue
        groups.setdefault(rec.megatron_name[: m.start()], []).append((int(m.group(1)), rec))
    if not groups:
        return
    n_rows = 0
    for base in sorted(groups):
        items = groups[base]
        suffixes = sorted(suffix for suffix, _ in items)
        n = len(suffixes)
        if suffixes == list(range(n)):
            offset = ep_rank * n
        elif suffixes == list(range(ep_rank * n, ep_rank * n + n)):
            # already-global suffixes on this rank; rank 0 is indistinguishable
            # between the two conventions and needs no offset either way.
            offset = 0
        else:
            raise NotImplementedError(
                f"{base}: cannot derive global expert ids from per-expert suffixes {suffixes} "
                f"(ep_rank={ep_rank}): expected the EP-local run 0..{n - 1} or this rank's "
                f"contiguous global block {ep_rank * n}..{ep_rank * n + n - 1}. Sparse expert "
                "delta export refuses to guess the expert routing."
            )
        for suffix, rec in items:
            rec.expert_gid = offset + suffix
        n_rows += n
    logger.warning(
        "delta export: bound global expert ids on %d per-expert row(s) across %d group(s) (ep_rank=%d); "
        "id-less fused probe outputs will be re-keyed to '...experts.{gid}.<proj>'",
        n_rows,
        len(groups),
        ep_rank,
    )


def build_export_index(bridge, megatron_model, slot_cache: dict | None = None) -> list[McoreParamExport]:
    """Enumerate every mcore parameter through the bridge's conversion tasks
    and precompute its probe + wire routing.

    No shard geometry is hand-computed here: the comm-stubbed probe emits final
    HF coordinates straight from the local shard, so the engine only needs to
    know WHICH group's entries to merge per parameter (the spec's
    ``gather_group``; ``full_shape`` is the nominal local shape) and which
    ranks contribute. The index is built once (parameter sets are static) and
    reused by both the shard export and the delta entry hook.

    Lockstep: ``get_conversion_tasks`` already enumerates the GLOBAL parameter
    list in an order identical on every rank (the bridge allgathers and sorts
    names across pp ranks, dedupes tied embeddings, and leaves
    ``param_weight=None`` placeholders for parameters owned by other pipeline
    stages). Under PP>1 those placeholders become zero-count lockstep rows and
    every param's entries merge over the WORLD group (owner-stage ranks
    contribute, dp/cp replicas and other stages stay empty); their slot tables
    are pre-seeded once via :func:`_preseed_slot_tables`. Under PP=1 the
    routing is unchanged (tp / etp x ep subgroups, replicated rank-0 direct).
    """
    from megatron.core import parallel_state as mpu

    pp_world = mpu.get_pipeline_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    world = torch.distributed.group.WORLD

    tasks = bridge.get_conversion_tasks(megatron_model)
    index: list[McoreParamExport] = []
    for task in tasks:
        mapping = task.mapping
        param = task.param_weight
        name = task.global_param_name
        if param is None:
            if pp_world == 1:
                # not a pp placeholder (e.g. a skipped mapping); nothing to ship
                continue
            # owned by another pipeline stage: zero-count lockstep row -- the
            # rank walks the same global directory and contributes nothing.
            index.append(
                McoreParamExport(
                    megatron_name=name,
                    param=None,
                    spec=ShardSpec(full_shape=(0,), place=0, gather_group=world, contributes=False),
                    probe=None,
                    module=None,
                    is_expert=bool(getattr(mapping, "is_expert", False)),
                )
            )
            continue
        module = task.megatron_module
        local_shape = tuple(int(x) for x in param.shape)

        is_expert = mapping.is_expert and (ep_size > 1 or tp_world > 1)
        is_tp_sharded = getattr(param, "tensor_model_parallel", False) and tp_world > 1

        if pp_world > 1:
            # single WORLD merge group for every param: the wire master (global
            # rank 0) sits in every gather regardless of which stage owns the
            # param, so no relay hop is needed. Owner-stage ranks contribute
            # their shard pieces; dp/cp replicas dedupe via ``contributes``
            # (identical copies -- exactly one replica set ships).
            if is_expert:
                contributes = mpu.get_expert_data_parallel_rank() == 0
            elif is_tp_sharded:
                contributes = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            else:
                contributes = (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                )
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=world, contributes=contributes)
        elif is_expert:
            # every rank holding a piece of this expert set contributes; the
            # engine merges their probe entries over the joint etp x ep group.
            spec = ShardSpec(
                full_shape=local_shape,
                place=0,
                gather_group=mpu.get_expert_tensor_and_model_parallel_group(),
            )
        elif is_tp_sharded:
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=tp_group)
        else:
            # replicated: engine's pg=None path (rank 0 consumes its own entry
            # directly, replicas stay in lockstep via zero counts).
            spec = ShardSpec(full_shape=local_shape)

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                spec=spec,
                probe=make_probe(mapping, module),
                module=module,
                is_expert=bool(getattr(mapping, "is_expert", False)),
            )
        )

    # Bind expert global ids BEFORE the slot-table exchange: the exchange
    # probes every owned row once, and the re-key in mcore_hf_delta_entry
    # must shape the slot names the tables (and the steady entries) carry.
    _bind_expert_global_ids(index, mpu.get_expert_model_parallel_rank())
    _exchange_slot_tables(index, slot_cache)
    # a row with an empty union has no owner on ANY rank: the bridge skipped
    # it everywhere (no mapping / missing HF key), so it is outside the
    # bridge's own export scope -- exactly what PP=1 does by skipping
    # param_weight=None tasks. Drop it (all ranks agree: the exchange result
    # is identical) instead of shipping unsized rows.
    unowned = [rec.megatron_name for rec in index if rec.slots is None]
    if unowned:
        logger.warning(
            "delta export drops %d row(s) the bridge skipped on every rank "
            "(no mapping or missing HF key; out of export scope): %s",
            len(unowned),
            unowned[:8],
        )
        index = [rec for rec in index if rec.slots is not None]

    # [FAIL-LOUD 2026-10-09] Sparse-delta expert-name contract check. An
    # expert row whose probe emits an id-less fused HF name (qwen3_5's fused
    # checkpoint format, '...mlp.experts.gate_up_proj') with a 2D per-expert
    # shape cannot carry the expert id on the wire. The dense seed tolerates
    # such names -- the bridge's real export stacks whole-layer 3D fused
    # tensors and the rollout's fused loader slices them -- but the steady
    # delta collapses every local expert into the same slot: the probe stub's
    # dict keying keeps one output per name, so all but one expert's delta is
    # silently dropped, and the receiver's fused loader mis-slices the 2D
    # survivor (observed on the 4+2 NPU cluster, raysubmit_UMB8tQQKKLLLW1Qf:
    # wire patch '...experts.gate_up_proj' shape (1024, 2048) = [2N, K],
    # N=512, crash dest (512, 2048) != src (2048,)). Rows whose megatron name
    # carries a parseable per-expert suffix ('...weight{i}') are now re-keyed
    # to per-expert HF names with the GLOBAL expert id before this check
    # (see _bind_expert_global_ids); what remains here is the residual class
    # whose expert id cannot be recovered from the megatron name. Whole-layer
    # 3D fused slots and per-expert names that carry the global expert id
    # pass untouched.
    for rec in index:
        if not rec.is_expert or rec.slots is None:
            continue
        for slot_name, slot_shape in rec.slots:
            if len(slot_shape) == 2 and _FUSED_EXPERT_NAME_RE.search(slot_name):
                raise NotImplementedError(
                    f"{rec.megatron_name}: the bridge's MoE conversion emits the id-less fused HF name "
                    f"{slot_name!r} with a per-expert 2D shape {tuple(slot_shape)}. Sparse delta export requires "
                    "per-expert HF names carrying the GLOBAL expert id ('<prefix>.experts.{global_id}.gate_up_proj' "
                    "[2N, K], '.gate_proj'/'.up_proj' [N, K], or '.down_proj' [K, N]) -- without it the expert id "
                    "is not on the wire, every local expert collapses into one slot, and the deltas are silently "
                    "dropped or mis-applied. Rows with a parseable '...weight{i}' megatron suffix are re-keyed "
                    "(see _bind_expert_global_ids); this one is not, so its expert id cannot be recovered. Fix "
                    "the model's Megatron-Bridge mapping to bind per-expert names for the delta path; the dense "
                    "seed's whole-layer 3D fused tensors are unaffected."
                )
    return index


def _exchange_slot_tables(index: list[McoreParamExport], slot_cache: dict) -> None:
    """One-time GLOBAL slot-table exchange, run at every world size, keyed by
    DIRECTORY ROW (not by param name).

    The engine's batched gather merges entries BY SLOT POSITION within each
    directory row and rank 0 names the merged pieces from ITS OWN list, so
    the per-row list must be identical on every rank of the row's merge
    group. Name-keyed merging is not enough on mcore: expert param NAMES
    embed the expert ids (``...experts.linear_fc1.weight0`` on ep rank 0 vs
    ``weight1`` on ep rank 1), so the same row carries different names per
    rank while still gathering together. Each rank therefore probes a zero
    delta through its OWNED rows to reveal its local ``(hf_name, hf_shape)``
    list, one ``all_gather_object`` over WORLD exchanges the ordered row
    lists, and every row's table becomes the rank-order first-seen union --
    identical everywhere by construction. Dense/TP rows (same list on every
    rank) dedup to themselves; expert rows concatenate the per-ep-rank
    lists; PP placeholder rows inherit the owners' union."""
    local_rows: list = []
    for rec in index:
        if rec.param is None:
            local_rows.append(None)
            continue
        if rec.megatron_name not in slot_cache:
            empty_idx = torch.empty(0, dtype=torch.int64, device=rec.param.device)
            empty_val = torch.empty(0, dtype=torch.bfloat16, device=rec.param.device)
            mcore_hf_delta_entry(rec, 0, empty_idx, empty_val, slot_cache)
        local_rows.append(slot_cache[rec.megatron_name])

    world = torch.distributed.get_world_size()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, local_rows)
    n_rows = len(local_rows)
    assert all(len(rows) == n_rows for rows in gathered), (
        f"directory row counts diverge across ranks: {[len(r) for r in gathered]} -- "
        "the bridge's global enumeration is expected to be structurally parallel"
    )
    for k, rec in enumerate(index):
        union: dict = {}
        for rows in gathered:  # rank order -> identical result on every rank
            row = rows[k]
            if row is None:
                continue
            for slot in row:
                union[(slot[0], tuple(slot[1]))] = None  # ordered-set semantics
        rec.slots = [(n, tuple(shape)) for (n, shape) in union.keys()] if union else None


def mcore_hf_delta_entry(rec: McoreParamExport, _place, lidx: torch.Tensor, lval: torch.Tensor, slot_cache: dict):
    """Probe one mcore param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the LOCAL shard shape (exactly what
    the real ``megatron_to_hf`` receives), runs the comm-stubbed probe -- real
    group sizes, gathers synthesized locally, so the mapping executes its real
    TP>1 code paths -- and extracts each output slot's surviving positions.
    The slot list is cached after the first call (the converter's output names
    are deterministic, so every rank's cache agrees and the batched gather
    stays aligned)."""
    dtype_str = str(lval.dtype).replace("torch.", "")
    assert lval.numel() == 0 or lval.is_floating_point(), (
        f"{rec.megatron_name}: NaN sentinels require a floating-point param, got {lval.dtype}"
    )

    cached = rec.slots if rec.slots is not None else slot_cache.get(rec.megatron_name)
    if rec.param is None:
        # owned by another pipeline stage: pure lockstep row. The slot table
        # was pre-seeded from the owner stage (fail loud if not -- an
        # unsized entry would silently misalign the batched gather).
        assert cached is not None, f"{rec.megatron_name}: slot table not pre-seeded for a non-owned PP param"
        assert lidx.numel() == 0, f"{rec.megatron_name}: delta reported for a param this rank does not own"
    if lidx.numel() == 0 and cached is not None:
        # empty delta: the slot table froze after the first probe, so the
        # zero-count lockstep entry needs no probe run at all -- skip the
        # buffer build, the transform and the full-output NaN scan.
        return (
            cached,
            dtype_str,
            torch.zeros(len(cached), dtype=torch.int64),
            torch.empty(0, dtype=torch.int32, device=lval.device),
            torch.empty(0, dtype=lval.dtype, device=lval.device),
        )

    buf = torch.full(tuple(rec.param.shape), float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        buf.view(-1)[lidx] = lval

    outs = rec.probe.megatron_to_hf(buf, rec.module)

    if rec.expert_gid is not None:
        # Fused-name expert row (qwen3_5's FusedGatedExpertMapping): re-key the
        # id-less fused output to the per-expert HF name carrying the GLOBAL
        # expert id -- the naming contract every other expert mapping satisfies
        # at task construction and the etp x ep slot merge relies on. Values
        # and orientation are already final HF coordinates (observed shard:
        # [2N, K] gate-rows-first for gate_up_proj, [K, N] for down_proj);
        # only the dict key was wrong. The re-keyed slot must be 2D per-expert
        # checkpoint layout -- anything else fails loud rather than mis-routing.
        gid = rec.expert_gid
        rekeyed: dict = {}
        for hf_name, hf_tensor in outs.items():
            m = _FUSED_EXPERT_NAME_RE.search(hf_name)
            if m is None:
                rekeyed[hf_name] = hf_tensor
                continue
            if hf_tensor.dim() != 2:
                raise NotImplementedError(
                    f"{rec.megatron_name}: fused expert slot {hf_name!r} emitted a {hf_tensor.dim()}D tensor "
                    f"{tuple(hf_tensor.shape)}; the per-expert re-key contract is 2D checkpoint layout "
                    "([2N, K] gate_up_proj / [N, K] gate|up_proj / [K, N] down_proj)."
                )
            rekeyed[f"{hf_name[: m.start()]}.experts.{gid}.{m.group(1)}"] = hf_tensor
        outs = rekeyed

    key = rec.megatron_name
    slots = rec.slots if rec.slots is not None else slot_cache.get(key)
    if slots is None:
        # only reachable from the index-build exchange itself, which probes a
        # zero delta to reveal this rank's local list before the row unions
        # are installed on the recs; steady entries always find rec.slots.
        slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
        slot_cache[key] = slots
    unknown = set(outs) - {n for n, _ in slots}
    assert not unknown, (
        f"{key}: probe emitted slots missing from the global table {sorted(unknown)[:4]} -- "
        "non-deterministic converter naming would misalign the batched gather"
    )
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, _sshape) in enumerate(slots):
        out = outs.get(sname)
        if out is None:
            continue  # another rank's slot (e.g. its ep-local experts): zero count here
        fl = out.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            idx_pieces.append(p_.to(torch.int32))
            val_pieces.append(fl[p_])
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val
