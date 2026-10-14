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
"""Sharded delta: diff each rank's local FSDP shard, gather only the changes to rank 0.

The default delta path all-gathers the full parameter (``DTensor.full_tensor()``) and
byte-diffs it against a full-model pinned-CPU snapshot on rank 0. This module instead lets
each rank keep a pinned snapshot of only *its* shard, byte-diff the shard locally, and
gather just the changed ``(within-parameter position, value)`` pairs to rank 0 -- so the
all-gather volume drops to the sparsity ratio (~1-3%) and rank 0 no longer needs a
full-model snapshot. The gathered result is bit-identical to the full-tensor diff, so the
downstream encode + broadcast and the receiver are unchanged.

Scope: FSDP2 ``Shard(0)`` DTensors (the common case) + replicated / non-DTensor params.
Other shard dims are strided in the flattened layout and raise NotImplementedError.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

# [Round 10] payload transport mode. "a2a" (default) moves each rank's exact
# totals[rank] elements with all_to_all_single (no max_n padding); "gather"
# (VERL_DELTA_GATHER_MODE=gather) forces the legacy padded dist.gather. The
# choice is made once per group, identically on every rank (see _decide_a2a).
_A2A_MODE: dict = {}
_GATHER_MODE_ENV = os.environ.get("VERL_DELTA_GATHER_MODE", "a2a").strip().lower()


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-diff a local shard against its snapshot; return (global_positions, values).

    Positions are int64 indices into the *full flattened parameter* (offset + local index).
    Dtype-agnostic, bytewise (view-as-int), no arithmetic -- matches ``bytewise_diff_mask``.
    """
    es = local_new.element_size()
    int_dtype = _DTYPE_INT.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    return global_idx, values


def _pg_torch_device(group) -> torch.device:
    """The device a process group's collectives run on (NPU/CUDA for HCCL/NCCL
    groups, cpu for gloo). Private-API first, hardware fallbacks behind it."""
    fn = getattr(torch.distributed, "_get_pg_default_device", None)
    if fn is None:
        fn = getattr(torch.distributed.distributed_c10d, "_get_pg_default_device", None)
    if fn is not None:
        try:
            return fn(group)
        except Exception:
            pass
    if torch.npu.is_available():
        return torch.device("npu", torch.npu.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _decide_a2a(group, world: int) -> str:
    """Decide ONCE per group, identically on every rank, whether the payload
    move uses exact-size ``all_to_all_single`` ("a2a") or the legacy padded
    ``dist.gather`` ("gather").

    [Round 10] R9 proved the padded gathers already run at line rate (~175MB/s
    sustained, matching the 146MB/s one-shot probe) but move 5.5x the useful
    bytes: 8811MB of max_n padding vs 1956MB useful. The padding is
    STRUCTURAL, not a batching artifact: MoE slot ownership is 1-of-world per
    slot, so max_n / avg(totals) stays ~4.5 in every window and the sub-round
    cuts (identical on every rank) cannot change the ratio. all_to_all_single
    with exact per-pair sizes moves only the useful bytes and is the standard
    unequal-split collective on this stack (vllm-ascend's MoE token dispatch).
    The probe below exercises empty send AND empty receive splits; the MIN
    all_reduce keeps the choice identical across ranks -- a per-rank fallback
    would desync the collectives and deadlock. gloo (unit tests) does not
    implement all_to_all and downgrades everyone to "gather".
    """
    key = id(group)
    mode = _A2A_MODE.get(key)
    if mode is not None:
        return mode
    mode = "a2a" if _GATHER_MODE_ENV != "gather" else "gather"
    if mode == "a2a":
        dev = _pg_torch_device(group)
        ok = torch.ones(1, dtype=torch.int32, device=dev)
        try:
            r = dist.get_rank(group)
            # rank 0 sends nothing and receives 1 int from every other rank;
            # ranks > 0 send 1 int and receive nothing.
            n = 0 if r == 0 else 1
            send = torch.arange(n, dtype=torch.int32, device=dev)
            in_split = [0] * world
            in_split[0] = n
            out_split = [0] * world if r != 0 else [0] + [1] * (world - 1)
            recv = torch.empty(sum(out_split), dtype=torch.int32, device=dev)
            dist.all_to_all_single(recv, send, output_split_sizes=out_split, input_split_sizes=in_split, group=group)
        except Exception as e:  # noqa: BLE001 - downgrade, not crash: gather still works
            logger.warning("[GQ-A2A] all_to_all_single unusable (world=%d): %r; using padded gather", world, e)
            ok.zero_()
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if int(ok.item()) == 0:
            mode = "gather"
        if dist.get_rank(group) == 0:
            logger.info("[GQ-A2A] payload transport mode=%s (world=%d)", mode, world)
    _A2A_MODE[key] = mode
    return mode


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
    stats: dict | None = None,
) -> list | None:
    """Variable-length sparse gather, batched: one collective round for K parameters.

    Each rank passes its K per-parameter deltas concatenated (``idx_concat``,
    ``val_concat``) plus the per-parameter length vector ``counts`` ([K] int64).
    One all_gather exchanges the K x world count matrix; the payloads then move
    to rank 0 (exact-size all_to_all_single by default, padded gathers as the
    fallback). Rank 0 slices per (rank, param) and returns K ``(idx, val)`` pairs
    (None elsewhere) -- bit-identical to K individual gathers, ~K x fewer
    collectives and host syncs.

    [Round 7] Host-resident inputs (the batched backend export diffs on host):
    tensors are moved to the gather group's device for the collectives.
    [Round 8] Rank 0's per-slot pieces STAY on that device: they feed the
    wire-side bucket/assemble/broadcast, which is device-resident end to end.
    Round 7 brought them back to the host with one pageable ``.cpu()`` per
    slot -- ~2 x 17k synchronizing D2H copies per sync on rank 0 (observed
    raysubmit_wKmFfrWNiRq6wu4a: gather 116-135s, flat in the changed ratio),
    i.e. the per-param device-sync disease this module was built to avoid,
    reintroduced on the return path.
    [Round 9] Optional ``stats`` accumulator (threaded through sub-rounds) splits
    the round time: "counts" = the count-matrix all_gather (a 2KB-class world
    barrier whose time is arrival skew + launch, not wire) vs "payload" = the two
    padded gathers (wire + gather-implementation cost); "pad_bytes"/"useful_bytes"
    expose the max_n padding amplification. Round 8 (raysubmit_X97dLkpbq5HKdaD9)
    left gq_coll=81.9s for ~1.7GB of useful payload (~21MB/s aggregate, vs
    1.35GB/s for the dense broadcast on the same cluster) -- these counters
    tell skew from wire from padding without another blind fix round.
    [Round 10] the verdict was PADDING (8811MB pad vs 1956MB useful at a
    ~175MB/s line; counts rounds were ~0.01s): the payload move now defaults
    to exact-size all_to_all_single (see _decide_a2a) -- pad_bytes stays 0 in
    a2a mode, and stats["a2a_rounds"] counts the rounds that used it.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    host_input = idx_concat.device.type == "cpu" and group is not None
    if host_input:
        dev = _pg_torch_device(group)
        if dev.type != "cpu":
            idx_concat = idx_concat.to(dev)
            val_concat = val_concat.to(dev)
            counts = counts.to(dev)
    dev = idx_concat.device
    k = int(counts.numel())

    counts_all = [torch.zeros_like(counts) for _ in range(world)]
    _tcounts = time.monotonic()
    dist.all_gather(counts_all, counts.to(dev), group=group)
    if stats is not None:
        stats["counts"] += time.monotonic() - _tcounts
    counts_cpu = torch.stack(counts_all).cpu().tolist()  # one D2H sync instead of `world`

    if max_round_bytes is not None and k > 1:
        # Deterministic sub-rounds: every rank sees the same counts matrix, so all
        # ranks derive the SAME slot partition (no per-rank trigger asymmetry) and
        # each padded round's largest blob stays within the byte budget.
        per_elem = idx_concat.element_size() + val_concat.element_size()
        budget = max(int(max_round_bytes) // per_elem, 1)  # elements per rank per round
        cuts = [0]
        run = [0] * world
        for i in range(k):
            run = [run[r] + counts_cpu[r][i] for r in range(world)]
            if max(run) > budget and cuts[-1] != i:
                cuts.append(i)
                run = [counts_cpu[r][i] for r in range(world)]
        cuts.append(k)
        if len(cuts) > 2:
            out_all: list = []
            my_off = [0]
            for i in range(k):
                my_off.append(my_off[-1] + counts_cpu[rank][i])
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
                sub_counts = torch.tensor(counts_cpu[rank][lo:hi], dtype=torch.int64, device=dev)
                sub = gather_slot_entries_to_rank0(
                    idx_concat[my_off[lo] : my_off[hi]],
                    val_concat[my_off[lo] : my_off[hi]],
                    sub_counts,
                    group=group,
                    stats=stats,
                )
                if rank == 0:
                    out_all.extend(sub)
            return _gather_ret(out_all, rank)

    totals = [sum(c) for c in counts_cpu]
    max_n = max(totals) if totals else 0
    if max_n == 0:
        if rank != 0:
            return None
        empty_i = torch.empty(0, dtype=idx_concat.dtype, device=dev)
        empty_v = torch.empty(0, dtype=val_concat.dtype, device=dev)
        return _gather_ret([(empty_i, empty_v) for _ in range(k)], rank)

    if _decide_a2a(group, world) == "a2a":
        # [Round 10] exact-size move: every rank sends only its own
        # totals[rank] elements to group rank 0 (empty split to the rest);
        # rank 0 receives the world's blobs concatenated in group-rank order
        # -- the same layout the padded gathers produced, minus the zeros, so
        # the per-slot slicing below is unchanged. Moves sum(totals) bytes
        # instead of world * max_n (R9: 1956MB useful vs 8811MB pad on one
        # ~175MB/s line = 60.7s of payload for ~12s of useful bytes).
        n = int(idx_concat.numel())
        in_split = [0] * world
        in_split[0] = n
        if rank == 0:
            out_split = [int(t) for t in totals]
            idx_out = torch.empty(sum(out_split), dtype=idx_concat.dtype, device=dev)
            val_out = torch.empty(sum(out_split), dtype=val_concat.dtype, device=dev)
        else:
            out_split = [0] * world
            idx_out = torch.empty(0, dtype=idx_concat.dtype, device=dev)
            val_out = torch.empty(0, dtype=val_concat.dtype, device=dev)
        _tpayload = time.monotonic()
        dist.all_to_all_single(
            idx_out, idx_concat, output_split_sizes=out_split, input_split_sizes=in_split, group=group
        )
        dist.all_to_all_single(
            val_out, val_concat, output_split_sizes=out_split, input_split_sizes=in_split, group=group
        )
        if stats is not None:
            stats["payload"] += time.monotonic() - _tpayload
            per_elem = idx_concat.element_size() + val_concat.element_size()
            stats["useful_bytes"] += sum(totals) * per_elem
            stats["a2a_rounds"] = stats.get("a2a_rounds", 0) + 1
        if rank != 0:
            return None
        idx_list = []
        val_list = []
        off = 0
        for t in totals:
            idx_list.append(idx_out[off : off + t])
            val_list.append(val_out[off : off + t])
            off += int(t)
    else:
        idx_pad = torch.zeros(max_n, dtype=idx_concat.dtype, device=dev)
        val_pad = torch.zeros(max_n, dtype=val_concat.dtype, device=dev)
        n = int(idx_concat.numel())
        idx_pad[:n] = idx_concat
        val_pad[:n] = val_concat

        idx_list = [torch.zeros(max_n, dtype=idx_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
        val_list = [torch.zeros(max_n, dtype=val_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
        _tpayload = time.monotonic()
        dist.gather(idx_pad, idx_list, dst=dst, group=group)
        dist.gather(val_pad, val_list, dst=dst, group=group)
        if stats is not None:
            stats["payload"] += time.monotonic() - _tpayload
            per_elem = idx_pad.element_size() + val_pad.element_size()
            stats["pad_bytes"] += world * max_n * per_elem
            stats["useful_bytes"] += sum(totals) * per_elem
        if rank != 0:
            return None

    # per-rank cumulative offsets into each blob, sliced per param then stitched across ranks
    offs = [[0] * (k + 1) for _ in range(world)]
    for r in range(world):
        for i in range(k):
            offs[r][i + 1] = offs[r][i] + counts_cpu[r][i]
    out = []
    for i in range(k):
        idx_pieces = [idx_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        val_pieces = [val_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        if idx_pieces:
            out.append((torch.cat(idx_pieces), torch.cat(val_pieces)))
        else:
            out.append(
                (torch.empty(0, dtype=idx_concat.dtype, device=dev), torch.empty(0, dtype=val_concat.dtype, device=dev))
            )
    return _gather_ret(out, rank)


def _gather_ret(pieces: list, rank: int):
    """Rank filter for the gather return: rank 0 gets the per-slot pieces
    (device-resident, [Round 8]); every other rank gets None."""
    if rank != 0:
        return None
    return pieces
