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
"""Receiver-side HCCL delta transport: zmq manifest subscription + PyHccl group.

Shared by the vLLM worker engine (the C1 production receiver) and direct
receivers used in tests. The SUB socket MUST be connected before the
communicator is built (rendezvous-as-ready-gate). Code order alone cannot make
the manifest stream lossless -- ``zmq.Socket.connect()`` is asynchronous, so a
"connected" SUB can still miss the first published manifest -- so every
receiver raises a per-rank ``delta_wire_ready/{rank}`` flag in the group's
TCPStore strictly after its communicator joined, and the trainer polls those
flags (:func:`wait_delta_wire_ready`) before its first publish.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch
import zmq

from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import is_valid_ipv6_address

from .encode import pack_spec

logger = logging.getLogger(__name__)

# A flush (broadcast + apply) normally takes seconds; anything beyond this is
# almost always the stall that precedes a group-wide HCCL hang, so shout early.
SLOW_FLUSH_SECONDS = 120.0

# A manifest is normally seconds behind its sender; a silent gap beyond this
# means the sender stalled or died mid-session, so fail loud -- the session's
# fail-fast abort then wakes the rest of the group instead of hanging it.
MANIFEST_RECV_TIMEOUT_SECONDS = 300.0

# vLLM startup + model load on the rollout side can take many minutes; the
# sender waits at most this long for every receiver's wire-ready flag before
# publishing the first manifest.
WIRE_READY_TIMEOUT_SECONDS = 1800.0


def connect_delta_manifest_sub(zmq_ip: str, zmq_port: int, topic: str = "bucket_metadata") -> zmq.Socket:
    """Subscribe to the trainer's flush-manifest PUB socket."""
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    if is_valid_ipv6_address(zmq_ip):
        sub.setsockopt(zmq.IPV6, 1)
        sub.connect(f"tcp://[{zmq_ip}]:{zmq_port}")
    else:
        sub.connect(f"tcp://{zmq_ip}:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sub


def build_delta_hccl_communicator(dist_ip: str, dist_port: int, group_rank: int, world_size: int, device):
    """Join the trainer's PyHccl broadcast group at ``group_rank``."""
    import os as _os
    _vis = _os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
    _cur = torch.npu.current_device()
    _ray_ids = None
    try:
        import ray as _ray
        _rt = _ray.get_runtime_context()
        _ray_ids = _rt.get_accelerator_ids() if _ray.is_initialized() else None
    except Exception:
        pass
    import logging as _logging
    _log = _logging.getLogger("verl.checkpoint_engine.delta_sync.hccl_receiver")
    _log.warning(
        "build_delta_hccl_communicator: group_rank=%d world_size=%d device=%s current_device=%s "
        "ASCEND_RT_VISIBLE_DEVICES=%s ray_accel_ids=%s",
        group_rank, world_size, device, _cur, _vis, _ray_ids,
    )
    _t0 = time.monotonic()
    _comm = stateless_init_process_group(dist_ip, dist_port, group_rank, world_size, device)
    _log.warning(
        "build_delta_hccl_communicator: group_rank=%d communicator JOINED in %.1fs",
        group_rank,
        time.monotonic() - _t0,
    )
    if group_rank > 0:
        # Raised strictly after joining so the sender's pre-publish poll proves
        # the SUB's SUBSCRIBE message already reached the PUB socket -- a
        # process-level flag is the only sound receipt (zmq connect is async).
        _comm.group.store.add(f"delta_wire_ready/{group_rank}", 1)
        _log.warning("delta-hccl rank=%d wire ready flag raised", group_rank)
    return _comm


def wait_delta_wire_ready(store, num_receivers: int, timeout: float = WIRE_READY_TIMEOUT_SECONDS) -> None:
    """Block until group ranks ``1..num_receivers`` raised ``delta_wire_ready``.

    ``store.add(key, 0)`` reads the counter without blocking, so a missing
    receiver stays visible (and nameable in the TimeoutError) instead of
    turning into a KeyError.
    """
    t0 = time.monotonic()
    missing = list(range(1, num_receivers + 1))
    while missing:
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(
                f"delta wire handshake timed out after {timeout:.0f}s: no wire-ready flag from group ranks {missing}"
            )
        missing = [r for r in missing if store.add(f"delta_wire_ready/{r}", 0) < 1]
        if missing:
            time.sleep(0.25)


def _collect_bucket_tensors(
    buf: torch.Tensor, bucket_meta: dict, merging: dict, bucket_size: int
) -> list[tuple[str, torch.Tensor]]:
    """Materialize complete tensors from one received dense bucket.

    Mirrors ``merge_weight_chunks``: tensors that fit in one bucket become views
    into the receive buffer (valid until that buffer is reused two swaps later
    -- the caller must apply before the next broadcast lands), oversized
    tensors accumulate into private storage keyed by name."""
    out: list[tuple[str, torch.Tensor]] = []
    for tm in bucket_meta.values():
        nbytes = int(tm.shape.numel()) * tm.dtype.itemsize
        chunk = buf[tm.offset : tm.offset + tm.chunk_size]
        if nbytes <= bucket_size:
            assert tm.chunk_offset == 0, f"{tm.name}: mid-tensor chunk for a bucket-sized tensor"
            out.append((tm.name, chunk.view(tm.dtype).view(tuple(tm.shape))))
            continue
        entry = merging.get(tm.name)
        if entry is None:
            assert tm.chunk_offset == 0, f"{tm.name}: first chunk at offset {tm.chunk_offset} != 0"
            entry = merging[tm.name] = [torch.empty(nbytes, dtype=torch.uint8, device=buf.device), 0]
        flat, next_off = entry
        assert next_off == tm.chunk_offset, f"{tm.name}: chunk gap, expected offset {next_off}, got {tm.chunk_offset}"
        flat[tm.chunk_offset : tm.chunk_offset + tm.chunk_size] = chunk
        entry[1] = tm.chunk_offset + tm.chunk_size
        if tm.chunk_offset + tm.chunk_size == nbytes:
            out.append((tm.name, flat.view(tm.dtype).view(tuple(tm.shape))))
            del merging[tm.name]
    return out


def _pump_dense_seed_buckets(
    sub: zmq.Socket,
    comm,
    device,
    bucket_size: int,
    on_buckets: Callable[[list[tuple[str, torch.Tensor]]], None],
    rank_tag: str,
) -> None:
    """Consume one ``dense_seed`` session: the dense checkpoint engine's bucket
    protocol carried over this group (the session-opening manifest was already
    consumed by the caller). Each round receives one bucket-metadata message,
    broadcasts the bucket buffer, then hands the complete tensors to
    ``on_buckets`` -- which must apply them before returning (they view the
    receive buffer)."""
    synchronize = getattr(torch, device.type).synchronize
    t0 = time.monotonic()
    send_buf = torch.empty(bucket_size, dtype=torch.uint8, device=device)
    recv_buf = torch.empty_like(send_buf)
    merging: dict = {}
    n_buckets = 0
    n_tensors = 0
    # [TIMING-T2 2026-10-10] bcast (wire) vs apply (weight_loader) split, same
    # scope as the indices pump's session totals, so the full seed and the
    # steady delta rounds report data-transfer time on one yardstick.
    bcast_total = 0.0
    apply_total = 0.0
    try:
        while True:
            if sub.poll(int(MANIFEST_RECV_TIMEOUT_SECONDS * 1000)) == 0:
                raise TimeoutError(
                    f"dense seed: no bucket metadata within {MANIFEST_RECV_TIMEOUT_SECONDS:.0f}s "
                    f"(buckets={n_buckets}); sender stalled mid-seed or exited without destroying the group"
                )
            sub.recv_string()
            bmeta = sub.recv_pyobj()
            _tb = time.monotonic()
            comm.broadcast(recv_buf, src=0)
            synchronize()
            bcast_total += time.monotonic() - _tb
            # swap: read the bucket just received from send_buf while the next
            # broadcast lands in recv_buf (stream ordering keeps the views of
            # the previous bucket valid through ``on_buckets``)
            send_buf, recv_buf = recv_buf, send_buf
            n_buckets += 1
            _ta = time.monotonic()
            named = _collect_bucket_tensors(send_buf, bmeta["bucket_meta"], merging, bucket_size)
            if named:
                on_buckets(named)
                n_tensors += len(named)
            apply_total += time.monotonic() - _ta
            if bmeta.get("is_last"):
                break
    finally:
        del send_buf, recv_buf
    assert not merging, f"dense seed ended with incomplete tensors: {sorted(merging)}"
    logger.warning(
        "delta-hccl pump rank=%s DENSE-SEED done in %.1fs (buckets=%d tensors=%d) "
        "[TIMING-T2 bcast_total=%.1fs apply_total=%.1fs]",
        rank_tag,
        time.monotonic() - t0,
        n_buckets,
        n_tensors,
        bcast_total,
        apply_total,
    )


def pump_hccl_delta_session(
    sub: zmq.Socket,
    comm,
    device,
    on_flush: Callable[[list[tuple[str, torch.Tensor]]], None],
    group_rank: int | None = None,
    on_dense_buckets: Callable[[list[tuple[str, torch.Tensor]]], None] | None = None,
) -> str:
    """Pull one delta session (manifests + broadcast payloads) and hand each
    flush to ``on_flush(named_tensors)``.

    A session whose opening manifest carries ``kind == "dense_seed"`` instead
    carries the dense engine's bucket protocol; its complete tensors go to
    ``on_dense_buckets`` (fail loud when the adapter does not provide one).

    Returns the session's wire encoding (``"dense"`` | ``"indices"`` |
    ``"dense_buckets"``); a terminal-only session (steady sync with zero
    changes) returns ``"indices"`` with zero ``on_flush`` calls. The device
    synchronize after each flush's broadcasts mirrors the sender: PyHccl
    enqueues on the current stream, so the received buffers must not be read
    before completion.

    Every flush emits one WARNING line tagged with ``group_rank`` and its
    session-local sequence number, plus broadcast/apply timings: at default
    log level the last line of each receiver is what names the rank and flush
    a group-wide stall died on.
    """
    rank_tag = "?" if group_rank is None else str(group_rank)
    synchronize = getattr(torch, device.type).synchronize
    encoding = "indices"
    flush_seq = 0
    # [TIMING-T2 2026-10-10] Per-session wire (bcast) vs apply totals: the
    # receiver-measured broadcast time is the honest "data transfer" number for
    # the T1/T2/T3 sync-cost decomposition (T2), on the same yardstick as the
    # dense seed's DENSE-SEED totals.
    bcast_total = 0.0
    apply_total = 0.0
    logger.warning("delta-hccl pump rank=%s session START", rank_tag)
    try:
        while True:
            if sub.poll(int(MANIFEST_RECV_TIMEOUT_SECONDS * 1000)) == 0:
                raise TimeoutError(
                    f"no manifest within {MANIFEST_RECV_TIMEOUT_SECONDS:.0f}s (flush={flush_seq}); "
                    "sender stalled mid-session or exited without destroying the group"
                )
            sub.recv_string()
            meta = sub.recv_pyobj()
            if meta.get("kind") == "dense_seed":
                if on_dense_buckets is None:
                    raise NotImplementedError(
                        "sender opened a dense_seed session but this receiver has no dense-bucket consumer"
                    )
                _pump_dense_seed_buckets(sub, comm, device, int(meta["bucket_size"]), on_dense_buckets, rank_tag)
                logger.warning("delta-hccl pump rank=%s session END (dense seed)", rank_tag)
                return "dense_buckets"
            if meta.get("terminal_empty"):
                logger.warning("delta-hccl pump rank=%s TERMINAL empty after %d flush(es)", rank_tag, flush_seq)
                break
            encoding = meta.get("encoding", "indices")
            dense = encoding == "dense"
            val_dtype = getattr(torch, meta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device=device)
            logger.warning(
                "delta-hccl pump rank=%s flush=%d enc=%s pos=%d val=%d last=%s: "
                "manifest in, entering broadcast wall=%.3f",
                rank_tag,
                flush_seq,
                encoding,
                meta.get("pos_numel", 0),
                meta["val_numel"],
                meta["is_last"],
                time.time(),
            )
            t_bcast = time.monotonic()
            if dense:
                pos = None
                comm.broadcast(val_u8, src=0)
            else:
                pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device=device)
                comm.broadcast(pos, src=0)
                comm.broadcast(val_u8, src=0)
            synchronize()
            t_apply = time.monotonic()
            bcast_seconds = t_apply - t_bcast
            bcast_total += bcast_seconds
            val = val_u8.view(val_dtype)
            # [Round 7] binary spec (DSB1) instead of json.dumps -- ~2.5s/flush
            # at 15k patches; kept host-resident because decode reads it back to
            # host anyway (the old .to(device) was a pure H2D+D2H round trip).
            spec_bytes = pack_spec(meta["spec"])
            spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8)
            named = [("__delta_spec__", spec_t), ("__values__", val)]
            if pos is not None:
                named.insert(1, ("__positions__", pos))
            on_flush(named)
            apply_seconds = time.monotonic() - t_apply
            apply_total += apply_seconds
            logger.warning(
                "delta-hccl pump rank=%s flush=%d APPLIED bcast=%.1fs apply=%.1fs%s wall=%.3f",
                rank_tag,
                flush_seq,
                bcast_seconds,
                apply_seconds,
                " (SLOW)" if apply_seconds > SLOW_FLUSH_SECONDS else "",
                time.time(),
            )
            del pos, val_u8, val, spec_t
            flush_seq += 1
            if meta["is_last"]:
                logger.warning(
                    "delta-hccl pump rank=%s session END after %d flush(es) "
                    "[TIMING-T2 bcast_total=%.1fs apply_total=%.1fs]",
                    rank_tag,
                    flush_seq,
                    bcast_total,
                    apply_total,
                )
                break
    except BaseException:
        logger.exception("delta-hccl pump rank=%s FAILED at flush=%d", rank_tag, flush_seq)
        raise
    return encoding
