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
"""Phase 0 hardware smoke tests for the delta HCCL port (go/no-go, B1/B2/B4/B5).

Run on an Ascend NPU host with CANN + torch_npu installed::

    pytest tests/special_npu/test_delta_hccl_phase0_smoke.py -v

- B1: torch.Tensor.copy_ monkey-patch layer visible to torch_npu tensors (the
  NaN-mask apply semantics rest on it; red -> engine-side where() fallback).
- B2: torch.hash_tensor(x.npu()).item() behavior record (informational; the
  wire checksum already uses the XOR tree by default).
- B4: SUB-first stateless rendezvous + PyHccl broadcast round trip -- the C1
  receiver geometry's primitive composition (red -> receiver falls back to C2).
- B5: dist.gather compatible-mode microbenchmark over ProcessGroupHCCL (the
  trainer-side sparse gather path; delta traffic is proportional to sparsity).
"""

import json
import multiprocessing as mp
import os
import time

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F841  (import side effect registers the npu backend)


def _npu_device():
    return torch.device("npu", torch.npu.current_device())


def test_b1_copy_monkeypatch_preserves_masked_values():
    """The NaN-mask apply patches Tensor.copy_; it must intercept torch_npu
    copies and the where() must keep the destination at NaN positions."""
    original_copy = torch.Tensor.copy_

    calls = []

    def counting_copy(self, src, **kwargs):
        calls.append((tuple(self.shape), str(self.device)))
        return original_copy(self, src, **kwargs)

    try:
        torch.Tensor.copy_ = counting_copy
        dst = torch.zeros(128, dtype=torch.bfloat16, device=_npu_device())
        src = torch.full((128,), float("nan"), dtype=torch.bfloat16, device=_npu_device())
        src[:64] = 1.0
        masked = torch.where(torch.isnan(src), dst, src)
        dst.copy_(masked)
    finally:
        torch.Tensor.copy_ = original_copy

    assert calls, "monkey-patched Tensor.copy_ did not intercept the npu copy"
    assert bool((dst[:64] == 1.0).all()), "non-NaN values were not applied"
    assert bool((dst[64:] == 0.0).all()), "NaN positions must keep the destination value"


def test_b2_hash_tensor_smoke_record():
    """Record uint64/.item() behavior on npu (informational for Phase 0 report)."""
    x = torch.randint(0, 256, (4097,), dtype=torch.uint8, device=_npu_device())
    try:
        value = int(torch.hash_tensor(x).item())
    except Exception as exc:  # noqa: BLE001  (behavior record is the point)
        pytest.skip(f"torch.hash_tensor not usable on this torch_npu build: {exc!r}")
    assert 0 <= value < (1 << 64)
    # determinism within a process
    assert int(torch.hash_tensor(x).item()) == value


def _b4_trainer(dist_port: int, zmq_port: int, q):
    import zmq

    from verl.utils.distributed import stateless_init_process_group

    device = _npu_device()
    pos = torch.randint(0, 256, (1024,), dtype=torch.uint8, device=device)
    val = torch.randn(1024, device=device).to(torch.bfloat16).view(torch.uint8)

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{zmq_port}")
    comm = stateless_init_process_group("127.0.0.1", dist_port, 0, 2, device)
    # manifest first, then broadcasts, then sync -- the engine's publish order
    pub.send_string("bucket_metadata", flags=zmq.SNDMORE)
    pub.send_pyobj({"pos_numel": pos.numel(), "val_numel": val.numel(), "is_last": True})
    comm.broadcast(pos, src=0)
    comm.broadcast(val, src=0)
    torch.npu.synchronize()
    q.put(("trainer_payload", pos.cpu(), val.cpu()))


def _b4_receiver(dist_port: int, zmq_port: int, q):
    import zmq

    from verl.checkpoint_engine.delta_sync.hccl_receiver import (
        build_delta_hccl_communicator,
        connect_delta_manifest_sub,
    )

    device = _npu_device()
    # SUB BEFORE communicator: the C1 no-lost-manifest ordering
    sub = connect_delta_manifest_sub("127.0.0.1", zmq_port)
    comm = build_delta_hccl_communicator("127.0.0.1", dist_port, 1, 2, device)
    sub.recv_string()
    meta = sub.recv_pyobj()
    pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device=device)
    val = torch.empty(meta["val_numel"], dtype=torch.uint8, device=device)
    comm.broadcast(pos, src=0)
    comm.broadcast(val, src=0)
    torch.npu.synchronize()
    q.put(("receiver_payload", pos.cpu(), val.cpu()))


def test_b4_sub_first_rendezvous_broadcast_roundtrip():
    """One trainer + one receiver: manifest over zmq PUB/SUB, payload over a
    lazily built PyHccl group; payload must arrive bit-exact."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    dist_port, zmq_port = 29511, 29512
    receiver = ctx.Process(target=_b4_receiver, args=(dist_port, zmq_port, q))
    trainer = ctx.Process(target=_b4_trainer, args=(dist_port, zmq_port, q))
    receiver.start()
    trainer.start()
    trainer.join(timeout=180)
    receiver.join(timeout=180)
    assert trainer.exitcode == 0, f"trainer failed: exitcode {trainer.exitcode}"
    assert receiver.exitcode == 0, f"receiver failed: exitcode {receiver.exitcode}"
    payloads = {}
    for _ in range(2):
        tag, pos, val = q.get_nowait()
        payloads[tag] = (pos, val)
    assert torch.equal(payloads["receiver_payload"][0], payloads["trainer_payload"][0]), "positions corrupted"
    assert torch.equal(payloads["receiver_payload"][1], payloads["trainer_payload"][1]), "values corrupted"


def _b5_gather_worker(rank: int, world_size: int, port: int, nnz: int, q):
    import torch.distributed as dist

    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    device = _npu_device()
    idx = torch.randint(0, 1 << 30, (nnz,), dtype=torch.int64, device=device)
    val = torch.randn(nnz, dtype=torch.bfloat16, device=device)
    t0 = time.time()
    rounds = 20
    for _ in range(rounds):
        if rank == 0:
            gathered = [torch.empty_like(val) for _ in range(world_size)]
            dist.gather(val, gathered)
        else:
            dist.gather(val)
    dist.barrier()
    elapsed = time.time() - t0
    if rank == 0:
        q.put({"rounds": rounds, "nnz": nnz, "seconds": elapsed})
    dist.destroy_process_group()


@pytest.mark.parametrize("nnz", [1024, 1 << 20])
def test_b5_dist_gather_compatible_mode_benchmark(nnz):
    """Time dist.gather over ProcessGroupHCCL (p2p simulated gather) at sparse
    payload sizes; records throughput for the Phase 0 report."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world_size = 2
    procs = [ctx.Process(target=_b5_gather_worker, args=(r, world_size, 29513 + nnz % 7, nnz, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0, f"gather worker failed: exitcode {p.exitcode}"
    stats = q.get_nowait()
    mb = stats["rounds"] * stats["nnz"] * 2 / (1 << 20)
    print(
        f"B5 dist.gather (hccl compat mode): {stats['rounds']} rounds x {stats['nnz']} elems"
        f" = {mb:.1f} MB in {stats['seconds']:.3f}s ({mb / stats['seconds']:.1f} MB/s)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
