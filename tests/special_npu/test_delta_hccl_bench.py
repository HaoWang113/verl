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
"""Timing bench: delta weight sync vs constructed sparsity (NPU).

Run on an Ascend NPU host with the same requirements as the E2E correctness
test (8 devices, ray, model on local disk)::

    pytest tests/special_npu/test_delta_hccl_bench.py -v -s

Protocol: the delta engine's seed round ships the full model over the same
wire, so it doubles as the full-sync baseline. Steady rounds perturb a
controlled fraction of the float elements with a value-band predicate
(element-wise, hence shard-safe on FSDP2 local shards); the bands are
upper-tail (top-r%) and nested, and the assigned constant (a small bf16
integer far above every band threshold) differs every round, so band
membership never depends on previous assignments and each round of the
same band yields a fresh delta of ~the same mass. Band thresholds come
from an approximate quantile probe over the trainer shards; achieved
ratios are read back from the engine's changed_ratio metric, so drift
from the target is visible in the table.

Only the sync channel is timed: the mock apply (dict overwrite /
index_copy_) is far cheaper than a real vLLM load_weights, so numbers are an
upper bound on real-run throughput. No performance assertions here -- the
printed table is the deliverable.
"""

import os
import time

import pytest
import ray
import torch

from tests.checkpoint_engine.test_correctness_on_npu import (
    TrainingWorkerDeltaTest,
    create_delta_rollout_worker_group,
)
from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.device import get_device_name
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig

TARGET_RATIOS = (0.0, 0.01, 0.10, 0.25, 0.50, 1.0)
ROUNDS_PER_RATIO = 3
QUANTILE_PROBE_SIZE = 1 << 22  # per-rank subsample for the threshold probe


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


class TrainingWorkerDeltaBench(TrainingWorkerDeltaTest):
    """E2E trainer worker plus a parameterized band perturbation and an
    approximate quantile probe (both shard-safe on FSDP2)."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def perturb_band(self, lo: float, hi: float, value: float):
        """Every float element with lo < x <= hi := value.

        ``value`` cycles over small bf16-exact integers far above every band
        threshold, each round's constant differing from the previous round's
        assignment -- band membership therefore never depends on previous
        rounds, and every round produces a fresh delta of the same mass.
        """
        for _, param in self.engine.module.state_dict().items():
            if not param.is_floating_point():
                continue
            local = param.to_local() if hasattr(param, "to_local") else param
            with torch.no_grad():
                local[(local > lo) & (local <= hi)] = value

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def float_quantiles(self, qs: list[float]) -> list[float]:
        """Approximate global quantiles of this rank's float shards
        (deterministic strided subsample, then sort)."""
        vals = []
        for _, param in self.engine.module.state_dict().items():
            if param.is_floating_point():
                local = param.to_local() if hasattr(param, "to_local") else param
                vals.append(local.detach().reshape(-1))
        flat = torch.cat(vals).float()
        flat = flat[:: max(1, flat.numel() // QUANTILE_PROBE_SIZE)]
        srt = torch.sort(flat)[0]
        idx = [min(int(q * srt.numel()), srt.numel() - 1) for q in qs]
        return srt[idx].tolist()


@pytest.mark.asyncio
@auto_await
async def test_delta_hccl_bench(
    num_trainer=2,
    num_rollout=6,
    num_nodes=1,
    num_gpus_per_node=8,
    model_path="~/models/Qwen/Qwen3-8B-Base",
):
    model_path = os.path.expanduser(model_path)
    ray.init(
        runtime_env={
            "env_vars": {
                "HCCL_CONNECT_TIMEOUT": "1500",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
                "VERL_LOGGING_LEVEL": "DEBUG",
            }
        }
    )

    # single 3072MB bucket: measure the wire, not the chunking path
    checkpoint_engine_config = CheckpointEngineConfig(
        backend="delta_sharded",
        engine_kwargs={"delta_sharded": {}},
        update_weights_bucket_megabytes=3072,
    )
    model_config = HFModelConfig(path=model_path, use_remove_padding=True)
    rollout_config = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=1,
        checkpoint_engine=checkpoint_engine_config,
    )

    resource_pool = RayResourcePool(process_on_nodes=[num_gpus_per_node] * num_nodes, max_colocate_count=3)
    resource_pool.get_placement_groups(device_name=get_device_name())
    trainer_pool, rollout_pool = split_resource_pool(resource_pool, [num_trainer, num_rollout])
    actor_wg = create_trainer_worker_group(
        trainer_pool, model_config, checkpoint_engine_config, worker_cls=TrainingWorkerDeltaBench
    )
    actor_wg.reset()
    # correctness is the E2E test's job; the bench only times the channel
    rollout, replicas = await create_delta_rollout_worker_group(
        rollout_pool, model_config, rollout_config, check_allclose=False
    )

    checkpoint_manager = CheckpointEngineManager(config=checkpoint_engine_config, actor_wg=actor_wg, replicas=replicas)

    # 1. seed round: the full model over the delta wire == full-sync baseline
    t0 = time.perf_counter()
    m = await checkpoint_manager.update_weights()
    seed_engine_s = m.get("checkpoint_engine/elapsed_seconds")
    seed_wall_s = time.perf_counter() - t0
    seed_payload_mb = m.get("checkpoint_engine/payload_mbytes", 0.0)
    assert seed_engine_s is not None and seed_engine_s > 0, m

    # 2. quantile probe on the pristine weights -> upper-tail band thresholds.
    #    Bands are nested top-r% tails; the assigned constants (2.0 and up)
    #    stay above every threshold, so band membership never depends on
    #    previous rounds' assignments.
    qs = [0.5, 0.75, 0.90, 0.99]
    per_rank = actor_wg.float_quantiles(qs)
    thr = [sum(col) / len(col) for col in zip(*per_rank)]
    bands = {
        0.01: (thr[3], float("inf")),
        0.10: (thr[2], float("inf")),
        0.25: (thr[1], float("inf")),
        0.50: (thr[0], float("inf")),
        1.00: (float("-inf"), float("inf")),
    }

    # 3. warm-up steady round (no changes): settles lazy paths, discarded
    await checkpoint_manager.update_weights()

    # 4. sweep. r=0 measures the fixed per-round overhead; the per-ratio
    #    assignment constant grows by 2 so it always differs from both the
    #    previous round and the previous ratio's last assignment.
    rows = []
    nz = 0  # index over the non-zero ratios
    for r in TARGET_RATIOS:
        lo, hi = (0.0, 0.0) if r == 0.0 else bands[r]
        samples = []
        for k in range(ROUNDS_PER_RATIO):
            if r > 0.0:
                base = 2.0 + 2 * nz
                actor_wg.perturb_band(lo, hi, base + (k % 2))
            t0 = time.perf_counter()
            m = (await checkpoint_manager.update_weights()) or {}
            wall = time.perf_counter() - t0
            samples.append(
                (
                    m.get("checkpoint_engine/changed_ratio", 0.0),
                    m.get("checkpoint_engine/payload_mbytes", 0.0),
                    m.get("checkpoint_engine/elapsed_seconds", wall),
                    wall,
                )
            )
        rows.append((r, *(_median(list(col)) for col in zip(*samples))))
        if r > 0.0:
            nz += 1

    assert max(ratio for _, ratio, *_ in rows) > 0.1, "perturbation produced no deltas"

    # 5. report
    print(f"\n{'target':>7} {'achieved':>9} {'payload_MB':>11} {'engine_s':>9} {'wall_s':>8} {'MB/s':>8} {'speedup':>8}")
    for r, ratio, mb, engine_s, wall_s in rows:
        mbps = mb / engine_s if engine_s > 0 else 0.0
        speedup = seed_engine_s / engine_s if engine_s > 0 else 0.0
        print(f"{r:>7.2f} {ratio:>9.4f} {mb:>11.1f} {engine_s:>9.2f} {wall_s:>8.2f} {mbps:>8.0f} {speedup:>7.1f}x")
    print(
        f"\nseed (full model over the same wire): {seed_engine_s:.2f}s engine, "
        f"{seed_wall_s:.2f}s wall, {seed_payload_mb:.0f} MB\n"
    )

    ray.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
