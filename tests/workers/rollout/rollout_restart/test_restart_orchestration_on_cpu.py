# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the rollout-restart orchestration pieces (PR-B/C/D).

Covers: RestartBudget window/cooldown semantics, the fault mailbox's
append+dedup+drain contract, the checkpoint locator's commit-point scan
(design §7.5, scenarios S4/S6), the supervisor's budget exhaustion -> DEAD
transition and false-alarm handling, and the CheckpointEngineManager's
fencing generation, commit-failure classification, and defer-resume split —
all without accelerators (half-object pattern for the manager, real Ray for
the mailbox, real temp directories for the locator).
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
import ray

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.utils.checkpoint.locate import CheckpointUnavailable, locate_latest_valid_checkpoint
from verl.workers.rollout.restart import (
    ReplicaFaultMailbox,
    ReplicaRestartSupervisor,
    ReplicaRuntimeState,
    RestartBudget,
)


class TestRestartBudget:
    def test_allows_up_to_max_restarts(self):
        budget = RestartBudget(max_restarts=2, window_s=3600.0, cooldown_s=0.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False

    def test_cooldown_blocks_bursts(self):
        budget = RestartBudget(max_restarts=3, window_s=3600.0, cooldown_s=60.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False  # inside the cooldown

    def test_window_slides(self):
        budget = RestartBudget(max_restarts=1, window_s=0.05, cooldown_s=0.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False
        time.sleep(0.08)
        assert budget.try_acquire() is True


class TestFaultMailbox:
    def test_report_dedup_and_drain_reset(self):
        ray.init(ignore_reinit_error=True)
        mailbox = ReplicaFaultMailbox.remote()
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "EngineDeadError"}))
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "EngineDeadError"}))
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "ActorDiedError"}))
        reports = ray.get(mailbox.drain.remote())
        assert len(reports) == 2  # same (server, error_class) deduped within the cycle
        assert ray.get(mailbox.drain.remote()) == []
        # After a drain the dedup window resets: the same evidence is accepted again.
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "EngineDeadError"}))
        assert len(ray.get(mailbox.drain.remote())) == 1


class TestMailboxWaitNext:
    def test_wait_next_wakes_on_report_and_resets_dedup(self):
        ray.init(ignore_reinit_error=True)
        mailbox = ReplicaFaultMailbox.remote()
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "EngineDeadError"}))
        reports = ray.get(mailbox.wait_next.remote(timeout_s=5.0))
        assert len(reports) == 1 and reports[0][0] == "addr-0"
        assert ray.get(mailbox.wait_next.remote(timeout_s=0.2)) == []
        ray.get(mailbox.report.remote("addr-0", {"source": "request_path", "error_class": "EngineDeadError"}))
        # Batch consumption resets the dedup window, same semantics as drain.
        assert len(ray.get(mailbox.wait_next.remote(timeout_s=5.0))) == 1


class TestMailboxAwaiter:
    def test_request_path_report_convicts_immediately_without_sweep(self):
        """§5.1① rev: a request-path report convicts via the resident awaiter —
        no update_weights sweep is ever invoked in this scenario."""
        import time

        from verl.workers.rollout.restart import FaultEvidence

        ray.init(ignore_reinit_error=True)
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor.start()
        try:
            scheduled = []
            supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

            class _DeadReplica:
                replica_rank = 0
                name_suffix = ""
                servers = []
                workers = []

            manager.rollout_replicas = [_DeadReplica()]
            manager.server_addresses = ["addr-0"]
            supervisor._probe_replica = lambda r: _async_value(
                FaultEvidence(source="sweep", error_class="EngineDeadError")
            )

            # Fire-and-forget exactly as the request path does (llm_server._report_fault).
            supervisor.mailbox_handle.report.remote(
                "addr-0", {"source": "request_path", "error_class": "EngineDeadError"}
            )
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and supervisor.get_state("0") != ReplicaRuntimeState.ISOLATED:
                time.sleep(0.05)
            assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
            assert scheduled == ["0"]
        finally:
            supervisor.stop()


def _make_megatron_step(root, step, complete=True, with_hf=True):
    step_dir = root / f"global_step_{step}"
    model_dir = step_dir / "actor" / "model"
    (model_dir / "huggingface").mkdir(parents=True, exist_ok=True)
    if with_hf:
        (model_dir / "huggingface" / "config.json").write_text("{}")
        (model_dir / "huggingface" / "tokenizer_config.json").write_text("{}")
        (model_dir / "huggingface" / "model-00001-of-00002.safetensors").write_text("x")
    if complete:
        (model_dir / "ckpt_contents.json").write_text("{}")
    return step_dir


def _make_fsdp_step(root, step, committed=True):
    step_dir = root / f"global_step_{step}"
    hf_dir = step_dir / "actor" / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "tokenizer_config.json").write_text("{}")
    (hf_dir / "model.safetensors").write_text("x")
    if committed:
        (step_dir / "manifest.json").write_text("{}")
    return step_dir


class TestLocateLatestValidCheckpoint:
    def test_prefers_tracker_step(self, tmp_path):
        _make_megatron_step(tmp_path, 10)
        _make_megatron_step(tmp_path, 20)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")
        path, step = locate_latest_valid_checkpoint(str(tmp_path))
        assert step == 10
        assert path.endswith("global_step_10/actor/model/huggingface")

    def test_falls_back_when_tracker_step_is_broken(self, tmp_path):
        # S4/S6: a crash landed mid-save at step 20 (no manifest); the scan must
        # fall back to the last complete one.
        _make_megatron_step(tmp_path, 10)
        _make_megatron_step(tmp_path, 20, complete=False)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")
        path, step = locate_latest_valid_checkpoint(str(tmp_path))
        assert step == 10

    def test_skips_steps_without_hf_export(self, tmp_path):
        _make_megatron_step(tmp_path, 30, complete=True, with_hf=False)
        _make_megatron_step(tmp_path, 24)
        path, step = locate_latest_valid_checkpoint(str(tmp_path))
        assert step == 24

    def test_fsdp_manifest_form(self, tmp_path):
        _make_fsdp_step(tmp_path, 7, committed=True)
        path, step = locate_latest_valid_checkpoint(str(tmp_path))
        assert step == 7
        assert path.endswith("global_step_7/actor/huggingface")

    def test_fsdp_uncommitted_manifest_is_invalid(self, tmp_path):
        _make_fsdp_step(tmp_path, 7, committed=False)
        with pytest.raises(CheckpointUnavailable):
            locate_latest_valid_checkpoint(str(tmp_path))

    def test_no_checkpoint_raises_unavailable(self, tmp_path):
        with pytest.raises(CheckpointUnavailable):
            locate_latest_valid_checkpoint(str(tmp_path))

    def test_missing_root_raises_unavailable(self, tmp_path):
        with pytest.raises(CheckpointUnavailable):
            locate_latest_valid_checkpoint(str(tmp_path / "does-not-exist"))


def _manager_half_object():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager._restart_enabled = False
    manager._restart_config = None
    manager._fault_report = None
    manager._sweep_hook = None
    manager._deferred_resume_callback = None
    manager._deferred_resumes = set()
    manager._sync_exclusivity = __import__("threading").Lock()
    manager._group_generation = 0
    manager._fatal_error = None
    return manager


class _FakeRolloutReplica:
    def __init__(self, rank, suffix=""):
        self.replica_rank = rank
        self.name_suffix = suffix
        self.workers = [SimpleNamespace()]


class TestManagerFencing:
    def test_gen0_is_inert(self):
        manager = _manager_half_object()
        assert manager._fenced_group_name() is None
        assert manager._group_generation == 0

    def test_invalidate_bumps_generation(self):
        manager = _manager_half_object()
        manager.enable_restart(SimpleNamespace(sync_lock_wait_s=1.0, bounded_abort_s=1.0))
        assert manager.invalidate_group() == "nccl_ft_g1"
        assert manager.invalidate_group() == "nccl_ft_g2"
        assert manager._fenced_group_name() == "nccl_ft_g2"

    def test_replica_key_includes_suffix(self):
        replica = _FakeRolloutReplica(3, "_reward")
        assert CheckpointEngineManager._replica_key(replica) == "3_reward"

    def test_classify_commit_failures(self):
        manager = _manager_half_object()
        replica_a, replica_b = _FakeRolloutReplica(0), _FakeRolloutReplica(1)
        slices = [(replica_a, 0, 1), (replica_b, 1, 2)]
        # replica A's worker raised; replica B completed.
        failed, stuck = manager._classify_commit_failures([ValueError("dead")], [True], slices)
        assert failed == [replica_a]
        assert stuck == []
        # replica B's worker never finished within the bounded abort (stuck collective).
        failed, stuck = manager._classify_commit_failures([None], [False], slices)
        assert failed == []
        assert stuck == replica_b.workers

    def test_defer_weight_sync_resume_and_callback(self):
        manager = _manager_half_object()
        manager.enable_restart(SimpleNamespace())
        callback_keys = []
        manager._deferred_resume_callback = callback_keys.append
        replica = _FakeRolloutReplica(0)

        class _Resumes:
            calls = 0

            async def resume_kv_cache(self):
                _Resumes.calls += 1

            async def resume_generation(self):
                _Resumes.calls += 1

        replica.resume_kv_cache = _Resumes().resume_kv_cache
        replica.resume_generation = _Resumes().resume_generation
        manager.defer_weight_sync_resume(replica)
        asyncio.run(manager._guarded_resume([replica]))
        assert callback_keys == ["0"]
        assert manager._deferred_resumes == set()

    def test_update_weights_raises_on_fatal(self):
        manager = _manager_half_object()
        manager.enable_restart(SimpleNamespace(sync_lock_wait_s=1.0))
        manager._fatal_error = "all rollout replicas are DEAD"

        async def scenario():
            try:
                await manager.update_weights(1)
                raised = None
            except RuntimeError as exc:
                raised = exc
            assert raised is not None and "fatal" in str(raised)

        asyncio.run(scenario())


def _supervisor_with_fake_manager(config=None, rollout_config=None):
    rollout_config = rollout_config or SimpleNamespace(
        name="vllm",
        nnodes=1,
        disaggregation=SimpleNamespace(enabled=False),
        checkpoint_engine=SimpleNamespace(backend="nccl"),
        quantization=SimpleNamespace(enabled=False),
    )
    manager = SimpleNamespace(
        rollout_config=rollout_config,
        model_config=SimpleNamespace(lora_rank=0),
        rollout_replicas=[],
        server_addresses=[],
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=None)),
    )
    supervisor = ReplicaRestartSupervisor(manager=manager, config=config or SimpleNamespace())
    return supervisor, manager


class TestSupervisor:
    def test_budget_exhaustion_goes_dead_and_sets_fatal(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._budgets["0"] = RestartBudget(max_restarts=1, window_s=3600.0, cooldown_s=0.0)
        supervisor._budgets["0"].try_acquire()
        cm = _manager_half_object()
        supervisor.attach_checkpoint_manager(cm)
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))
        assert supervisor.get_state("0") == ReplicaRuntimeState.DEAD
        assert cm._fatal_error is not None

    def test_false_alarm_returns_to_serving(self):
        supervisor, manager = _supervisor_with_fake_manager()

        class _ProbeReplica:
            replica_rank = 0
            name_suffix = ""
            servers = [SimpleNamespace()]
            workers = []

            def __init__(self):
                self.probe_count = 0

            async def probe_once(self):
                from verl.workers.rollout.restart import FaultEvidence

                self.probe_count += 1
                if self.probe_count == 1:
                    return FaultEvidence(source="sweep", error_class="probe_timeout")
                return None

        replica = _ProbeReplica()
        supervisor._probe_replica = lambda r: r.probe_once()
        asyncio.run(supervisor._probe_and_classify(replica, "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING
        assert replica.probe_count == 2

    def test_double_failure_confirms(self):
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        replica = _DeadReplica()
        supervisor._probe_replica = lambda r: _async_value(
            __import__("verl.workers.rollout.restart", fromlist=["FaultEvidence"]).FaultEvidence(
                source="sweep", error_class="EngineDeadError"
            )
        )
        asyncio.run(supervisor._probe_and_classify(replica, "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert scheduled == ["0"]

    def test_applicability_gates(self):
        for bad in [
            dict(name="trtllm"),
            dict(nnodes=2),
            dict(disaggregation=SimpleNamespace(enabled=True)),
            dict(checkpoint_engine=SimpleNamespace(backend="delta_sharded")),
            dict(quantization=SimpleNamespace(enabled=True)),
        ]:
            base = SimpleNamespace(
                name="vllm",
                nnodes=1,
                disaggregation=SimpleNamespace(enabled=False),
                checkpoint_engine=SimpleNamespace(backend="nccl"),
                quantization=SimpleNamespace(enabled=False),
            )
            for key, value in bad.items():
                setattr(base, key, value)
            supervisor, _ = _supervisor_with_fake_manager(rollout_config=base)
            with pytest.raises(NotImplementedError):
                supervisor._validate_applicability()

    def test_unmerged_lora_rejected(self):
        rollout_config = SimpleNamespace(
            name="vllm",
            nnodes=1,
            disaggregation=SimpleNamespace(enabled=False),
            checkpoint_engine=SimpleNamespace(backend="nccl"),
            quantization=SimpleNamespace(enabled=False),
        )
        manager = SimpleNamespace(
            rollout_config=rollout_config,
            model_config=SimpleNamespace(lora_rank=16, lora={"merge": False}),
            rollout_replicas=[],
            server_addresses=[],
            config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=None)),
        )
        supervisor = ReplicaRestartSupervisor(manager=manager, config=SimpleNamespace())
        with pytest.raises(NotImplementedError):
            supervisor._validate_applicability()


async def _async_value(value):
    return value
