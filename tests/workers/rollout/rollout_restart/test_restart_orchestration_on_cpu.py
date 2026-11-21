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
"""CPU tests for the rollout-restart orchestration pieces.

RestartBudget semantics, the supervisor's single-channel polling detection /
conviction / recovery transitions (half-object pattern with a fake manager),
the checkpoint-cold-start vs deferred-weight-sync decision (weight source
rides the restart chain), the membership-authority snapshot, and the
deferred weight-sync fulfillment path.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.utils.checkpoint.locate import CheckpointUnavailable, locate_latest_valid_checkpoint
from verl.workers.rollout.fault_tolerance import (
    FaultEvidence,
    MembershipSnapshot,
    ReplicaFaultSupervisor,
    ReplicaRuntimeState,
    RestartBudget,
    WeightSourceOverride,
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
        global_load_balancer=None,
        _group_generation=0,
        _fatal_error=None,
    )
    manager.invalidate_group = lambda: manager.__setattr__("_group_generation", manager._group_generation + 1)
    manager.mark_fatal = lambda message: manager.__setattr__("_fatal_error", message)
    supervisor = ReplicaFaultSupervisor(manager=manager, config=config or SimpleNamespace())
    return supervisor, manager


class _FakeRolloutReplica:
    def __init__(self, rank, suffix=""):
        self.replica_rank = rank
        self.name_suffix = suffix
        self.workers = [SimpleNamespace()]


async def _async_value(value):
    return value


class _RecordingLoadBalancer:
    """Plain stand-in for the embedded GlobalRequestLoadBalancer: the
    supervisor calls it in-process (no .remote indirection)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def remove_server_if_current(self, server_id, expected_handle):
        self.calls.append(("remove_server_if_current", server_id, expected_handle))
        return True

    def remove_servers(self, server_ids):
        self.calls.append(("remove_servers", tuple(server_ids)))

    def add_servers(self, servers):
        self.calls.append(("add_servers", tuple(servers)))

    def clear_sticky_cache(self):
        self.calls.append(("clear_sticky_cache",))
        return {"cleared_entries": 0, "server_loads": {}}


def _stub_recovery_steps(supervisor, restart_result=None):
    """Fence the LB/register/verify edges; keep the real recovery sequencing."""
    supervisor._isolate = lambda replica, key: _async_value(None)
    supervisor._verify = lambda replica, expected_version: _async_value(None)
    supervisor._register = lambda replica, key: _async_value(None)

    async def _restart(*, weight_source=None):
        return restart_result

    return _restart


class TestSupervisor:
    def test_checkpoint_cold_start_passes_weight_source_through_the_restart_chain(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(
            WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
        )
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        replica.servers = [SimpleNamespace(set_global_steps=SimpleNamespace(remote=lambda v: _async_value(None)))]
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        # _register is faked (the real one owns the SERVING flip), so VERIFYING
        # proves the checkpoint cold-start walk: restart(weight_source) -> stamp -> verify.
        assert supervisor.get_state("0") == ReplicaRuntimeState.VERIFYING
        # Fencing generation bumped at conviction.
        assert manager._group_generation == 1

    def test_cold_start_verify_failure_falls_back_to_deferred_sync(self):
        """The rebuild is up when VERIFYING fails — the replica must fall back
        to the deferred-weight-sync semantics (COLD_START), not stay ISOLATED
        (a healthy probe on an ISOLATED replica is never re-convicted, so it
        would be fenced forever)."""
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(
            WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
        )
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        replica.servers = [SimpleNamespace(set_global_steps=SimpleNamespace(remote=lambda v: _async_value(None)))]
        manager.rollout_replicas = [replica]

        async def _failing_verify(replica, expected_version):
            raise RuntimeError("not healthy yet")

        supervisor._verify = _failing_verify

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        # The pulled membership flags it as deferred: the next sync installs
        # weights and the fulfillment callback finishes recovery.
        assert supervisor.get_membership().deferred_keys == {"0"}

    def test_deferred_weight_sync_relays_empty_and_waits_for_sync(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(None)
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        # The pulled membership flags the deferred-sync replica as deferred.
        snapshot = supervisor.get_membership()
        assert snapshot.deferred_keys == {"0"}
        assert snapshot.replicas == [replica]
        assert snapshot.group_generation == 1
        assert snapshot.fatal_error is None

    def test_budget_exhaustion_goes_dead_and_marks_fatal(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._budgets["0"] = RestartBudget(max_restarts=1, window_s=3600.0, cooldown_s=0.0)
        supervisor._budgets["0"].try_acquire()
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.DEAD
        # Fatal lives on the manager (the membership authority) — the
        # pulled snapshot propagates it to the checkpoint manager.
        assert manager._fatal_error is not None
        snapshot = supervisor.get_membership()
        assert snapshot.fatal_error is not None
        # DEAD replicas are fenced out of the sync group entirely.
        assert snapshot.replicas == []
        assert snapshot.deferred_keys == set()

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
                self.probe_count += 1
                if self.probe_count == 1:
                    return FaultEvidence(source="detect_faults", error_class="probe_timeout")
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
            FaultEvidence(source="detect_faults", error_class="EngineDeadError")
        )
        asyncio.run(supervisor._probe_and_classify(replica, "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert scheduled == ["0"]

    def test_gcs_dead_convicts_without_reprobe(self):
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        probe_calls = []

        async def _probe(replica):
            probe_calls.append(1)
            return FaultEvidence(source="actor_state", error_class="ActorGCSDead")

        supervisor._probe_replica = _probe
        asyncio.run(supervisor._probe_and_classify(_DeadReplica(), "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert len(probe_calls) == 1  # authoritative: no re-verification

    def test_polling_round_convicts_a_cold_start_replica(self):
        """Polling backstop: a deferred-sync replica that dies while waiting for the next
        weight sync is convicted by the poll — the sync-tolerance path reports
        nothing by design."""
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        replica = _DeadReplica()
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START
        supervisor._probe_replica = lambda r: _async_value(
            FaultEvidence(source="detect_faults", error_class="EngineDeadError")
        )

        asyncio.run(supervisor.detect_faults())
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert scheduled == ["0"]

    def test_polling_round_skips_transitional_states(self):
        supervisor, manager = _supervisor_with_fake_manager()

        class _SilentReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

            def boom(self):
                raise AssertionError("probing a driven state must not happen")

        replica = _SilentReplica()
        manager.rollout_replicas = [replica]
        for state in (ReplicaRuntimeState.RESTARTING, ReplicaRuntimeState.VERIFYING, ReplicaRuntimeState.DEAD):
            supervisor._states["0"] = state
            supervisor._probe_replica = lambda r: r.boom()
            asyncio.run(supervisor.detect_faults())

    def test_deferred_weight_sync_fulfilled_verifies_and_registers(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._verify = lambda replica, expected_version: _async_value(None)
        registered = []

        def _fake_register(replica, key):
            registered.append(key)
            supervisor._states[key] = ReplicaRuntimeState.SERVING  # the real _register owns this flip
            return _async_value(None)

        supervisor._register = _fake_register
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START

        asyncio.run(supervisor.deferred_weight_sync_fulfilled("0"))

        assert registered == ["0"]
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING
        # The deferred flag is derived from the state: fulfilled => not deferred.
        assert supervisor.get_membership().deferred_keys == set()

    def test_deferred_weight_sync_fulfilled_stale_notification_is_noop(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._verify = lambda replica, expected_version: _async_value(None)
        registered = []
        supervisor._register = lambda replica, key: registered.append(key) or _async_value(None)
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.SERVING

        asyncio.run(supervisor.deferred_weight_sync_fulfilled("0"))

        assert registered == []
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING

    def test_deferred_weight_sync_fulfilled_verification_failure_retries_next_sync(self):
        supervisor, manager = _supervisor_with_fake_manager()

        async def _failing_verify(replica, expected_version):
            raise RuntimeError("not healthy yet")

        supervisor._verify = _failing_verify
        registered = []
        supervisor._register = lambda replica, key: registered.append(key) or _async_value(None)
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START

        asyncio.run(supervisor.deferred_weight_sync_fulfilled("0"))

        assert registered == []
        # Back to COLD_START: the next sync pulls it as deferred again — the
        # derived state self-heals the fulfillment loop.
        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        assert supervisor.get_membership().deferred_keys == {"0"}

    def test_isolate_and_register_call_the_embedded_balancer_in_process(self):
        supervisor, manager = _supervisor_with_fake_manager()
        lb = _RecordingLoadBalancer()
        manager.global_load_balancer = lb
        replica = _FakeRolloutReplica(0)
        replica._server_address = "addr-0"
        replica._server_handle = "handle-0"
        replica.server_address = "addr-0"
        replica.server_handle = "handle-0"
        replica.abort_all_requests = lambda **kwargs: _async_value(None)

        asyncio.run(supervisor._isolate(replica, "0"))
        asyncio.run(supervisor._register(replica, "0"))

        assert lb.calls == [
            ("remove_server_if_current", "addr-0", "handle-0"),
            ("add_servers", ("addr-0",)),
            ("clear_sticky_cache",),
        ]
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING

    def test_membership_snapshot_filters_non_syncable_states(self):
        supervisor, manager = _supervisor_with_fake_manager()
        replicas = [_FakeRolloutReplica(0), _FakeRolloutReplica(1), _FakeRolloutReplica(2), _FakeRolloutReplica(3)]
        manager.rollout_replicas = replicas
        manager._group_generation = 2
        supervisor._states = {
            "0": ReplicaRuntimeState.SERVING,
            "1": ReplicaRuntimeState.COLD_START,
            "2": ReplicaRuntimeState.ISOLATED,
            "3": ReplicaRuntimeState.DEAD,
        }

        snapshot = supervisor.get_membership()
        assert snapshot.replicas == [replicas[0], replicas[1]]
        assert snapshot.deferred_keys == {"1"}
        assert snapshot.group_generation == 2
        assert isinstance(snapshot, MembershipSnapshot)

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
        supervisor, manager = _supervisor_with_fake_manager()
        manager.model_config = SimpleNamespace(lora_rank=16, lora={"merge": False})
        with pytest.raises(NotImplementedError):
            supervisor._validate_applicability()


# ── Checkpoint locator (cold-start discovery) ───────────────────────────────────


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
        with pytest.raises(CheckpointUnavailable):
            locate_latest_valid_checkpoint(str(tmp_path / "does-not-exist"))


# ── CheckpointEngineManager survival surface (half-object pattern) ─────────────


def _manager_half_object():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._sync_survival_enabled = False
    manager._pre_sync_hook = None
    manager._deferred_resume_callback = None
    manager._deferred_resumes = set()
    manager._group_generation = 0
    manager._fatal_error = None
    return manager


class _FakeReplicaForCM:
    def __init__(self, rank, suffix=""):
        self.replica_rank = rank
        self.name_suffix = suffix
        self.workers = [SimpleNamespace()]


class TestManagerFencing:
    def test_replica_key_includes_suffix(self):
        replica = _FakeReplicaForCM(3, "_reward")
        assert CheckpointEngineManager._replica_key(replica) == "3_reward"

    def test_sync_membership_adopts_the_authoritative_snapshot(self):
        manager = _manager_half_object()
        manager.enable_sync_survival()
        replicas = [_FakeReplicaForCM(0), _FakeReplicaForCM(1)]
        snapshot = MembershipSnapshot(
            replicas=replicas,
            deferred_keys={"1"},
            group_generation=4,
            fatal_error=None,
        )

        manager.sync_membership(snapshot)

        assert manager.replicas == replicas
        assert manager._deferred_resumes == {"1"}
        assert manager._group_generation == 4
        assert manager._fatal_error is None

        # A later snapshot replaces the earlier one wholesale (derived state).
        fresh = MembershipSnapshot(replicas=[replicas[0]], deferred_keys=set(), group_generation=5, fatal_error="dead")
        manager.sync_membership(fresh)
        assert manager.replicas == [replicas[0]]
        assert manager._deferred_resumes == set()
        assert manager._group_generation == 5
        assert manager._fatal_error == "dead"

    def test_classify_commit_failures(self):
        manager = _manager_half_object()
        replica_a, replica_b = _FakeReplicaForCM(0), _FakeReplicaForCM(1)
        slices = [(replica_a, 0, 1), (replica_b, 1, 2)]
        # replica A's worker raised; replica B completed.
        failed, stuck = manager._classify_commit_failures([ValueError("dead")], [True], slices)
        assert failed == [replica_a]
        assert stuck == []
        # replica B's worker never finished within the bounded abort (stuck collective).
        failed, stuck = manager._classify_commit_failures([None], [False], slices)
        assert failed == []
        assert stuck == replica_b.workers

    def test_guarded_resume_fires_deferred_callback(self):
        manager = _manager_half_object()
        manager.enable_sync_survival()
        callback_keys = []
        manager._deferred_resume_callback = callback_keys.append
        replica = _FakeReplicaForCM(0)

        class _Resumes:
            calls = 0

            async def resume_kv_cache(self):
                _Resumes.calls += 1

            async def resume_generation(self):
                _Resumes.calls += 1

        replica.resume_kv_cache = _Resumes().resume_kv_cache
        replica.resume_generation = _Resumes().resume_generation
        manager._deferred_resumes = {"0"}
        asyncio.run(manager._guarded_resume([replica]))
        assert callback_keys == ["0"]
        assert manager._deferred_resumes == set()

    def test_update_weights_raises_on_fatal(self):
        manager = _manager_half_object()
        manager.enable_sync_survival()
        manager.mark_fatal("all rollout replicas are DEAD")

        async def scenario():
            try:
                await manager.update_weights(1)
                raised = None
            except RuntimeError as exc:
                raised = exc
            assert raised is not None and "fatal" in str(raised)

        asyncio.run(scenario())

    def test_update_weights_runs_pre_sync_reconcile_first(self):
        """The pre-sync hook (the membership pull) runs before anything else and
        its snapshot is what the sync reads; a hook failure is non-fatal."""
        manager = _manager_half_object()
        manager.enable_sync_survival()
        manager.backend = "naive"
        manager.actor_wg = SimpleNamespace(
            update_weights=lambda **kwargs: [_async_value({"x": 1})],
        )
        pulls = []

        async def _reconcile():
            pulls.append(1)
            replica = _FakeReplicaForCM(0)
            manager.sync_membership(
                MembershipSnapshot(replicas=[replica], deferred_keys=set(), group_generation=1, fatal_error=None)
            )

        manager._pre_sync_hook = _reconcile

        manager.update_weights(1)  # @auto_await: sync call runs the coroutine
        assert pulls == [1]
        assert manager._group_generation == 1
