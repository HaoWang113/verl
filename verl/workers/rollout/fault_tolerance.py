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
"""Rollout-side independent restart: engine-agnostic lifecycle core.

Importable without vllm/sglang/torch, so its semantics stay CPU-testable:
error taxonomy, the frozen evidence/weight-source dataclasses, owned
engine-process accounting with the verified cleanup ladder, the
``(pid, create_time)`` identity probe, the per-replica restart budget, the
replica runtime state machine, and the polling fault supervisor the
actorized manager drives. Nothing here runs unless
``rollout.restart.enabled`` is true.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


# ── Internal tunables (exposed knobs live in RolloutFaultToleranceConfig) ──────────

# check_health probe timeout (detection probes, VERIFYING gate).
PROBE_TIMEOUT_S = 10.0

# Upper bound for the checkpoint cold-start locate (a hung shared FS must not stall the cold start).
_LOCATE_TIMEOUT_S = 30.0


# ── Error taxonomy ───────────────────────────────────────────────────────────


class ReplicaRestartError(RuntimeError):
    """Restart primitive failed (cleanup/fence/probe). Fail-closed: keep ISOLATED."""


# ── Frozen data structures ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultEvidence:
    """One observation of a (suspected) fault; frozen so it is safe across retries."""

    source: str
    """Where the evidence came from: "detect_faults" | "actor_state"."""

    error_class: str
    """Normalized failure class, e.g. "EngineDeadError" | "ActorDiedError" | "check_health_failed"."""


@dataclass(frozen=True)
class WeightSourceOverride:
    """Checkpoint cold start: the replacement engine cold-loads this checkpoint HF export at
    startup; its version stamp is set to ``global_steps``.

    Rides the restart chain explicitly (``replica.restart(weight_source=...)``
    -> ``launch_servers(weight_source=...)`` -> server constructor).
    """

    model_path: str
    global_steps: Optional[int] = None


# ── Process identity probe (witness of the teardown) ──────────────────────────


def verify_process_dead(pid: int, create_time: float) -> bool:
    """Whether the process identified by ``(pid, create_time)`` is dead.

    ``create_time`` pins the identity so a recycled PID cannot impersonate the
    original; a zombie counts as dead. Used by the teardown to double-confirm
    a Ray actor kill actually reached the OS (GCS DEAD can precede OS death).
    """
    try:
        proc = psutil.Process(int(pid))
    except (psutil.NoSuchProcess, ValueError):
        return True
    try:
        if abs(proc.create_time() - float(create_time)) > 1e-6:
            return True  # PID was recycled; the original process is gone.
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


# ── Owned engine-process accounting ───────────────────────────────────────────


class EngineProcessAccounting:
    """Track — and verified-cleanup — the engine-owned child processes of one
    server actor. The construction-time baseline excludes pre-existing children,
    so only processes spawned by *this* actor's engine are owned.

    :meth:`shutdown_owned` is the verified cleanup ladder: bounded wait ->
    SIGTERM -> SIGKILL -> any survivor raises (fail-closed: a surviving engine
    would fight its replacement for the same devices).
    """

    def __init__(self):
        self._pid = os.getpid()
        self._baseline = {child.pid for child in self._snapshot_children()}
        self._owned: list[psutil.Process] = []

    def _snapshot_children(self) -> list[psutil.Process]:
        try:
            return psutil.Process(self._pid).children(recursive=False)
        except psutil.Error:
            return []

    def record(self) -> list[psutil.Process]:
        """Refresh the owned set: current direct children minus the baseline."""
        current = {child.pid: child for child in self._snapshot_children()}
        self._owned = [child for pid, child in current.items() if pid not in self._baseline]
        return list(self._owned)

    @property
    def owned_pids(self) -> list[int]:
        return [proc.pid for proc in self._owned]

    @staticmethod
    def _is_running_alive(proc: psutil.Process) -> bool:
        try:
            return proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def shutdown_owned(self, wait_s: float = 5.0, term_s: float = 2.0, kill_s: float = 2.0) -> None:
        """Bounded wait -> SIGTERM -> SIGKILL over the owned children; any
        survivor raises (the engine's own graceful shutdown is the caller's
        business and may fail without blocking this ladder)."""
        owned = [proc for proc in self._owned if self._is_running_alive(proc)]
        if not owned:
            return

        _, alive = psutil.wait_procs(owned, timeout=wait_s)
        if alive:
            for proc in alive:
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=term_s)
        if alive:
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=kill_s)

        survivors = [proc for proc in alive if self._is_running_alive(proc)]
        if survivors:
            raise ReplicaRestartError(
                f"{len(survivors)} owned engine process(es) survived the cleanup ladder "
                f"(pids={[p.pid for p in survivors]}); refusing to restart — a surviving "
                "engine would contend with its replacement for the same devices"
            )


# ── Restart budget (anti-flapping) ────────────────────────────────────────────


class RestartBudget:
    """Per-replica anti-flapping budget: sliding-window count plus a forced
    cooldown so a crash-looping engine cannot burn it in one burst."""

    def __init__(self, max_restarts: int = 3, window_s: float = 3600.0, cooldown_s: float = 30.0):
        self.max_restarts = max_restarts
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._events: list[float] = []

    def try_acquire(self) -> bool:
        now = time.monotonic()
        self._events = [ts for ts in self._events if now - ts <= self.window_s]
        if self._events and now - self._events[-1] < self.cooldown_s:
            return False
        if len(self._events) >= self.max_restarts:
            return False
        self._events.append(now)
        return True


# ── Replica runtime state machine ─────────────────────────────────────────────


class ReplicaRuntimeState(str, Enum):
    SERVING = "serving"
    ISOLATED = "isolated"
    RESTARTING = "restarting"
    COLD_START = "cold_start"
    VERIFYING = "verifying"
    DEAD = "dead"


def _get_actor_state_dict(handle) -> Optional[dict]:
    """Actor state via the Ray GCS state API (the ``_is_worker_alive`` pattern)."""
    try:
        from ray.experimental.state.api import get_actor

        return get_actor(handle._actor_id.hex())
    except Exception:
        return None


def _unwrap_ray_exception(exc: BaseException) -> BaseException:
    """Ray wraps task exceptions; classify on the innermost cause."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "__cause__", None)
        if inner is None:
            inner = getattr(current, "cause", None)
        if inner is None:
            return current
        current = inner
    return exc


# ── Membership snapshot (pull-reconcile payload) ──────────────────────────────


@dataclass(frozen=True)
class MembershipSnapshot:
    """Authoritative membership snapshot pulled by the driver-side reconcile at
    every ``update_weights`` entry: the manager is the single source of truth
    and the checkpoint manager's bookkeeping is a derived copy.
    """

    replicas: list
    """Syncable replicas (SERVING / COLD_START) — the weight-sync group members;
    everything else is fenced out of the group."""

    deferred_keys: set[str]
    """Replicas awaiting their first post-restart weight sync — staged empty
    start, deferred to the next sync (state == COLD_START)."""

    group_generation: int
    """Fencing generation: bumped at every conviction; the next sync builds
    under the derived name."""

    fatal_error: Optional[str]
    """Not None when every replica is DEAD — the sync pipeline fails closed."""


# ── Supervisor ─────────────────────────────────────────────────────────────────


class ReplicaFaultSupervisor:
    """Detection + orchestration state machine, owned by the actorized
    ``LLMServerManager``.

    The manager actor's event loop drives everything: ``start()`` spawns the
    periodic polling task (interval ``probe_interval_s``), each round
    ``detect_faults()`` probes every registered replica; recovery runs as
    on-demand asyncio tasks on the same loop, so a restart never blocks the
    manager's other RPCs. No resident thread, no client->manager report
    channel — the poll is the single detection path.
    """

    def __init__(
        self,
        manager: Any,
        config: Any,
    ):
        self._manager = manager
        self._config = config
        self._states: dict[str, ReplicaRuntimeState] = {}
        self._budgets: dict[str, RestartBudget] = {}
        self._recovering: set[str] = set()
        self._recovery_tasks: set[asyncio.Task] = set()  # strong refs so rebuilds are not GC-ed
        self._poll_task: asyncio.Task | None = None
        self._stopped = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Validate applicability, then start the periodic polling task on the
        manager actor's event loop (must be called from a coroutine context)."""
        self._validate_applicability()
        self._poll_task = asyncio.create_task(self._poll_loop())

    def stop(self) -> None:
        """Idempotent shutdown: cancel the polling task. In-flight rebuilds are
        left to finish — tearing one down mid-restart is worse than letting it
        complete."""
        self._stopped = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        interval = float(getattr(self._config, "probe_interval_s", 10.0))
        while not self._stopped:
            await asyncio.sleep(interval)
            try:
                await self.detect_faults()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Same discipline as every detection round: a broken round must
                # never take the supervisor (or the manager actor) down.
                logger.warning("fault detection round failed (non-fatal)", exc_info=True)

    # ── detection ───────────────────────────────────────────────────────

    async def detect_faults(self) -> None:
        """One polling round: probe every registered replica concurrently and
        reconcile runtime states. Never raises: a broken detection channel
        must not break the sync or the actor."""
        replicas = list(getattr(self._manager, "rollout_replicas", []) or [])
        await asyncio.gather(*(self._probe_and_classify(replica, self._replica_key(replica)) for replica in replicas))

    async def _probe_replica(self, replica) -> Optional[FaultEvidence]:
        """One health pass: GCS actor states (authoritative) + engine probe."""
        actors = list(getattr(replica, "servers", []) or []) + list(getattr(replica, "workers", []) or [])
        for actor in actors:
            # Blocking GCS query — keep it off the manager actor's event loop.
            state_dict = await asyncio.to_thread(_get_actor_state_dict, actor)
            if state_dict is not None and state_dict.get("state") != "ALIVE":
                return FaultEvidence(source="actor_state", error_class="ActorGCSDead")
        try:
            await asyncio.wait_for(replica.check_health(), timeout=PROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return FaultEvidence(source="detect_faults", error_class="probe_timeout")
        except Exception as exc:
            inner = _unwrap_ray_exception(exc)
            return FaultEvidence(source="detect_faults", error_class=type(inner).__name__)
        return None

    async def _probe_and_classify(self, replica, key: str) -> None:
        state = self._states.get(key, ReplicaRuntimeState.SERVING)
        if state in (ReplicaRuntimeState.RESTARTING, ReplicaRuntimeState.VERIFYING, ReplicaRuntimeState.DEAD):
            # RESTARTING / VERIFYING are driven by their own recovery task; DEAD is terminal.
            return
        if state == ReplicaRuntimeState.ISOLATED:
            # A restart that failed left the replica ISOLATED with "retry on next
            # evidence" — a polling round IS that next evidence.
            evidence = await self._probe_replica(replica)
            if evidence is None:
                logger.info("replica %s: ISOLATED but probe now healthy; leaving state unchanged", key)
                return
            self._on_confirmed(replica, key, evidence)
            return
        # SERVING, or COLD_START (deferred weight sync: waiting for the next
        # weight sync — if it died meanwhile, the poll is the backstop that
        # convicts it, since the sync-tolerance path deliberately reports
        # nothing).
        evidence = await self._probe_replica(replica)
        if evidence is None:
            return
        if evidence.error_class == "ActorGCSDead":
            # Ray GCS death is authoritative — no re-verification.
            self._on_confirmed(replica, key, evidence)
            return
        reprobe = await self._probe_replica(replica)
        if reprobe is None:
            logger.info("replica %s: first probe failure %s was a transient false alarm", key, evidence.error_class)
            return
        self._on_confirmed(replica, key, reprobe)

    # ── orchestration ───────────────────────────────────────────────────

    def _on_confirmed(self, replica, key: str, evidence: FaultEvidence) -> None:
        logger.warning(
            "replica %s confirmed dead (source=%s, error=%s) — scheduling recovery",
            key,
            evidence.source,
            evidence.error_class,
        )
        self._states[key] = ReplicaRuntimeState.ISOLATED
        self._schedule_recovery(replica, key, evidence)

    def _schedule_recovery(self, replica, key: str, evidence: FaultEvidence) -> None:
        if self._stopped:
            return
        if key in self._recovering:
            logger.info("recovery already in flight for replica %s", key)
            return
        self._recovering.add(key)
        task = asyncio.create_task(self._recover_replica(replica, key, evidence))
        self._recovery_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            self._recovering.discard(key)
            self._recovery_tasks.discard(finished)

        task.add_done_callback(_done)

    async def _recover_replica(self, replica, key: str, evidence: FaultEvidence) -> None:
        try:
            await self._handle_confirmed_fault(replica, key, evidence)
        except Exception:
            logger.exception("recovery pipeline failed for replica %s; staying ISOLATED", key)
            self._states[key] = ReplicaRuntimeState.ISOLATED

    async def _handle_confirmed_fault(self, replica, key: str, evidence: FaultEvidence) -> None:
        """ISOLATED -> (budget) -> restart -> weight-source pick -> VERIFYING -> SERVING/DEAD."""
        # Conviction fences, before any budget decision: the dead replica must
        # leave the LB, and the weight-sync group generation must be bumped so
        # the next sync rebuilds under the derived name — both also hold on the
        # budget-exhausted DEAD path, where no restart follows.
        await self._isolate(replica, key)
        self._manager.invalidate_group()

        budget = self._budgets.get(key)
        if budget is None:
            budget = RestartBudget(max_restarts=int(getattr(self._config, "max_restarts", 3)))
            self._budgets[key] = budget
        if not budget.try_acquire():
            logger.error("restart budget exhausted for replica %s — going DEAD", key)
            self._states[key] = ReplicaRuntimeState.DEAD
            self._check_all_dead()
            return

        self._states[key] = ReplicaRuntimeState.RESTARTING
        override = await self._plan_weight_source()
        try:
            await replica.restart(weight_source=override)
        except Exception:
            logger.exception("restart failed for replica %s; staying ISOLATED (retry on next polling round)", key)
            self._states[key] = ReplicaRuntimeState.ISOLATED
            return

        self._states[key] = ReplicaRuntimeState.COLD_START

        if override is None:
            # Deferred weight sync: staged empty start, wait for the next
            # weight sync — the pulled membership flags this replica as
            # deferred and the sync's deferred-resume callback fires
            # deferred_weight_sync_fulfilled once the weights are installed.
            logger.warning(
                "replica %s relaunched without a checkpoint weight source (deferred to the next weight sync)", key
            )
            return

        # Checkpoint cold start: stamp the version, then VERIFY -> register. A
        # failure here falls back to the deferred-weight-sync semantics — the
        # rebuild is up, so stranding it ISOLATED would fence a healthy
        # replica forever (an ISOLATED replica with a healthy probe is never
        # re-convicted); as COLD_START the next sync installs weights and the
        # fulfillment callback finishes recovery.
        try:
            new_server = replica.servers[0]
            await new_server.set_global_steps.remote(override.global_steps)
            self._states[key] = ReplicaRuntimeState.VERIFYING
            await self._verify(replica, expected_version=override.global_steps)
            await self._register(replica, key)
        except Exception:
            logger.exception("post-restart verification failed for replica %s; deferring to the next weight sync", key)
            self._states[key] = ReplicaRuntimeState.COLD_START
            return
        logger.info("replica %s recovered via checkpoint cold start at version %s", key, override.global_steps)

    async def deferred_weight_sync_fulfilled(self, replica_key: str) -> None:
        """Deferred-weight-sync fulfillment (the checkpoint manager's fire-and-forget RPC
        after installing a deferred replica's weights): VERIFY (health-only —
        the sync stamped the version) -> register. A stale notification (state
        already moved on) is a no-op."""
        replica = self._replica_by_key(replica_key)
        if replica is None:
            return
        if self._states.get(replica_key) != ReplicaRuntimeState.COLD_START:
            logger.info(
                "deferred weight-sync fulfillment for replica %s ignored (state=%s)",
                replica_key,
                self._states.get(replica_key),
            )
            return
        self._states[replica_key] = ReplicaRuntimeState.VERIFYING
        try:
            await self._verify(replica, expected_version=None)
        except Exception:
            logger.exception("deferred weight-sync verification failed for replica %s", replica_key)
            # Back to COLD_START: the next sync pulls it as deferred again and
            # fulfillment retries — the derived deferred state self-heals the loop.
            self._states[replica_key] = ReplicaRuntimeState.COLD_START
            return
        await self._register(replica, replica_key)
        logger.info("replica %s recovered via deferred weight sync", replica_key)

    async def _isolate(self, replica, key: str) -> None:
        """LB CAS fence + best-effort abort of the dead replica.

        The balancer is embedded in the manager actor (the supervisor's host),
        so these are in-process calls — no cross-actor RPC."""
        lb = self._manager.global_load_balancer
        address = getattr(replica, "_server_address", None)
        handle = getattr(replica, "_server_handle", None)
        if address is not None:
            try:
                lb.remove_server_if_current(address, handle)
            except Exception:
                logger.warning("CAS removal unavailable; falling back to remove_servers", exc_info=True)
                try:
                    lb.remove_servers([address])
                except Exception:
                    logger.warning("remove_servers fallback failed (non-fatal)", exc_info=True)
        try:
            await asyncio.wait_for(replica.abort_all_requests(reject_request=True), timeout=PROBE_TIMEOUT_S)
        except Exception:
            # Expected when the actor is already gone — best-effort by design.
            logger.debug("best-effort abort on dead replica %s failed (expected)", key, exc_info=True)

    async def _plan_weight_source(self) -> Optional[WeightSourceOverride]:
        """Weight-source planning: locate a valid checkpoint for a cold start,
        else defer the replica to the next weight sync.

        The locate runs in the manager with the full config, so the discovery
        side needs no parameter threading; the result rides the restart
        chain explicitly."""
        if str(getattr(self._config, "weight_source", "checkpoint")) != "checkpoint":
            return None
        trainer_cfg = getattr(getattr(self._manager, "config", None), "trainer", None)
        checkpoint_dir = getattr(trainer_cfg, "default_local_dir", None)
        if not checkpoint_dir:
            logger.warning(
                "no checkpoint root for the cold start (trainer.default_local_dir unset) "
                "— deferring to the next weight sync"
            )
            return None
        checkpoint_dir = str(checkpoint_dir)
        from verl.utils.checkpoint.locate import locate_latest_valid_checkpoint

        try:
            model_path, global_steps = await asyncio.wait_for(
                asyncio.to_thread(locate_latest_valid_checkpoint, checkpoint_dir),
                timeout=_LOCATE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "checkpoint locate timed out after %ss — deferring to the next weight sync", _LOCATE_TIMEOUT_S
            )
            return None
        except Exception as exc:
            logger.warning(
                "checkpoint cold start unavailable (%s: %s) — deferring to the next weight sync",
                type(exc).__name__,
                exc,
            )
            return None
        logger.info("checkpoint cold start from %s (version %s)", model_path, global_steps)
        return WeightSourceOverride(model_path=model_path, global_steps=global_steps)

    async def _verify(self, replica, expected_version: Optional[int]) -> None:
        """VERIFYING: health probe + version stamp check."""
        await asyncio.wait_for(replica.check_health(), timeout=PROBE_TIMEOUT_S)
        if expected_version is not None:
            server = replica.servers[0]
            actual = await server.get_global_steps.remote()
            if actual != expected_version:
                raise ReplicaRestartError(
                    f"version stamp mismatch after restart: expected {expected_version}, got {actual}"
                )

    async def _register(self, replica, key: str) -> None:
        """Re-register with the LB — strictly the last step of recovery.
        In-process calls into the embedded balancer (same actor)."""
        lb = self._manager.global_load_balancer
        lb.add_servers({replica.server_address: replica.server_handle})
        lb.clear_sticky_cache()
        self._states[key] = ReplicaRuntimeState.SERVING

    def _check_all_dead(self) -> None:
        replica_keys = {self._replica_key(r) for r in getattr(self._manager, "rollout_replicas", []) or []}
        dead_keys = {k for k, s in self._states.items() if s == ReplicaRuntimeState.DEAD}
        if replica_keys and dead_keys >= replica_keys:
            message = "all rollout replicas are DEAD"
            logger.critical(message)
            # Fatal lives on the manager (the membership authority); the pulled
            # snapshot propagates it to the checkpoint manager, whose next
            # update_weights fails closed.
            self._manager.mark_fatal(message)

    # ── membership authority ────────────────────────────────────────────

    def get_membership(self) -> MembershipSnapshot:
        """Authoritative membership snapshot for the pull reconcile:
        syncable replicas (SERVING / COLD_START — everything else is fenced
        out of the weight-sync group), the deferred keys (replicas awaiting
        their first post-restart weight sync), the group generation, and
        the fatal flag."""
        replicas = []
        deferred = set()
        for replica in getattr(self._manager, "rollout_replicas", []) or []:
            key = self._replica_key(replica)
            state = self._states.get(key, ReplicaRuntimeState.SERVING)
            if state in (ReplicaRuntimeState.SERVING, ReplicaRuntimeState.COLD_START):
                replicas.append(replica)
                if state == ReplicaRuntimeState.COLD_START:
                    deferred.add(key)
        return MembershipSnapshot(
            replicas=replicas,
            deferred_keys=deferred,
            group_generation=getattr(self._manager, "_group_generation", 0),
            fatal_error=getattr(self._manager, "_fatal_error", None),
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _replica_key(replica) -> str:
        return f"{replica.replica_rank}{getattr(replica, 'name_suffix', '') or ''}"

    def _replica_by_key(self, key: str):
        for replica in getattr(self._manager, "rollout_replicas", []) or []:
            if self._replica_key(replica) == key:
                return replica
        return None

    def _validate_applicability(self) -> None:
        """Fail-loud applicability gate."""
        rollout_cfg = self._manager.rollout_config
        disaggregation = getattr(rollout_cfg, "disaggregation", None)
        if disaggregation is not None and getattr(disaggregation, "enabled", False):
            raise NotImplementedError("rollout.restart is not supported with PD disaggregation")
        rollout_name = str(getattr(rollout_cfg, "name", ""))
        if rollout_name != "vllm":
            # SGLang ships in a follow-up PR: no check_health/shutdown yet, so
            # detection would mis-convict healthy replicas.
            raise NotImplementedError(f"rollout.restart does not support rollout.name={rollout_name!r} yet")
        if int(getattr(rollout_cfg, "nnodes", 1)) != 1:
            raise NotImplementedError("rollout.restart requires a single-node replica (nnodes == 1)")
        model_cfg = getattr(self._manager, "model_config", None)
        lora_rank = int(getattr(model_cfg, "lora_rank", 0) or 0)
        lora_merged = bool(getattr(getattr(model_cfg, "lora", None), "get", lambda *_: False)("merge", False))
        if lora_rank > 0 and not lora_merged:
            raise NotImplementedError("rollout.restart requires merged full weights (unmerged LoRA is Phase 2)")
        engine_cfg = getattr(rollout_cfg, "checkpoint_engine", None)
        if engine_cfg is not None and str(getattr(engine_cfg, "backend", "")) == "delta_sharded":
            raise NotImplementedError("rollout.restart is not supported with checkpoint_engine.backend=delta_sharded")
        quantization = getattr(rollout_cfg, "quantization", None)
        if quantization is not None and getattr(quantization, "enabled", False):
            raise NotImplementedError(
                "rollout.restart does not support quantized rollout in Phase 1 (refit × HF cold start unverified)"
            )

    # ── observability ───────────────────────────────────────────────────

    def get_state(self, replica_key: str) -> ReplicaRuntimeState:
        return self._states.get(replica_key, ReplicaRuntimeState.SERVING)


# ── Driver-side reconcile helpers (wired by the trainer) ──────────────────────


def make_pre_sync_reconcile(checkpoint_manager, manager_handle):
    """The checkpoint manager's pre-sync hook: pull the manager's authoritative
    membership snapshot and write it into the checkpoint manager
    (``sync_membership``). The checkpoint engine's membership is a derived
    copy, refreshed at the only boundary that reads it."""

    async def _reconcile() -> None:
        snapshot = await manager_handle.get_membership.remote()
        checkpoint_manager.sync_membership(snapshot)

    return _reconcile


def make_deferred_resume_callback(manager_handle):
    """The checkpoint manager's deferred-resume callback: fire-and-forget RPC
    into the manager actor once a deferred replica's weights are installed —
    the replica walks VERIFYING -> re-registration. Fire-and-forget by
    design: a lost notification is retried by the next sync (the deferred flag
    is derived from the pulled state)."""

    def _fulfill(replica_key: str) -> None:
        manager_handle.deferred_weight_sync_fulfilled.remote(replica_key)

    return _fulfill
