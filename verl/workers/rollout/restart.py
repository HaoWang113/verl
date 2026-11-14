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
"""Rollout-side independent restart (Phase 1): engine-agnostic lifecycle core.

This module hosts the pieces of the rollout restart design
(reports/rollout_restart_design_v1.md) that must stay importable without
vllm/sglang/torch so their semantics are CPU-testable:

- error taxonomy (design §7.1) shared by server / replica / supervisor / client;
- frozen evidence and weight-source dataclasses (§7.1);
- owned engine-process accounting with the verified cleanup ladder, ported
  from #8168's ``_LegacyMPEngineCleanup`` (§7.2);
- the ``(pid, create_time)`` process-identity probe used as the witness in the
  five-segment teardown (§6.1);
- ``engine_failed`` progress receipt construction (§7.2) implementing the
  completeness-first rule of invariant I2: a prefix survives only when its
  per-token logprobs are complete, otherwise the whole request is replayed.

The supervision state machine (``ReplicaRestartSupervisor`` /
``ReplicaFaultMailbox``) is layered on top of this core by later commits and
keeps the same import discipline. Nothing here runs unless
``rollout.restart.enabled`` is true: the module has no import-time side
effects and existing code paths reach it only through restart-gated branches.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import psutil
import ray

logger = logging.getLogger(__name__)


# ── Error taxonomy (design §7.1, #8168 P7) ──────────────────────────────────


class ReplicaRestartError(RuntimeError):
    """Restart primitive failed (cleanup/fence/probe). Fail-closed: keep ISOLATED."""


class RestartBudgetExhausted(RuntimeError):
    """Per-replica restart budget blown. Replica -> DEAD (fatal if it was the last one)."""


class CheckpointUnavailable(FileNotFoundError):
    """Route A: no valid (manifest-passing) checkpoint under checkpoint_dir."""


class EngineShutdownError(ReplicaRestartError):
    """Owned engine processes survived the verified cleanup ladder."""


# ── Frozen data structures (design §7.1) ─────────────────────────────────────


@dataclass(frozen=True)
class FaultEvidence:
    """One observation of a (suspected) fault; frozen so it is safe across retries."""

    source: str
    """Where the evidence came from: "sweep" | "request_path" | "actor_state"."""

    error_class: str
    """Normalized failure class, e.g. "EngineDeadError" | "ActorDiedError" | "check_health_failed"."""

    detail: str = ""
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ReplicaFault:
    """A confirmed-to-investigate fault report bound to one replica."""

    replica_key: str
    evidence: FaultEvidence


@dataclass(frozen=True)
class WeightSourceOverride:
    """Route A: relaunch-time engine weight source (design §6.3).

    The replacement server's engine loads weights from ``model_path`` at
    startup (engine-native cold start; no new transfer channel), and its
    version stamp is set to ``global_steps`` (N-K of the trainer).
    """

    model_path: str
    load_format: str = "safetensors"
    global_steps: Optional[int] = None


# ── Process identity probe (design §6.1 witness, #8168 port) ─────────────────


def verify_process_dead(pid: int, create_time: float) -> bool:
    """Whether the process identified by ``(pid, create_time)`` is dead.

    ``create_time`` pins the identity so a recycled PID cannot impersonate the
    original process: a mismatch means the original is gone. A zombie counts
    as dead — its resources are released and the parent only owes a reap.

    Used by the five-segment teardown to double-confirm that a Ray actor kill
    actually reached the OS: Ray GCS can publish DEAD before the process dies.
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


# ── Owned engine-process accounting (design §7.2, #8168 _LegacyMPEngineCleanup) ─


class EngineProcessAccounting:
    """Track — and verified-cleanup — the engine-owned child processes of one server actor.

    The baseline snapshot taken at construction excludes every child that
    already existed, so only processes spawned by *this* actor's engine are
    owned. Call :meth:`record` around engine initialization (and from the
    constructor's ``finally`` on the startup-failure path) to refresh the owned
    set.

    :meth:`shutdown_owned` implements the verified cleanup ladder: bounded
    wait -> SIGTERM -> SIGKILL -> any survivor raises :class:`EngineShutdownError`
    (fail-closed: a surviving engine would fight its replacement for the same
    devices).
    """

    def __init__(self, pid: Optional[int] = None):
        self._pid = pid if pid is not None else os.getpid()
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
    def owned(self) -> list[psutil.Process]:
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
        """Run the verified cleanup ladder over the owned children (§7.2).

        The engine's own graceful shutdown is the caller's business — it may
        fail; that failure is logged by the caller and must not block this
        ladder. This method guarantees the OS-level outcome: every owned
        process is gone, or :class:`EngineShutdownError` is raised and the
        restart must not proceed.
        """
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
            raise EngineShutdownError(
                f"{len(survivors)} owned engine process(es) survived the cleanup ladder "
                f"(pids={[p.pid for p in survivors]}); refusing to restart — a surviving "
                "engine would contend with its replacement for the same devices"
            )


# ── engine_failed progress receipt (design §7.2, invariant I2) ──────────────


def build_engine_failed_receipt(global_steps: Optional[int], partial_output: Any = None):
    """Build the ``engine_failed`` progress receipt for an in-flight request.

    Completeness-first (invariant I2): the accumulated prefix is kept only when
    its per-token logprobs are complete (one plain-dict entry per token);
    anything partial or unparsable drops the whole prefix so the client
    replays from scratch — recompute beats corrupt data. ``partial_output`` is
    the vLLM ``RequestOutput`` captured when ``EngineDeadError`` surfaced, or
    ``None`` when the core died before emitting anything.
    """
    from verl.workers.rollout.replica import TokenOutput

    token_ids: list[int] = []
    log_probs: Optional[list[float]] = None

    outputs = getattr(partial_output, "outputs", None) or []
    if outputs:
        out = outputs[0]
        token_ids = list(getattr(out, "token_ids", None) or [])
        raw = getattr(out, "logprobs", None)
        if token_ids and isinstance(raw, list) and len(raw) >= len(token_ids):
            try:
                # strict=False: logprobs may legitimately trail tokens; the length
                # check above already guarantees every kept token has its entry.
                log_probs = [float(entry[token_id].logprob) for entry, token_id in zip(raw, token_ids, strict=False)]
            except (KeyError, TypeError, AttributeError, ValueError):
                log_probs = None

    if log_probs is None:
        # Incomplete or unparsable progress: drop the prefix entirely (I2).
        token_ids = []

    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs,
        routed_experts=None,
        stop_reason="engine_failed",
        num_preempted=None,
        extra_fields={"engine_failed": True, "global_steps": global_steps},
    )


# ── Restart budget (design §7.1 RestartBudget, anti-flapping) ───────────────


class RestartBudget:
    """Per-replica restart budget: sliding-window count plus a forced cooldown.

    ``try_acquire`` consumes one slot; exhaustion returns False (the caller
    sends the replica to DEAD). The cooldown separates attempts so a poisoned
    device or a crash-looping engine cannot burn the budget in one burst.
    """

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


# ── Replica runtime state machine (design §4.1) ─────────────────────────────


class ReplicaRuntimeState(str, Enum):
    SERVING = "serving"
    SUSPECTED = "suspected"
    ISOLATED = "isolated"
    RESTARTING = "restarting"
    COLD_START = "cold_start"
    VERIFYING = "verifying"
    DEAD = "dead"


# ── Passive fault mailbox (design §5.1①, v1 defect fix) ─────────────────────


@ray.remote
class ReplicaFaultMailbox:
    """Append-only mailbox for cross-process death reports.

    Clients (AgentLoopWorker processes) fire-and-forget ``report`` at it. Two
    consumers: the supervisor's resident awaiter (``wait_next`` — event-driven,
    woken by the report RPC itself, §5.1① rev 2026-11-14) and the sweep's
    drain-based reconciliation backstop (§5.1②). Handlers only append, dedup
    (O(1)) and flip an asyncio.Event, so the reporting channel can never be
    blocked by a supervisor that is busy executing a minutes-long rebuild
    (design §5.1①, the v1 flaw where a serialized detector copy silently
    dropped reports).
    """

    def __init__(self):
        self._reports: list[tuple[str, dict]] = []
        self._seen: set[tuple[str, str]] = set()
        self._event = asyncio.Event()

    async def report(self, server_id: str, evidence: dict) -> None:
        key = (server_id, str(evidence.get("error_class", "")))
        if key in self._seen:
            return
        self._seen.add(key)
        self._reports.append((server_id, dict(evidence)))
        self._event.set()

    async def wait_next(self, timeout_s: float | None = None) -> list[tuple[str, dict]]:
        """Block until at least one report is pending, then return and clear.

        Event-driven hand-off: no timer, no probing — the caller's future is
        woken by the ``report`` RPC itself. ``timeout_s=None`` waits
        indefinitely (the awaiter); a finite timeout returns ``[]`` on expiry
        (tests). Never raises. Batch consumption resets the dedup window,
        same semantics as ``drain``.
        """
        if not self._reports:
            try:
                await asyncio.wait_for(self._event.wait(), timeout_s)
            except asyncio.TimeoutError:
                return []
        reports, self._reports = self._reports, []
        self._seen = set()
        self._event.clear()
        return reports

    async def drain(self) -> list[tuple[str, dict]]:
        reports, self._reports = self._reports, []
        self._seen = set()
        self._event.clear()
        return reports


# ── Supervisor (design §4/§5/§6) ─────────────────────────────────────────────


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


class ReplicaRestartSupervisor:
    """Detection + orchestration state machine, owned by ``LLMServerManager`` (design §3.3).

    Detection is event-driven (design §5.1, 2026-11-12 decision, revised
    2026-11-14): request-path reports are consumed immediately by a resident
    awaiter that the report RPC itself wakes (no timers, no probing, zero idle
    cost); the ``update_weights`` entry sweep remains the reconciliation and
    idle-window backstop. Recovery runs as on-demand tasks on the
    supervisor's own loop so a rebuild never blocks the trainer's sync.

    ``restart.enabled=false`` means this class is never instantiated — every
    hook below is unreachable in the default scenario (zero-impact promise).
    """

    def __init__(
        self,
        manager: Any,
        config: Any,
        checkpoint_manager: Any = None,
    ):
        self._manager = manager
        self._config = config
        self._checkpoint_manager = checkpoint_manager
        self._states: dict[str, ReplicaRuntimeState] = {}
        self._budgets: dict[str, RestartBudget] = {}
        self._recovering: set[str] = set()
        self._mailbox: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._awaiter_handle: Any = None
        self._schedule_lock = threading.Lock()  # sweep (trainer loop) vs awaiter/tolerance (supervisor loop)
        self._stopped = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Validate applicability, start the on-demand loop, create the mailbox."""
        self._validate_applicability()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="rollout-restart-supervisor", daemon=True
        )
        self._loop_thread.start()
        self._mailbox = ReplicaFaultMailbox.remote()
        self._start_mailbox_awaiter()

    def recovery_in_flight(self) -> bool:
        """True while any replica is between fault confirmation and re-registration."""
        return bool(self._recovering)

    def _start_mailbox_awaiter(self) -> None:
        """Spawn the resident event-driven consumer of request-path reports (§5.1① rev).

        The awaiter blocks on the mailbox's ``wait_next`` — no timer, no probing,
        zero idle cost; the report RPC itself is the wake-up source. This is what
        makes conviction independent of the sync step (2026-11-14 ruling: a
        load-window fault must start recovery immediately, not at the next
        update_weights).
        """
        if self._mailbox is None:
            return
        self._awaiter_handle = asyncio.run_coroutine_threadsafe(self._mailbox_awaiter(), self._loop)

    async def _mailbox_awaiter(self) -> None:
        while not self._stopped:
            try:
                reports = await self._mailbox.wait_next.remote()
                await self._consume_reports(reports)
            except asyncio.CancelledError:
                return
            except Exception:
                # Awaiter discipline (§9.1-4, same as the sweep): a broken
                # detection channel must never take the supervisor down.
                logger.warning("mailbox awaiter iteration failed (non-fatal); 1s backoff", exc_info=True)
                await asyncio.sleep(1.0)

    def stop(self) -> None:
        """Idempotent shutdown: stop accepting events; in-flight rebuilds finish."""
        self._stopped = True
        if self._awaiter_handle is not None:
            self._awaiter_handle.cancel()
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    def attach_checkpoint_manager(self, checkpoint_manager: Any) -> None:
        """Backfilled by the trainer wiring after CheckpointEngineManager exists."""
        self._checkpoint_manager = checkpoint_manager

    @property
    def mailbox_handle(self):
        return self._mailbox

    # ── detection (design §5) ───────────────────────────────────────────

    async def sweep(self) -> None:
        """Full-membership verification at the update_weights entry (§5.1②).

        Never raises: a broken detection channel must not break the sync
        (sweep discipline, §9.1-4). Reconciles mailbox reports the awaiter
        has not consumed yet (backstop), probes every SERVING replica, and
        schedules recovery for confirmed faults.
        """
        try:
            reports = await self._mailbox.drain.remote() if self._mailbox is not None else []
        except Exception:
            logger.warning("mailbox drain failed (non-fatal)", exc_info=True)
            reports = []
        await self._consume_reports(reports)

        replicas = list(getattr(self._manager, "rollout_replicas", []) or [])
        replica_by_key = {self._replica_key(r): r for r in replicas}
        for key, replica in replica_by_key.items():
            await self._probe_and_classify(replica, key)

    async def _consume_reports(self, reports: list[tuple[str, dict]]) -> None:
        """Verify mailbox death reports against live membership (§5.1①).

        Shared consumer for the awaiter (immediate) and the sweep
        (reconciliation backstop); idempotent — verification and conviction
        are state-gated, so double consumption is harmless.
        """
        if not reports:
            return
        replicas = list(getattr(self._manager, "rollout_replicas", []) or [])
        addresses = list(getattr(self._manager, "server_addresses", []) or [])
        replica_by_key = {self._replica_key(r): r for r in replicas}
        key_by_address = dict(zip(addresses, (self._replica_key(r) for r in replicas), strict=True))
        for server_id, evidence in reports:
            key = key_by_address.get(server_id)
            if key is None:
                continue
            await self._verify_report(replica_by_key[key], key, evidence)

    async def _probe_replica(self, replica) -> Optional[FaultEvidence]:
        """One health pass: GCS actor states (authoritative) + engine probe."""
        for server in getattr(replica, "servers", []) or []:
            state_dict = _get_actor_state_dict(server)
            if state_dict is not None and state_dict.get("state") != "ALIVE":
                return FaultEvidence(source="actor_state", error_class="ActorGCSDead", detail="server actor")
        for worker in getattr(replica, "workers", []) or []:
            state_dict = _get_actor_state_dict(worker)
            if state_dict is not None and state_dict.get("state") != "ALIVE":
                return FaultEvidence(source="actor_state", error_class="ActorGCSDead", detail="CE worker actor")
        try:
            await asyncio.wait_for(replica.check_health(), timeout=self._probe_timeout_s())
        except asyncio.TimeoutError:
            return FaultEvidence(source="sweep", error_class="probe_timeout")
        except Exception as exc:
            inner = _unwrap_ray_exception(exc)
            return FaultEvidence(source="sweep", error_class=type(inner).__name__, detail=str(inner)[:200])
        return None

    async def _probe_and_classify(self, replica, key: str) -> None:
        """Event-axis conviction (§5.1③): first probe failure + immediate re-probe."""
        state = self._states.get(key, ReplicaRuntimeState.SERVING)
        if state == ReplicaRuntimeState.ISOLATED:
            # A rebuild that failed left the replica ISOLATED with "retry on next
            # evidence" — the sweep IS that next evidence. The fault is still
            # live, so re-confirm and schedule recovery again (budget-guarded).
            evidence = await self._probe_replica(replica)
            if evidence is None:
                logger.info("replica %s: ISOLATED but probe now healthy; leaving state unchanged", key)
                return
            self._on_confirmed(replica, key, evidence)
            return
        if state != ReplicaRuntimeState.SERVING:
            return
        evidence = await self._probe_replica(replica)
        if evidence is None:
            return
        if evidence.error_class == "ActorGCSDead":
            # Ray GCS death is authoritative — no re-verification (§5.1②).
            self._on_confirmed(replica, key, evidence)
            return
        reprobe = await self._probe_replica(replica)
        if reprobe is None:
            logger.info("replica %s: first probe failure %s was a transient false alarm", key, evidence.error_class)
            return
        self._on_confirmed(replica, key, reprobe)

    async def _verify_report(self, replica, key: str, evidence: dict) -> None:
        """Mailbox evidence gets one verification probe before conviction (§5.1①)."""
        if self._states.get(key, ReplicaRuntimeState.SERVING) != ReplicaRuntimeState.SERVING:
            return
        fresh = await self._probe_replica(replica)
        if fresh is None:
            logger.info("replica %s: mailbox report %s was a false alarm", key, evidence.get("error_class"))
            return
        self._on_confirmed(replica, key, fresh)

    def report_external_fault(self, replica_key: str, evidence: dict) -> None:
        """Synchronous driver-side entry: the sync tolerance path's passive death report."""
        replica = self._replica_by_key(replica_key)
        if replica is None or self._loop is None or self._stopped:
            return
        evidence_obj = FaultEvidence(
            source=str(evidence.get("source", "sync_commit")),
            error_class=str(evidence.get("error_class", "unknown")),
        )
        asyncio.run_coroutine_threadsafe(self._verify_report(replica, replica_key, vars(evidence_obj)), self._loop)

    def deferred_replica_ready(self, replica_key: str) -> None:
        """Route B fulfillment: the next sync installed the weights — verify + register."""
        replica = self._replica_by_key(replica_key)
        if replica is None or self._loop is None or self._stopped:
            return
        asyncio.run_coroutine_threadsafe(self._verify_and_register(replica, replica_key), self._loop)

    # ── orchestration (design §6) ───────────────────────────────────────

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
        if self._loop is None or self._stopped:
            return
        with self._schedule_lock:  # sweep (trainer loop) vs awaiter/tolerance (supervisor loop)
            if key in self._recovering:
                logger.info("recovery already in flight for replica %s", key)
                return
            self._recovering.add(key)
        future = asyncio.run_coroutine_threadsafe(self._handle_confirmed_fault(replica, key, evidence), self._loop)
        future.add_done_callback(lambda _fut, _key=key: self._recovery_finished(_key))

    def _recovery_finished(self, key: str) -> None:
        self._recovering.discard(key)

    async def _handle_confirmed_fault(self, replica, key: str, evidence: FaultEvidence) -> None:
        """ISOLATED -> (budget) -> rebuild -> route -> VERIFYING -> SERVING/DEAD (§4/§6.1)."""
        budget = self._budget(key)
        if not budget.try_acquire():
            logger.error("restart budget exhausted for replica %s — going DEAD", key)
            self._states[key] = ReplicaRuntimeState.DEAD
            self._check_all_dead()
            return

        await self._isolate(replica, key)
        await self._bookkeeping_change(replica, remove=True)

        self._states[key] = ReplicaRuntimeState.RESTARTING
        override = await self._plan_weight_route()
        try:
            await replica.rebuild(
                cleanup_timeout_s=self._cleanup_timeout_s(),
                weight_source_override=override,
            )
        except Exception:
            logger.exception("rebuild failed for replica %s; staying ISOLATED (retry on next evidence)", key)
            self._states[key] = ReplicaRuntimeState.ISOLATED
            return

        await self._bookkeeping_change(replica, remove=False)
        self._states[key] = ReplicaRuntimeState.COLD_START

        if override is None:
            # Route B (explicit or A-fallback, §6.2): staged empty start, wait for the
            # next weight sync; the manager's deferred-resume notifies this supervisor.
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.defer_weight_sync_resume(replica)
            logger.warning("replica %s relaunched via route B (no checkpoint weight source)", key)
            return

        new_server = replica.servers[0]
        await new_server.set_global_steps.remote(override.global_steps)
        self._states[key] = ReplicaRuntimeState.VERIFYING
        await self._verify(replica, expected_version=override.global_steps)
        await self._register(replica, key)
        logger.info("replica %s recovered via route A at version %s", key, override.global_steps)

    async def _verify_and_register(self, replica, key: str) -> None:
        """Route B completion: VERIFYING (health-only; the sync stamped the version)."""
        if self._states.get(key) != ReplicaRuntimeState.COLD_START:
            return
        self._states[key] = ReplicaRuntimeState.VERIFYING
        try:
            await self._verify(replica, expected_version=None)
        except Exception:
            logger.exception("route B verification failed for replica %s", key)
            self._states[key] = ReplicaRuntimeState.COLD_START
            return
        await self._register(replica, key)
        logger.info("replica %s recovered via route B after weight sync", key)

    async def _isolate(self, replica, key: str) -> None:
        """LB CAS fence + best-effort abort of the dead replica (§4.3 ISOLATED)."""
        lb = self._manager.global_load_balancer
        address = getattr(replica, "_server_address", None)
        handle = getattr(replica, "_server_handle", None)
        if address is not None:
            try:
                await lb.remove_server_if_current.remote(address, handle)
            except Exception:
                logger.warning("CAS removal unavailable; falling back to remove_servers", exc_info=True)
                try:
                    await lb.remove_servers.remote([address])
                except Exception:
                    logger.warning("remove_servers fallback failed (non-fatal)", exc_info=True)
        try:
            await asyncio.wait_for(replica.abort_all_requests(reject_request=True), timeout=self._probe_timeout_s())
        except Exception:
            # Expected when the actor is already gone — best-effort by design.
            logger.debug("best-effort abort on dead replica %s failed (expected)", key, exc_info=True)

    async def _bookkeeping_change(self, replica, remove: bool) -> None:
        """Membership bookkeeping under the manager's lock (remove-early / add-late, §6.5).

        Yields to an in-flight sync first (wait_for_weight_update), then takes
        the lock (millisecond window: list change + one generation bump).
        """
        cm = self._checkpoint_manager
        if cm is None:
            return
        deadline = time.monotonic() + self._sync_lock_wait_s()
        while not cm._sync_exclusivity.acquire(blocking=False):
            if time.monotonic() > deadline:
                raise ReplicaRestartError(
                    f"could not acquire the bookkeeping lock for replica {self._replica_key(replica)} in time"
                )
            try:
                await asyncio.wait_for(cm.wait_for_weight_update(), timeout=max(0.1, deadline - time.monotonic()))
            except Exception:
                pass
            await asyncio.sleep(0.05)
        try:
            if remove:
                cm.remove_replicas([replica])
            else:
                cm.add_replicas([replica])
            cm.invalidate_group()
        finally:
            cm._sync_exclusivity.release()

    async def _plan_weight_route(self) -> Optional[WeightSourceOverride]:
        """Route selection (§6.2): locate a valid checkpoint, else degrade to B."""
        if str(getattr(self._config, "weight_source", "checkpoint")) != "checkpoint":
            return None
        checkpoint_dir = self._resolve_checkpoint_dir()
        if checkpoint_dir is None:
            logger.warning("route A has no checkpoint_dir — degrading to route B")
            return None
        from verl.utils.checkpoint.locate import locate_latest_valid_checkpoint

        try:
            model_path, global_steps = await asyncio.wait_for(
                asyncio.to_thread(locate_latest_valid_checkpoint, checkpoint_dir),
                timeout=self._locate_timeout_s(),
            )
        except asyncio.TimeoutError:
            logger.warning("checkpoint locate timed out after %ss — degrading to route B", self._locate_timeout_s())
            return None
        except Exception as exc:
            logger.warning("route A unavailable (%s: %s) — degrading to route B", type(exc).__name__, exc)
            return None
        logger.info("route A: cold start from %s (version %s)", model_path, global_steps)
        return WeightSourceOverride(model_path=model_path, global_steps=global_steps)

    async def _verify(self, replica, expected_version: Optional[int]) -> None:
        """VERIFYING: health probe + version stamp check (design §4.3)."""
        await asyncio.wait_for(replica.check_health(), timeout=self._probe_timeout_s())
        if expected_version is not None:
            server = replica.servers[0]
            actual = await server.get_global_steps.remote()
            if actual != expected_version:
                raise ReplicaRestartError(
                    f"version stamp mismatch after rebuild: expected {expected_version}, got {actual}"
                )

    async def _register(self, replica, key: str) -> None:
        """Re-register with the LB — strictly the last step of recovery (§4.2)."""
        lb = self._manager.global_load_balancer
        await lb.add_servers.remote({replica.server_address: replica.server_handle})
        await lb.clear_sticky_cache.remote()
        self._states[key] = ReplicaRuntimeState.SERVING

    def _check_all_dead(self) -> None:
        replica_keys = {self._replica_key(r) for r in getattr(self._manager, "rollout_replicas", []) or []}
        dead_keys = {k for k, s in self._states.items() if s == ReplicaRuntimeState.DEAD}
        if replica_keys and dead_keys >= replica_keys:
            message = "all rollout replicas are DEAD"
            logger.critical(message)
            if self._checkpoint_manager is not None:
                self._checkpoint_manager._fatal_error = message

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _replica_key(replica) -> str:
        return f"{replica.replica_rank}{getattr(replica, 'name_suffix', '') or ''}"

    def _replica_by_key(self, key: str):
        for replica in getattr(self._manager, "rollout_replicas", []) or []:
            if self._replica_key(replica) == key:
                return replica
        return None

    def _resolve_checkpoint_dir(self) -> Optional[str]:
        explicit = getattr(self._config, "checkpoint_dir", None)
        if explicit:
            return str(explicit)
        trainer_cfg = getattr(getattr(self._manager, "config", None), "trainer", None)
        default_dir = getattr(trainer_cfg, "default_local_dir", None)
        return str(default_dir) if default_dir else None

    def _probe_timeout_s(self) -> float:
        return float(getattr(self._config, "probe_timeout_s", 10.0))

    def _locate_timeout_s(self) -> float:
        return float(getattr(self._config, "locate_timeout_s", 30.0))

    def _cleanup_timeout_s(self) -> float:
        return float(getattr(self._config, "cleanup_timeout_s", 60.0))

    def _sync_lock_wait_s(self) -> float:
        return float(getattr(self._config, "sync_lock_wait_s", 120.0))

    def _budget(self, key: str) -> RestartBudget:
        budget_cfg = getattr(self._config, "retry_budget", None)
        budget = self._budgets.get(key)
        if budget is None:
            budget = RestartBudget(
                max_restarts=int(getattr(budget_cfg, "max_restarts", 3)),
                window_s=float(getattr(budget_cfg, "window_s", 3600.0)),
                cooldown_s=float(getattr(budget_cfg, "cooldown_s", 30.0)),
            )
            self._budgets[key] = budget
        return budget

    def _validate_applicability(self) -> None:
        """Fail-loud applicability gate (design §3.4, D7)."""
        rollout_cfg = self._manager.rollout_config
        disaggregation = getattr(rollout_cfg, "disaggregation", None)
        if disaggregation is not None and getattr(disaggregation, "enabled", False):
            raise NotImplementedError("rollout.restart is not supported with PD disaggregation")
        rollout_name = str(getattr(rollout_cfg, "name", ""))
        if rollout_name not in ("vllm", "sglang"):
            raise NotImplementedError(f"rollout.restart does not support rollout.name={rollout_name!r}")
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

    def snapshot_states(self) -> dict[str, str]:
        return {key: state.value for key, state in self._states.items()}


def _get_actor_state_dict(handle) -> Optional[dict]:
    """Actor state via the Ray GCS state API (the ``_is_worker_alive`` pattern)."""
    try:
        from ray.experimental.state.api import get_actor

        return get_actor(handle._actor_id.hex())
    except Exception:
        return None
