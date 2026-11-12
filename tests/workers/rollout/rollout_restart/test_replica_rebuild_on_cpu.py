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
"""CPU tests for the unified replica rebuild (rollout-restart design §6.1, PR-A2).

Uses a minimal RolloutReplica subclass with real Ray actors for the server and
CE-worker roles, and a stub RayWorkerGroup for the respawn step (the real one
constructs CheckpointEngineWorker actors whose engines need accelerators). The
five-segment teardown — verified shutdown, actor kill + bounded terminal wait,
witness probe, CE kill, respawn+relaunch — runs for real against Ray's actor
lifecycle; only the "engine" behind each actor is fake.
"""

import asyncio
from types import SimpleNamespace

import psutil
import pytest
import ray

import verl.workers.rollout.replica as replica_module
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.restart import ReplicaRestartError, WeightSourceOverride


def _fake_rollout_config(tp=1):
    return SimpleNamespace(
        tensor_model_parallel_size=tp,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


@ray.remote
class _FakeServerActor:
    async def shutdown(self):
        proc = psutil.Process()
        return proc.pid, proc.create_time()

    async def check_health(self):
        return None


@ray.remote
class _FakeWorkerActor:
    pass


class _RespawningWorkerGroup:
    """Stands in for RayWorkerGroup: hands out the next batch of fake CE workers."""

    spawn_calls: list[dict] = []

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        type(self).spawn_calls.append(dict(kwargs))
        self.workers = [_FakeWorkerActor.remote(), _FakeWorkerActor.remote()]


class _FakeReplica(RolloutReplica):
    """Minimal concrete replica: records the relaunch instead of starting engines."""

    def __init__(self, **kwargs):
        super().__init__(config=_fake_rollout_config(), model_config=SimpleNamespace(), **kwargs)
        self.relaunch_calls = []

    async def launch_servers(self, weight_source_override=None):
        self.relaunch_calls.append(weight_source_override)
        self.servers = [_FakeServerActor.remote()]
        self._server_handle = self.servers[0]
        self._server_address = f"fake-{self.replica_rank}"


@pytest.fixture()
def replica():
    ray.init(ignore_reinit_error=True)
    instance = _FakeReplica(replica_rank=0, gpus_per_node=2)
    instance.rollout_mode = RolloutMode.STANDALONE
    instance.resource_pool = object()
    instance.workers = [_FakeWorkerActor.remote() for _ in range(2)]
    instance.servers = [_FakeServerActor.remote()]
    return instance


def test_rebuild_replaces_workers_and_servers(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    old_workers = list(replica.workers)
    old_server = replica.servers[0]

    asyncio.run(replica.rebuild())

    assert replica.relaunch_calls == [None]
    assert replica.workers != old_workers
    assert replica.servers != [old_server]
    assert replica._restart_in_progress is False
    assert _RespawningWorkerGroup.spawn_calls[-1]["name_prefix"] == "rollout_standalone_0"


def test_rebuild_passes_weight_source_override_to_relaunch(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    override = WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
    asyncio.run(replica.rebuild(weight_source_override=override))
    assert replica.relaunch_calls == [override]


def test_rebuild_rejects_non_standalone_mode(replica):
    replica.rollout_mode = RolloutMode.HYBRID
    with pytest.raises(NotImplementedError):
        asyncio.run(replica.rebuild())
    assert replica._restart_in_progress is False


def test_rebuild_rejects_missing_resource_pool(replica):
    replica.resource_pool = None
    with pytest.raises(ReplicaRestartError):
        asyncio.run(replica.rebuild())


def test_rebuild_reentry_guard(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)
    replica._restart_in_progress = True
    with pytest.raises(ReplicaRestartError, match="already in progress"):
        asyncio.run(replica.rebuild())


def test_rebuild_failure_resets_server_state(replica, monkeypatch):
    monkeypatch.setattr(replica_module, "RayWorkerGroup", _RespawningWorkerGroup)

    async def broken_launch(weight_source_override=None):
        raise RuntimeError("engine failed to start")

    replica.launch_servers = broken_launch
    with pytest.raises(RuntimeError, match="engine failed to start"):
        asyncio.run(replica.rebuild())
    assert replica.servers == []
    assert replica._server_handle is None
    assert replica._server_address is None
    assert replica._restart_in_progress is False


def test_rebuild_witness_refuses_when_server_process_survives(replica, monkeypatch):
    """Fail-closed: the shutdown() identity never dies -> no relaunch."""

    @ray.remote
    class _IdentityActor:
        def identity(self):
            proc = psutil.Process()
            return proc.pid, proc.create_time()

    @ray.remote
    class _UndyingServerActor:
        def __init__(self, foreign_identity):
            # Reports a foreign, healthy process (a live sibling actor): the
            # witness probe must observe it alive and refuse to relaunch.
            self._foreign_identity = foreign_identity

        async def shutdown(self):
            return self._foreign_identity

    buddy = _IdentityActor.remote()
    foreign = ray.get(buddy.identity.remote())
    replica.servers = [_UndyingServerActor.remote(foreign)]

    async def must_not_relaunch(weight_source_override=None):
        raise AssertionError("relaunch must not happen after a failed witness")

    replica.launch_servers = must_not_relaunch
    with pytest.raises(ReplicaRestartError, match="witness probe"):
        asyncio.run(replica.rebuild())


def test_check_health_without_servers_raises(replica):
    replica.servers = []
    with pytest.raises(ReplicaRestartError, match="no live server actor"):
        asyncio.run(replica.check_health())


def test_check_health_probes_head_server(replica):
    asyncio.run(replica.check_health())
