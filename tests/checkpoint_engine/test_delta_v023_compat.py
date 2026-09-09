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
"""Unit tests for the vllm 0.23.x delta weight-transfer compat shim.

NOT FOR UPSTREAM (deployment/test-only layer; see
delta_weight_transfer_v023_compat). The pure-logic cases (host
classification, per-worker update-info indexing) run on any host with vLLM
importable; engine-construction cases only run on vLLM 0.23.x hosts; the
vendored checkpoint-patch cases need only torch (CPU).
"""

import pytest
import torch

pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout import delta_weight_transfer_v023_compat as v023_compat
from verl.workers.rollout.vllm_rollout.delta_weight_transfer import _checkpoint_patch_api

_HOST = v023_compat.classify_installed_host()


@pytest.mark.parametrize(
    ("init_params", "vllm_version", "expected"),
    [
        ({"self", "config", "vllm_config", "device", "model"}, "0.28.1rc0", v023_compat.HOST_ANCHORED),
        ({"self", "config", "vllm_config", "device", "model"}, "0.23.1", v023_compat.HOST_ANCHORED),
        ({"self", "config", "parallel_config", "model"}, "0.23.0", v023_compat.HOST_V023),
        ({"self", "config", "parallel_config", "model"}, "0.23.0rc1", v023_compat.HOST_V023),
        ({"self", "config", "parallel_config", "model"}, "0.23.2.dev4", v023_compat.HOST_V023),
        ({"self", "config", "parallel_config", "model"}, "0.24.0", v023_compat.HOST_UNSUPPORTED),
        ({"self", "config", "parallel_config", "model"}, "0.25.0", v023_compat.HOST_UNSUPPORTED),
        ({"self", "config", "parallel_config", "model"}, "not-a-version", v023_compat.HOST_UNSUPPORTED),
        ({"self", "config", "model"}, "0.23.0", v023_compat.HOST_UNSUPPORTED),
        ({"self", "config", "vllm_config", "device"}, "0.23.0", v023_compat.HOST_UNSUPPORTED),
    ],
)
def test_classify_weight_transfer_host(init_params, vllm_version, expected):
    assert v023_compat.classify_weight_transfer_host(init_params, vllm_version) == expected


def test_require_vllm_delta_support_gates_on_host(monkeypatch):
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import require_vllm_delta_support

    monkeypatch.setattr(v023_compat, "classify_installed_host", lambda: v023_compat.HOST_UNSUPPORTED)
    with pytest.raises(RuntimeError, match="delta_sharded"):
        require_vllm_delta_support()

    monkeypatch.setattr(v023_compat, "classify_installed_host", lambda: v023_compat.HOST_V023)
    require_vllm_delta_support()

    monkeypatch.setattr(v023_compat, "classify_installed_host", lambda: v023_compat.HOST_ANCHORED)
    require_vllm_delta_support()


def test_checkpoint_patch_api_prefers_native_module():
    checkpoint_weight_patch, load_checkpoint_weight_patches = _checkpoint_patch_api()
    try:
        from vllm.model_executor.model_loader import checkpoint_weight_patch as native
    except ImportError:
        assert checkpoint_weight_patch.__module__ == (
            "verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch"
        )
    else:
        assert checkpoint_weight_patch is native.CheckpointWeightPatch
        assert load_checkpoint_weight_patches is native.load_checkpoint_weight_patches


@pytest.mark.parametrize(
    ("update_info", "data_parallel_rank", "world_size", "rank", "expected"),
    [
        ([{"group_rank": i} for i in range(6)], 0, 4, 2, {"group_rank": 2}),
        ([{"group_rank": i} for i in range(6)], 1, 2, 0, {"group_rank": 2}),
        ({"group_rank": 9}, 1, 2, 0, {"group_rank": 9}),
    ],
)
def test_local_update_info_indexing(update_info, data_parallel_rank, world_size, rank, expected):
    assert (
        v023_compat._local_update_info(
            update_info,
            data_parallel_rank=data_parallel_rank,
            world_size=world_size,
            rank=rank,
        )
        == expected
    )


class _CheckpointFormatModel(torch.nn.Module):
    """Minimal checkpoint-format model: load_weights copies named tensors."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2))

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(tensor)
        return [name for name, _ in weights]


def _sparse_patch():
    from verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch import CheckpointWeightPatch

    return CheckpointWeightPatch(
        name="weight",
        shape=(4, 2),
        dtype=torch.float32,
        values=torch.tensor([42.0]),
        indices=torch.tensor([3]),
    )


def _dense_patch():
    from verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch import CheckpointWeightPatch

    return CheckpointWeightPatch(
        name="weight",
        shape=(4, 2),
        dtype=torch.float32,
        values=torch.arange(8, dtype=torch.float32) + 100.0,
        indices=None,
    )


def test_vendored_patch_applies_sparse_and_dense_updates():
    from verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch import (
        load_checkpoint_weight_patches,
    )

    model = _CheckpointFormatModel()
    baseline = model.weight.detach().flatten().clone()

    loaded = load_checkpoint_weight_patches(model, [_sparse_patch()], validate_unique_indices=False)
    assert loaded == {"weight"}
    flat = model.weight.detach().flatten()
    assert flat[3] == 42.0
    unchanged = [value for i, value in enumerate(flat.tolist()) if i != 3]
    assert unchanged == [value for i, value in enumerate(baseline.tolist()) if i != 3]

    load_checkpoint_weight_patches(model, [_dense_patch()])
    assert torch.equal(model.weight.detach().flatten(), torch.arange(8, dtype=torch.float32) + 100.0)


def test_vendored_patch_rejects_mixed_dense_and_sparse():
    from verl.workers.rollout.vllm_rollout._vendored.checkpoint_weight_patch import (
        load_checkpoint_weight_patches,
    )

    model = _CheckpointFormatModel()
    with pytest.raises(ValueError, match="cannot be mixed"):
        load_checkpoint_weight_patches(model, [_dense_patch(), _sparse_patch()])


@pytest.mark.skipif(_HOST != v023_compat.HOST_V023, reason="requires a vLLM 0.23.x host")
def test_compat_engine_constructs_parses_and_updates():
    from types import SimpleNamespace

    model_config = SimpleNamespace(dtype=torch.bfloat16)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
        quant_config=None,
        speculative_config=None,
    )
    engine = v023_compat.VerlDeltaHCCLWeightTransferEngineV023(
        SimpleNamespace(), vllm_config.parallel_config, SimpleNamespace()
    )
    engine.hydrate_v023(vllm_config=vllm_config, device=torch.device("cpu"))
    assert engine.vllm_config is vllm_config
    assert engine.model_config is model_config
    assert engine.device.type == "cpu"

    received = []
    engine.receive_weights = lambda update_info: received.append(update_info)
    engine.update_weights({"group_rank": 3})
    assert received and received[0].group_rank == 3

    with pytest.raises(ValueError, match="Invalid update_info"):
        engine.update_weights({})


@pytest.mark.skipif(_HOST != v023_compat.HOST_V023, reason="requires a vLLM 0.23.x host")
def test_factory_creates_compat_engine_on_v023_host():
    from types import SimpleNamespace

    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import (
        register_verl_delta_hccl_weight_transfer_engine,
    )

    register_verl_delta_hccl_weight_transfer_engine()
    engine = WeightTransferEngineFactory.create_engine(
        SimpleNamespace(backend="verl_delta_hccl"),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert isinstance(engine, v023_compat.VerlDeltaHCCLWeightTransferEngineV023)


@pytest.mark.skipif(_HOST != v023_compat.HOST_ANCHORED, reason="requires a #44353 host")
def test_compat_engine_cls_identity_on_anchored_host():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import (
        VerlDeltaHCCLWeightTransferEngine,
        VerlDeltaIPCWeightTransferEngine,
    )

    assert v023_compat.compat_engine_cls(VerlDeltaIPCWeightTransferEngine) is VerlDeltaIPCWeightTransferEngine
    assert v023_compat.compat_engine_cls(VerlDeltaHCCLWeightTransferEngine) is VerlDeltaHCCLWeightTransferEngine


if __name__ == "__main__":
    pytest.main([__file__])
