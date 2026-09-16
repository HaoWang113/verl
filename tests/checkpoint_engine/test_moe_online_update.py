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
"""Minimal unit tests for the MoE sparse layout window (delta_sharded, NPU).

Pure-logic tier of the L7 ladder: torch on CPU, no torch_npu/vllm-ascend
required except where explicitly branched. Execution happens on the NPU test
environment per tests/special_npu/DELTA_HCCL_TEST_GUIDE.md (L7.0) -- never on
the dev machine.
"""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.rollout.vllm_rollout import moe_online_update as mo

E, K, N = 2, 6, 4  # w13 runtime [E, K, 2N] / w2 runtime [E, N, K]


class FakeExperts(torch.nn.Module):
    """Expert weights at *runtime* layout, i.e. what vllm-ascend leaves after load."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.randn(E, K, 2 * N, dtype=dtype), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.randn(E, N, K, dtype=dtype), requires_grad=False)


class FakeModel(torch.nn.Module):
    def __init__(self, *children):
        super().__init__()
        for i, child in enumerate(children):
            self.add_module(f"experts_{i}", child)


def _make_engine():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import VerlDeltaIPCWeightTransferEngine

    engine = object.__new__(VerlDeltaIPCWeightTransferEngine)
    engine.model = torch.nn.Module()
    engine._session_encoding = None
    engine._update_failed = False
    engine._sparse_session_begun = False
    return engine


@pytest.fixture()
def patched_patch_api(monkeypatch):
    applied = []

    def fake_load(model, patches, validate_unique_indices=False):
        applied.extend(patches)

    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.delta_weight_transfer._checkpoint_patch_api",
        lambda: (SimpleNamespace, fake_load),
    )
    return applied


def _flush(engine, encoding, monkeypatch):
    patch = {"name": "w13_weight", "shape": None, "dtype": None, "values": None, "indices": None}

    def fake_decode(payload):
        return encoding, [patch]

    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.delta_weight_transfer.decode_delta_payload", fake_decode
    )
    engine._decode_and_apply(payload=[])


def test_window_roundtrip_bitexact():
    layer = FakeExperts()
    w13_runtime = layer.w13_weight.data.clone()
    w2_runtime = layer.w2_weight.data.clone()

    window = mo.open_expert_layout_window(FakeModel(layer))
    assert window is not None and len(window.flipped_layers) == 1
    # runtime -> checkpoint coordinate layout
    assert layer.w13_weight.shape == (E, 2 * N, K)
    assert layer.w2_weight.shape == (E, K, N)
    assert torch.equal(layer.w13_weight.data, w13_runtime.transpose(1, 2))

    mo.close_expert_layout_window(window)
    assert layer.w13_weight.shape == (E, K, 2 * N)
    assert layer.w2_weight.shape == (E, N, K)
    assert torch.equal(layer.w13_weight.data, w13_runtime)
    assert torch.equal(layer.w2_weight.data, w2_runtime)


def test_window_is_noop_without_expert_modules():
    assert mo.open_expert_layout_window(FakeModel()) is None


@pytest.mark.parametrize("attr", ["w13_weight_list", "w2_weight_list"])
def test_list_split_variant_fails_loud(attr):
    layer = FakeExperts()
    setattr(layer, attr, [torch.randn(1)])
    with pytest.raises(NotImplementedError, match="list"):
        mo.open_expert_layout_window(FakeModel(layer))


def test_non_bf16_or_wrong_rank_fails_loud():
    fp32_layer = FakeExperts(dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="BF16"):
        mo.open_expert_layout_window(FakeModel(fp32_layer))

    bad_rank = FakeExperts()
    bad_rank.w2_weight = torch.nn.Parameter(torch.randn(E, K), requires_grad=False)
    with pytest.raises(NotImplementedError, match="3D"):
        mo.open_expert_layout_window(FakeModel(bad_rank))


def test_unknown_vllm_ascend_version_fails_loud(monkeypatch):
    monkeypatch.setattr(mo, "_package_version", lambda name: "9.9.9")
    with pytest.raises(NotImplementedError, match="9.9.9"):
        mo.open_expert_layout_window(FakeModel(FakeExperts()))


def test_fingerprint_host_shape_probe(monkeypatch):
    monkeypatch.setattr(mo, "_package_version", lambda name: "0.23.0rc1")
    try:
        import vllm_ascend  # noqa: F401

        has_vllm_ascend = True
    except ImportError:
        has_vllm_ascend = False

    if has_vllm_ascend:
        assert mo.assert_online_update_fingerprint() in (mo._ANCHORED_HOST_MODULE, mo._V023_HOST_MODULE)
    else:
        with pytest.raises(NotImplementedError, match="neither"):
            mo.assert_online_update_fingerprint()


@pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("vllm_ascend"),
    reason="vllm-ascend not installed",
)
def test_nz_mode_gate_fails_loud(monkeypatch):
    from vllm_ascend import ascend_config as ascend_config_mod

    monkeypatch.setattr(mo, "_package_version", lambda name: "0.23.0rc1")
    monkeypatch.setattr(
        ascend_config_mod, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=2), raising=True
    )
    with pytest.raises(NotImplementedError, match="weight_nz_mode=2"):
        mo.open_expert_layout_window(FakeModel(FakeExperts()))


def test_resolve_moe_delta_settings_platform_contracts():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import resolve_moe_delta_settings as resolve

    assert resolve(is_npu=True, moe_backend=None, enable_eplb=False) is None
    assert resolve(is_npu=True, moe_backend="auto", enable_eplb=False) is None
    with pytest.raises(NotImplementedError, match="checkpoint-layout window"):
        resolve(is_npu=True, moe_backend="triton", enable_eplb=False)
    with pytest.raises(NotImplementedError, match="EPLB"):
        resolve(is_npu=True, moe_backend=None, enable_eplb=True)

    assert resolve(is_npu=False, moe_backend=None, enable_eplb=False) == "triton"
    assert resolve(is_npu=False, moe_backend="auto", enable_eplb=False) == "triton"
    with pytest.raises(NotImplementedError, match="triton"):
        resolve(is_npu=False, moe_backend="flashinfer_cutlass", enable_eplb=False)
    with pytest.raises(NotImplementedError, match="EPLB"):
        resolve(is_npu=False, moe_backend=None, enable_eplb=True)


def test_sparse_session_hook_sequence(patched_patch_api, monkeypatch):
    engine = _make_engine()
    calls = []
    engine._on_sparse_session_begin = lambda: calls.append("begin")
    engine._on_sparse_session_end = lambda: calls.append("end")

    _flush(engine, "indices", monkeypatch)
    assert engine._session_encoding == "indices"
    assert engine._sparse_session_begun is True
    _flush(engine, "indices", monkeypatch)
    assert calls == ["begin"], "begin must fire once per session, before the first sparse flush"

    engine.finish_weight_update()
    assert calls == ["begin", "end"]
    assert engine._session_encoding is None and engine._sparse_session_begun is False


def test_dense_session_skips_sparse_hooks(patched_patch_api, monkeypatch):
    import vllm.model_executor.model_loader.reload as reload_mod

    reload_calls = []
    monkeypatch.setattr(reload_mod, "initialize_layerwise_reload", lambda model: reload_calls.append("init"))
    monkeypatch.setattr(reload_mod, "finalize_layerwise_reload", lambda model, cfg: reload_calls.append("fin"))

    engine = _make_engine()
    sparse_calls = []
    engine._on_sparse_session_begin = lambda: sparse_calls.append("begin")
    engine._on_sparse_session_end = lambda: sparse_calls.append("end")

    _flush(engine, "dense", monkeypatch)
    assert engine._session_encoding == "dense" and reload_calls == ["init"]
    engine.finish_weight_update()
    assert reload_calls == ["init", "fin"]
    assert sparse_calls == [] and engine._sparse_session_begun is False


def test_mixed_encoding_rejected(patched_patch_api, monkeypatch):
    engine = _make_engine()
    _flush(engine, "indices", monkeypatch)
    with pytest.raises(ValueError, match="mix"):
        _flush(engine, "dense", monkeypatch)


def test_error_path_best_effort_restore(patched_patch_api, monkeypatch):
    engine = _make_engine()
    calls = []

    def broken_end():
        calls.append("end-attempt")
        raise RuntimeError("restore failed")

    engine._on_sparse_session_begin = lambda: calls.append("begin")
    engine._on_sparse_session_end = broken_end

    def broken_payload(*, zmq_handle):
        raise RuntimeError("transport boom")

    engine._receive_payload = broken_payload
    engine._sparse_session_begun = True  # simulate a window already opened by the first flush
    update_info = SimpleNamespace(zmq_handle="h")

    with pytest.raises(RuntimeError, match="transport boom"):
        engine.receive_weights(update_info)

    assert calls == ["end-attempt"]
    assert engine._update_failed is True
    assert engine._sparse_session_begun is False, "exactly one end attempt: the flag is cleared before the try"


def test_start_rejects_unclean_session(patched_patch_api, monkeypatch):
    engine = _make_engine()
    engine._sparse_session_begun = True
    with pytest.raises(RuntimeError, match="never finished cleanly"):
        engine.start_weight_update()
