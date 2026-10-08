# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
from types import SimpleNamespace

import pytest

# vllm is not part of the `cpu` extra (it conflicts with the cpu torch world), so
# cpu_unit_tests skips this module; vllm.yml runs it in the vllm venv.
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.utils import (
    _resolve_vllm_weight_sync_local_rank,
    build_cli_args_from_config,
    vLLMColocateWorkerExtension,
)


class TestBuildCliArgsFromConfig:
    """Tests for CLI argument serialization from config dictionaries."""

    def test_string_value(self):
        """String values become '--key value'."""
        config = {"model": "gpt2"}
        result = build_cli_args_from_config(config)
        assert result == ["--model", "gpt2"]

    def test_integer_value(self):
        """Integer values are converted to strings."""
        config = {"tensor-parallel-size": 4}
        result = build_cli_args_from_config(config)
        assert result == ["--tensor-parallel-size", "4"]

    def test_float_value(self):
        """Float values are converted to strings."""
        config = {"temperature": 0.7}
        result = build_cli_args_from_config(config)
        assert result == ["--temperature", "0.7"]

    def test_bool_true(self):
        """Bool True adds flag without value."""
        config = {"enable-prefix-caching": True}
        result = build_cli_args_from_config(config)
        assert result == ["--enable-prefix-caching"]

    def test_bool_false(self):
        """Optional[bool] args emit '--no-key' for an explicit False."""
        config = {"enable-prefix-caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable-prefix-caching"]

    def test_bool_false_plain_bool_omitted(self):
        """Bool False on a plain-bool arg is skipped (parser default is False)."""
        config = {"enforce_eager": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_union_str_arg_omitted(self):
        """Bool False on a `bool | str | None` arg is skipped (string flag, no --no- form)."""
        config = {"hf_token": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_underscore_key(self):
        """Underscore keys emit the negative flag in the key's own spelling."""
        config = {"enable_prefix_caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable_prefix_caching"]

    def test_bool_false_non_engine_arg_omitted(self):
        """Bool False on args unknown to AsyncEngineArgs is omitted."""
        config = {"disable-log-requests": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_none_value(self):
        """None values are skipped."""
        config = {"lora-path": None}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_values(self):
        """List values are expanded into multiple arguments."""
        config = {"cudagraph-capture-sizes": [1, 2, 4, 8]}
        result = build_cli_args_from_config(config)
        assert result == ["--cudagraph-capture-sizes", "1", "2", "4", "8"]

    def test_empty_list(self):
        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
        config = {"cudagraph-capture-sizes": []}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_with_strings(self):
        """List of strings is properly expanded."""
        config = {"allowed-origins": ["http://localhost", "http://example.com"]}
        result = build_cli_args_from_config(config)
        assert result == ["--allowed-origins", "http://localhost", "http://example.com"]

    def test_dict_value(self):
        """Dict values are JSON serialized."""
        config = {"extra-config": {"key": "value", "nested": True}}
        result = build_cli_args_from_config(config)
        assert result[0] == "--extra-config"
        # JSON output may have different key ordering, so parse and compare
        assert json.loads(result[1]) == {"key": "value", "nested": True}

    def test_mixed_config(self):
        """Test a realistic mixed configuration."""
        config = {
            "tensor-parallel-size": 4,
            "enable-prefix-caching": True,
            "disable-log-requests": False,
            "lora-path": None,
            "cudagraph-capture-sizes": [1, 2, 4, 8],
            "max-model-len": 2048,
        }
        result = build_cli_args_from_config(config)

        # Check expected args are present
        assert "--tensor-parallel-size" in result
        assert "4" in result
        assert "--enable-prefix-caching" in result
        assert "--cudagraph-capture-sizes" in result
        assert "1" in result
        assert "8" in result
        assert "--max-model-len" in result
        assert "2048" in result

        # Check skipped values are not present
        assert "--disable-log-requests" not in result
        assert "--lora-path" not in result

    def test_preserves_order(self):
        """Arguments should preserve dictionary order (Python 3.7+)."""
        config = {"first": "a", "second": "b", "third": "c"}
        result = build_cli_args_from_config(config)
        assert result == ["--first", "a", "--second", "b", "--third", "c"]

    def test_empty_config(self):
        """Empty config returns empty list."""
        config = {}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_single_element_list(self):
        """Single element list works correctly."""
        config = {"sizes": [42]}
        result = build_cli_args_from_config(config)
        assert result == ["--sizes", "42"]


class TestCliArgsVllmParserRoundTrip:
    """Serialized args must round-trip through vLLM's serve CLI parser."""

    @staticmethod
    def _build_parser():
        import vllm.entrypoints.cli.serve as serve_mod
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="test")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        for cmd in serve_mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        return parser

    def test_explicit_false_survives_parsing(self):
        """Explicit False survives parse_args and AsyncEngineArgs.from_cli_args."""
        from vllm.engine.arg_utils import AsyncEngineArgs

        parser = self._build_parser()
        config = {
            "skip_tokenizer_init": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "enable_sleep_mode": True,
            "enforce_eager": False,
            "disable_log_stats": False,
        }
        argv = ["serve", "dummy-model"] + build_cli_args_from_config(config)
        namespace = parser.parse_args(args=argv)
        engine_args = AsyncEngineArgs.from_cli_args(namespace)
        assert engine_args.enable_prefix_caching is False
        assert engine_args.enable_chunked_prefill is True
        assert engine_args.enable_sleep_mode is True
        assert engine_args.skip_tokenizer_init is False
        assert engine_args.enforce_eager is False
        assert engine_args.disable_log_stats is False


class TestVllmColocateZmqHandle:
    def test_dp_local_rank_offsets_tensor_parallel_rank(self):
        """DP workers on the same node must not reuse the same TP-local socket."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 3
        assert _resolve_vllm_weight_sync_local_rank(3, parallel_config) == 3

    def test_single_dp_keeps_local_rank(self):
        """The old single-DP handle layout remains unchanged."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 1

    def test_pp_only_keeps_in_shard_ranks_distinct(self):
        """PP>1 with DP=1: the four (pp,tp) workers keep distinct socket ranks."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )

        # vLLM in-shard layout is tp-minor: rank = pp_rank * tp + tp_rank.
        assert [_resolve_vllm_weight_sync_local_rank(r, parallel_config) for r in range(4)] == [0, 1, 2, 3]

    def test_dp_and_pp_fold_is_invariant_to_host_dp_offset(self):
        """Whether the host already folded dp into local_rank, the result matches."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        raw = [_resolve_vllm_weight_sync_local_rank(r, parallel_config) for r in range(4)]
        adjusted = [_resolve_vllm_weight_sync_local_rank(4 + r, parallel_config) for r in range(4)]
        assert raw == [4, 5, 6, 7]
        assert adjusted == [4, 5, 6, 7]

    def test_dp_fold_degenerates_with_pp_equals_one(self):
        """pp=1 keeps the historical dp*tp+tp formula byte-for-byte."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 3

    def test_uses_global_dp_rank_when_local_rank_is_unset(self):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=None,
            data_parallel_rank=3,
        )

        assert _resolve_vllm_weight_sync_local_rank(0, parallel_config) == 2

    def test_zmq_handle_uses_resolved_dp_rank(self, monkeypatch):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )
        worker = SimpleNamespace(
            local_rank=1,
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config),
            ),
        )
        monkeypatch.setenv("VERL_REPLICA_RANK", "2")
        monkeypatch.setenv("VERL_RAY_JOB_ID", "job-123")

        handle = vLLMColocateWorkerExtension._get_zmq_handle(worker)

        assert handle == "ipc:///tmp/rl-colocate-zmq-job-123-replica-2-rank-3.sock"

    def test_pp_handle_cross_check_passes_when_groups_agree(self, monkeypatch):
        """Agreeing config- and group-derived ranks must not raise."""
        import vllm.distributed as vllm_dist

        from verl.workers.rollout.vllm_rollout.utils import _assert_weight_sync_rank_matches_groups

        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )
        # worker rank 3 = (pp=1, tp=1) under the tp-minor layout.
        monkeypatch.setattr(vllm_dist, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
        monkeypatch.setattr(vllm_dist, "get_tp_group", lambda: SimpleNamespace(rank_in_group=1))

        _assert_weight_sync_rank_matches_groups(3, parallel_config)

    def test_pp_handle_cross_check_raises_on_mismatch(self, monkeypatch):
        """A config/group disagreement must fail loud, not bind a wrong GPU socket."""
        import vllm.distributed as vllm_dist

        from verl.workers.rollout.vllm_rollout.utils import _assert_weight_sync_rank_matches_groups

        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )
        monkeypatch.setattr(vllm_dist, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
        monkeypatch.setattr(vllm_dist, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))

        with pytest.raises(RuntimeError, match="weight-sync rank mismatch"):
            _assert_weight_sync_rank_matches_groups(3, parallel_config)

    def test_get_zmq_handle_skips_cross_check_when_pp_is_one(self, monkeypatch):
        """tp-only topologies keep the historical handle path without group queries."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )
        worker = SimpleNamespace(
            local_rank=1,
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config),
            ),
        )
        monkeypatch.setenv("VERL_REPLICA_RANK", "2")
        monkeypatch.setenv("VERL_RAY_JOB_ID", "job-123")

        def _explode():
            raise AssertionError("group lookup must not happen when pp=1")

        import vllm.distributed as vllm_dist

        monkeypatch.setattr(vllm_dist, "get_pp_group", _explode)
        handle = vLLMColocateWorkerExtension._get_zmq_handle(worker)
        assert handle == "ipc:///tmp/rl-colocate-zmq-job-123-replica-2-rank-3.sock"


class TestExternalDpGeometry:
    def test_pp2_tp2_eight_gpus_yields_two_local_shards(self):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import _compute_external_dp_args

        config = SimpleNamespace(
            data_parallel_size=2,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
        dp_args = _compute_external_dp_args(
            config=config,
            gpus_per_node=8,
            num_workers=8,
            node_rank=0,
            master_address="10.0.0.1",
            dp_rpc_port=1234,
        )
        assert dp_args["data_parallel_size_local"] == 2
        assert dp_args["data_parallel_start_rank"] == 0

    def test_pp1_degenerates_to_tp_divisor(self):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import _compute_external_dp_args

        config = SimpleNamespace(
            data_parallel_size=2,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
        )
        dp_args = _compute_external_dp_args(
            config=config,
            gpus_per_node=8,
            num_workers=8,
            node_rank=1,
            master_address="10.0.0.1",
            dp_rpc_port=1234,
        )
        assert dp_args["data_parallel_size_local"] == 2
        assert dp_args["data_parallel_start_rank"] == 2

    def test_non_divisible_tp_pp_rejected(self):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import _compute_external_dp_args

        config = SimpleNamespace(
            data_parallel_size=2,
            tensor_model_parallel_size=3,
            pipeline_model_parallel_size=3,
        )
        with pytest.raises(AssertionError, match="divisible"):
            _compute_external_dp_args(
                config=config,
                gpus_per_node=8,
                num_workers=8,
                node_rank=0,
                master_address="10.0.0.1",
                dp_rpc_port=1234,
            )

    def test_worker_count_mismatch_rejected(self):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import _compute_external_dp_args

        config = SimpleNamespace(
            data_parallel_size=2,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
        with pytest.raises(AssertionError, match="num workers"):
            _compute_external_dp_args(
                config=config,
                gpus_per_node=8,
                num_workers=6,
                node_rank=0,
                master_address="10.0.0.1",
                dp_rpc_port=1234,
            )


class TestDeltaShardedTopologyGate:
    @staticmethod
    def _mock_server(tp=2, pp=1, dp=1, gpus_per_node=8, disaggregation=False):
        config = SimpleNamespace(
            data_parallel_size=dp,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            disaggregation=SimpleNamespace(enabled=disaggregation),
            checkpoint_engine=SimpleNamespace(backend="delta_sharded", engine_kwargs={}),
        )
        return SimpleNamespace(
            config=config,
            gpus_per_node=gpus_per_node,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )

    @staticmethod
    def _patch_delta_support(monkeypatch, *, moe=False):
        import verl.workers.rollout.vllm_rollout.delta_weight_transfer as delta_mod
        import verl.workers.rollout.vllm_rollout.vllm_async_server as server_mod

        monkeypatch.setattr(delta_mod, "require_vllm_delta_support", lambda: None)
        monkeypatch.setattr(server_mod, "is_torch_npu_available", lambda check_device=True: False)
        monkeypatch.setattr(delta_mod, "is_moe_model", lambda hf_config: moe)

    def test_accepts_pp_greater_than_one_single_node(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        engine_kwargs = {}
        vLLMHttpServer._preprocess_engine_kwargs(self._mock_server(tp=2, pp=2, gpus_per_node=8), engine_kwargs)
        assert engine_kwargs["weight_transfer_config"]["backend"] == "verl_delta_ipc"

    def test_accepts_dp_and_pp_combined(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        engine_kwargs = {}
        vLLMHttpServer._preprocess_engine_kwargs(self._mock_server(tp=2, pp=2, dp=2, gpus_per_node=8), engine_kwargs)
        assert engine_kwargs["weight_transfer_config"]["backend"] == "verl_delta_ipc"

    def test_rejects_cross_node_pp(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        with pytest.raises(NotImplementedError, match="fit within one node"):
            vLLMHttpServer._preprocess_engine_kwargs(self._mock_server(tp=4, pp=4, gpus_per_node=8), {})

    def test_rejects_moe_with_pp(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch, moe=True)
        with pytest.raises(NotImplementedError, match="MoE models with pipeline_parallel_size > 1"):
            vLLMHttpServer._preprocess_engine_kwargs(self._mock_server(tp=2, pp=2, gpus_per_node=8), {})

    def test_rejects_engine_kwargs_dp_bypass(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        server = self._mock_server(tp=2, pp=1, dp=1, gpus_per_node=8)
        with pytest.raises(NotImplementedError, match="data_parallel_size to match"):
            vLLMHttpServer._preprocess_engine_kwargs(server, {"data_parallel_size": 2})

    def test_rejects_engine_kwargs_pp_bypass(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        server = self._mock_server(tp=2, pp=1, dp=1, gpus_per_node=8)
        with pytest.raises(NotImplementedError, match="pipeline_parallel_size to match"):
            vLLMHttpServer._preprocess_engine_kwargs(server, {"pipeline_parallel_size": 4})

    def test_matching_engine_kwargs_topology_passes(self, monkeypatch):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        self._patch_delta_support(monkeypatch)
        server = self._mock_server(tp=2, pp=2, dp=2, gpus_per_node=8)
        engine_kwargs = {"data_parallel_size": 2, "pipeline_parallel_size": 2}
        vLLMHttpServer._preprocess_engine_kwargs(server, engine_kwargs)
        assert engine_kwargs["weight_transfer_config"]["backend"] == "verl_delta_ipc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
