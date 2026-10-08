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

import pytest

from verl.workers.config.rollout import RolloutConfig


class TestRolloutPipelineParallelGate:
    def test_vllm_accepts_pp_greater_than_one(self):
        config = RolloutConfig(name="vllm", pipeline_model_parallel_size=2)
        assert config.pipeline_model_parallel_size == 2

    @pytest.mark.parametrize("name", ["sglang", "trtllm"])
    def test_other_rollouts_still_reject_pp(self, name):
        with pytest.raises(NotImplementedError, match="pipeline_model_parallel_size > 1"):
            RolloutConfig(name=name, pipeline_model_parallel_size=2)

    @pytest.mark.parametrize("name", ["vllm", "sglang", "trtllm"])
    def test_pp_equals_one_unchanged_for_all(self, name):
        config = RolloutConfig(name=name, pipeline_model_parallel_size=1)
        assert config.pipeline_model_parallel_size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
