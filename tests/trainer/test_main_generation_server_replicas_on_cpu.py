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

pytest.importorskip("aiohttp")
pytest.importorskip("pandas")
pytest.importorskip("openai")

from verl.trainer.main_generation_server import _num_rollout_replicas


class TestNumRolloutReplicas:
    def test_tp2_pp2_eight_gpus_yields_two_replicas(self):
        assert _num_rollout_replicas(8, 1, tp_size=2, pp_size=2) == 2

    def test_pp1_degenerates_to_tp_divisor(self):
        assert _num_rollout_replicas(8, 2, tp_size=4, pp_size=1) == 4

    def test_multi_node_counts_all_gpus(self):
        assert _num_rollout_replicas(8, 2, tp_size=2, pp_size=2) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
