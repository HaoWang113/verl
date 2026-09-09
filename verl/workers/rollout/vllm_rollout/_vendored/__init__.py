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
"""Vendored upstream vLLM modules for the vllm 0.23.x compat shim.

**NOT FOR UPSTREAM** -- these copies exist only so the VERL delta
weight-transfer backends can run on a vLLM 0.23.x host (the deployed stack
of the Ascend NPU test environment), which predates the module below.

``checkpoint_weight_patch`` is a byte-identical copy of vLLM #50723
("Support sparse checkpoint updates through native weight loaders", commit
``236f78cc5c``), taken from ``vllm/model_executor/model_loader/
checkpoint_weight_patch.py`` at the anchored vLLM main commit
``29af8bd672`` (zero drift between the two). Verify with::

    cmp vllm/model_executor/model_loader/checkpoint_weight_patch.py \\
        verl/workers/rollout/vllm_rollout/_vendored/checkpoint_weight_patch.py

``delta_weight_transfer._checkpoint_patch_api`` always prefers the native
module when the installed vLLM provides it; the vendored copy is a fallback
only. See ``delta_weight_transfer_v023_compat`` for the full compat story.
"""
