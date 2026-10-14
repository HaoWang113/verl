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
"""Unit tests for the delta wire checksum (device-agnostic XOR tree).

Runs on CPU anywhere torch is installed, and on NPU/GPU devices when visible;
the checksum is a pure function of tensor bytes, so these assertions hold on
every device.
"""

import pytest
import torch

from verl.checkpoint_engine.delta_sync.encode import _xor_tree, checksum


def _ref_xor(bytes_like) -> int:
    acc = 0
    for b in bytes_like:
        acc ^= b
    return acc


def test_empty_arguments_checksum_to_zero():
    pos = torch.empty(0, dtype=torch.uint8)
    val = torch.empty(0, dtype=torch.bfloat16)
    assert _xor_tree(pos) == 0
    assert checksum(pos, val) == 0


def test_matches_bytewise_reference():
    for n in [1, 2, 3, 7, 8, 9, 31, 64, 100, 4097]:
        pos = torch.randint(0, 256, (n,), dtype=torch.uint8)
        val = torch.randn(n + 13).to(torch.bfloat16)
        assert _xor_tree(pos) == _ref_xor(pos.numpy().tolist())
        assert _xor_tree(val) == _ref_xor(val.view(torch.uint8).numpy().tolist())


def test_positions_and_values_combine_deterministically():
    pos = torch.tensor([1, 2, 3], dtype=torch.uint8)
    val = torch.tensor([4.0], dtype=torch.bfloat16)
    expected = (_ref_xor([1, 2, 3]) ^ (_ref_xor(val.view(torch.uint8).numpy().tolist()) << 1)) & ((1 << 64) - 1)
    assert checksum(pos, val) == expected
    assert checksum(pos, val) == checksum(pos.clone(), val.clone())


def test_single_bit_flip_is_detected():
    for which in ["pos", "val"]:
        pos = torch.randint(0, 256, (97,), dtype=torch.uint8)
        val = torch.randn(33).to(torch.bfloat16)
        before = checksum(pos, val)
        if which == "pos":
            pos = pos.clone()
            pos[41] ^= 0x01
        else:
            val = val.clone()
            raw = val.view(torch.uint8)
            raw[5] ^= 0x80
        assert checksum(pos, val) != before, f"flip in {which} went undetected"


def test_dtype_view_invariance():
    # same bytes, different dtype interpretation -> same checksum
    raw = torch.randint(0, 256, (64,), dtype=torch.uint8)
    as_bf16 = raw.view(torch.bfloat16)
    assert _xor_tree(raw) == _xor_tree(as_bf16)
    assert _xor_tree(as_bf16.contiguous()) == _xor_tree(as_bf16.reshape(-1, 4).contiguous())


def test_input_storage_not_mutated():
    pos = torch.randint(0, 256, (33,), dtype=torch.uint8)
    val = torch.randn(17).to(torch.bfloat16)
    pos_before, val_before = pos.clone(), val.clone()
    checksum(pos, val)
    assert torch.equal(pos, pos_before)
    assert torch.equal(val.view(torch.uint8), val_before.view(torch.uint8))


def test_large_tensor_against_reference():
    pos = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8)
    val = torch.randn(1 << 18).to(torch.bfloat16)
    assert _xor_tree(pos) == _ref_xor(pos.numpy().tolist())
    assert _xor_tree(val) == _ref_xor(val.view(torch.uint8).numpy().tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_checksum_agrees_across_cpu_and_cuda():
    pos = torch.randint(0, 256, (1000,), dtype=torch.uint8)
    val = torch.randn(777).to(torch.bfloat16)
    assert checksum(pos, val) == checksum(pos.cuda(), val.cuda())


if __name__ == "__main__":
    pytest.main([__file__])
