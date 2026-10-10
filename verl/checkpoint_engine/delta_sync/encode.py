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
"""On-wire schema for delta sync: per-parameter manifest, flush container, checksum.

One layout: a uint8 positions blob (``indices`` encoding -- int32 absolute
positions, 4 bytes / nnz) plus a parameter-dtype values tensor, described by a
per-parameter manifest. Values are sent verbatim in the parameter's dtype.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices"]


# ---------- diff ----------------------------------------------------------


@dataclass
class DeltaParam:
    """Per-parameter manifest entry for a single chunk / bucket.

    Offsets are byte offsets into the surrounding ``__positions__`` blob and
    element offsets into the surrounding ``__values__`` tensor.
    """

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


def _xor_tree(t: torch.Tensor) -> int:
    """Device-side XOR fold of a tensor's raw bytes (log2(n) rounds, one D2H).

    Replaces ``torch.hash_tensor`` (PyTorch >= 2.9, CPU/CUDA kernels only -- no
    NPU kernel, and its torch_npu CPU fallback costs a full D2H per call). The
    fold is out-of-place, so the argument's storage is never mutated, and it is
    a pure function of the bytes, so sender and receiver agree on any device.
    """
    if t.numel() == 0:
        return 0
    flat = t.contiguous().reshape(-1)
    v = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    # Halving XOR fold; an odd middle element carries to the next round.
    while v.numel() > 1:
        n = v.numel()
        half = n // 2
        if n % 2 == 0:
            v = v[:half] ^ v[half:]
        else:
            v = torch.cat((v[:half] ^ v[half + 1 :], v[half : half + 1]))
    return int(v.item())


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    XOR tree over each argument's bytes (device-side, one ``.item()`` sync per
    argument); the same function runs on both ends, so the value is stable
    across dtypes and devices.
    """
    p = _xor_tree(positions)
    v = _xor_tree(values)
    return (p ^ (v << 1)) & ((1 << 64) - 1)


# ---------- compact binary spec --------------------------------------------
# [Round 7] The flush spec rides the wire as JSON through the __delta_spec__
# byte tensor; at ~15k patches per flush the receiver paid json.dumps (pump) +
# H2D + D2H + json.loads (decode) ~= 2.5-3s per flush (round 6 decode=2.79s).
# pack_spec/unpack_spec replace the text round-trip with a fixed-layout binary
# blob (~6x smaller, parse-free); decoders sniff the magic and fall back to
# JSON, so a mixed-version producer keeps working.

_SPEC_MAGIC = b"DSB1"
_SPEC_HEADER = "<4sBBqi"  # magic, enc_len, flags(bit0 verify | bit1 is_last), checksum, n_params
_SPEC_PARAM_HEAD = "<HBb"  # name_len, dtype_len, ndim
_SPEC_PARAM_TAIL = "<qqbqq"  # pos_start, pos_end, pos_width, val_start, val_end


def pack_spec(spec: dict) -> bytes:
    """Encode one flush spec dict as the DSB1 binary blob; any unexpected shape
    falls back to the JSON encoding (decoders sniff the magic either way)."""
    try:
        enc = str(spec["encoding"]).encode()
        flags = (1 if spec.get("verify") else 0) | (2 if spec.get("is_last") else 0)
        params = spec["params"]
        out = [struct.pack(_SPEC_HEADER, _SPEC_MAGIC, len(enc), flags, int(spec["checksum"]), len(params)), enc]
        for p in params:
            name = str(p["name"]).encode()
            dtype = str(p["dtype"]).encode()
            shape = [int(x) for x in p["shape"]]
            out.append(struct.pack(_SPEC_PARAM_HEAD, len(name), len(dtype), len(shape)))
            out.append(name)
            out.append(dtype)
            if shape:
                out.append(struct.pack(f"<{len(shape)}i", *shape))
            out.append(
                struct.pack(
                    _SPEC_PARAM_TAIL,
                    int(p["pos_start"]),
                    int(p["pos_end"]),
                    int(p["pos_width"]),
                    int(p["val_start"]),
                    int(p["val_end"]),
                )
            )
        return b"".join(out)
    except (KeyError, TypeError, ValueError, struct.error):
        return json.dumps(spec).encode()


def unpack_spec(data: bytes) -> dict:
    """Decode a DSB1 blob back into the spec dict (same keys as the JSON form,
    including per-param dicts, so existing consumers are unchanged)."""
    _magic, enc_len, flags, checksum, n = struct.unpack_from(_SPEC_HEADER, data, 0)
    off = struct.calcsize(_SPEC_HEADER)
    enc = data[off : off + enc_len].decode()
    off += enc_len
    params = []
    head = struct.calcsize(_SPEC_PARAM_HEAD)
    tail = struct.calcsize(_SPEC_PARAM_TAIL)
    for _ in range(n):
        name_len, dtype_len, ndim = struct.unpack_from(_SPEC_PARAM_HEAD, data, off)
        off += head
        name = data[off : off + name_len].decode()
        off += name_len
        dtype = data[off : off + dtype_len].decode()
        off += dtype_len
        dims = struct.unpack_from(f"<{ndim}i", data, off) if ndim else ()
        off += 4 * ndim
        pos_start, pos_end, pos_width, val_start, val_end = struct.unpack_from(_SPEC_PARAM_TAIL, data, off)
        off += tail
        params.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(dims),
                "pos_start": pos_start,
                "pos_end": pos_end,
                "pos_width": pos_width,
                "val_start": val_start,
                "val_end": val_end,
            }
        )
    return {
        "encoding": enc,
        "verify": bool(flags & 1),
        "is_last": bool(flags & 2),
        "params": params,
        "checksum": checksum,
    }


# ---------- encode --------------------------------------------------------


@dataclass
class DeltaFlush:
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. Despite the name it lives on
      the GPU in the sharded engine (the wire broadcasts from the GPU, so a
      host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        return self.values_gpu.numel()

    @property
    def wire_bytes(self) -> int:
        return self.positions_cpu.numel() + self.values_gpu.numel() * self.values_gpu.element_size()
