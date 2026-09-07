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
"""Receiver-side HCCL delta transport: zmq manifest subscription + PyHccl group.

Shared by the vLLM worker engine (the C1 production receiver) and direct
receivers used in tests. The SUB socket MUST be connected before the
communicator is built (rendezvous-as-ready-gate): the trainer publishes the
first manifest only after its lazily built communicator is complete, and a zmq
PUB socket drops messages published before any subscriber connected -- so
"communicator ready" on the trainer implies "every receiver subscribed".
"""

from __future__ import annotations

import json
from collections.abc import Callable

import torch
import zmq

from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import is_valid_ipv6_address


def connect_delta_manifest_sub(zmq_ip: str, zmq_port: int, topic: str = "bucket_metadata") -> zmq.Socket:
    """Subscribe to the trainer's flush-manifest PUB socket."""
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    if is_valid_ipv6_address(zmq_ip):
        sub.setsockopt(zmq.IPV6, 1)
        sub.connect(f"tcp://[{zmq_ip}]:{zmq_port}")
    else:
        sub.connect(f"tcp://{zmq_ip}:{zmq_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sub


def build_delta_hccl_communicator(dist_ip: str, dist_port: int, group_rank: int, world_size: int, device):
    """Join the trainer's PyHccl broadcast group at ``group_rank``."""
    return stateless_init_process_group(dist_ip, dist_port, group_rank, world_size, device)


def pump_hccl_delta_session(
    sub: zmq.Socket,
    comm,
    device,
    on_flush: Callable[[list[tuple[str, torch.Tensor]]], None],
) -> str:
    """Pull one delta session (manifests + broadcast payloads) and hand each
    flush to ``on_flush(named_tensors)``.

    Returns the session's wire encoding (``"dense"`` | ``"indices"``); a
    terminal-only session (steady sync with zero changes) returns
    ``"indices"`` with zero ``on_flush`` calls. The device synchronize after
    each flush's broadcasts mirrors the sender: PyHccl enqueues on the current
    stream, so the received buffers must not be read before completion.
    """
    synchronize = getattr(torch, device.type).synchronize
    encoding = "indices"
    while True:
        sub.recv_string()
        meta = sub.recv_pyobj()
        if meta.get("terminal_empty"):
            break
        encoding = meta.get("encoding", "indices")
        dense = encoding == "dense"
        val_dtype = getattr(torch, meta["val_dtype"])
        elem = torch.empty(0, dtype=val_dtype).element_size()
        val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device=device)
        if dense:
            pos = None
            comm.broadcast(val_u8, src=0)
        else:
            pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device=device)
            comm.broadcast(pos, src=0)
            comm.broadcast(val_u8, src=0)
        synchronize()
        val = val_u8.view(val_dtype)
        spec_bytes = json.dumps(meta["spec"]).encode()
        spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to(device)
        named = [("__delta_spec__", spec_t), ("__values__", val)]
        if pos is not None:
            named.insert(1, ("__positions__", pos))
        on_flush(named)
        del pos, val_u8, val, spec_t
        if meta["is_last"]:
            break
    return encoding
