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
"""Fault injector for the rollout-restart e2e demo: SIGKILL one vLLM EngineCore.

Scans the process table for the engine cores spawned by the rollout replica's
server actor (process title/cmdline contains "EngineCore"), optionally waits
until a valid checkpoint exists (so a route-A recovery has a weight source),
settles for a configurable window (to land mid-rollout with requests in
flight), then kills one engine core and records its identity for the verifier.

The identity record (pid, create_time, cmdline) lets the verifier prove the
old process is really gone after recovery — PID reuse cannot spoof it.
"""

import argparse
import json
import sys
import time

import psutil


def find_engine_cores():
    matches = []
    for proc in psutil.process_iter(["pid", "create_time", "cmdline", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.pid == psutil.Process().pid:
            continue
        if "enginecore" in name or "enginecore" in cmdline:
            matches.append(proc)
    return matches


def wait_for_checkpoint(checkpoint_root, timeout_s):
    import os

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.isdir(checkpoint_root):
            for entry in os.listdir(checkpoint_root):
                if entry.startswith("global_step_"):
                    step_dir = os.path.join(checkpoint_root, entry)
                    for manifest in (
                        os.path.join(step_dir, "actor", "ckpt_contents.json"),
                        os.path.join(step_dir, "actor", "model", "ckpt_contents.json"),
                        os.path.join(step_dir, "manifest.json"),
                    ):
                        if os.path.isfile(manifest):
                            return entry
        time.sleep(5.0)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-checkpoint", default=None, help="directory to poll for a committed global_step_* save")
    parser.add_argument("--checkpoint-timeout", type=float, default=900.0)
    parser.add_argument("--wait-cores", type=int, default=1, help="wait until at least N EngineCore processes exist")
    parser.add_argument(
        "--settle-s", type=float, default=15.0, help="seconds to wait after all gates pass (mid-rollout window)"
    )
    parser.add_argument("--kill-count", type=int, default=1, help="number of engine cores to kill")
    parser.add_argument("--out", default="injection_record.json", help="where to write the identity record")
    parser.add_argument("--core-timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.wait_checkpoint:
        step = wait_for_checkpoint(args.wait_checkpoint, args.checkpoint_timeout)
        if step is None:
            print(f"INJECTOR: no committed checkpoint appeared under {args.wait_checkpoint} in time", file=sys.stderr)
            sys.exit(2)
        print(f"INJECTOR: checkpoint {step} is committed; weight source for route A is available")

    deadline = time.monotonic() + args.core_timeout
    while True:
        cores = find_engine_cores()
        if len(cores) >= args.wait_cores:
            break
        if time.monotonic() > deadline:
            print(f"INJECTOR: only {len(cores)} EngineCore process(es) found; giving up", file=sys.stderr)
            sys.exit(2)
        time.sleep(5.0)
    print(f"INJECTOR: {len(cores)} EngineCore process(es) alive; settling {args.settle_s}s before the kill")

    time.sleep(args.settle_s)

    cores = find_engine_cores()
    if not cores:
        print("INJECTOR: engine cores disappeared before the kill", file=sys.stderr)
        sys.exit(2)

    killed = []
    for proc in cores[: args.kill_count]:
        record = {
            "pid": proc.pid,
            "create_time": proc.create_time(),
            "cmdline": proc.info.get("cmdline") if hasattr(proc, "info") else None,
        }
        proc.kill()
        killed.append(record)
        print(f"INJECTOR: SIGKILL sent to EngineCore pid={proc.pid} create_time={record['create_time']:.3f}")

    with open(args.out, "w") as f:
        json.dump({"killed": killed, "ts": time.time()}, f, indent=2)
    print(f"INJECTOR: identity record written to {args.out}")


if __name__ == "__main__":
    main()
