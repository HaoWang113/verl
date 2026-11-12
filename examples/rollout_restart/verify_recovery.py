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
"""Recovery verifier for the rollout-restart e2e demo.

Reads the trainer driver log plus the injector's identity record and asserts
the expected outcome:

- expect=recover (restart.enabled=true): the supervisor logged the full
  detection->rebuild->re-serve timeline, the job ran to its final step after
  the injection, and every killed engine-core process is really gone
  (pid + create_time identity; PID reuse cannot spoof it).
- expect=crash (restart.enabled=false, the control group): the job died —
  the current behavior when an engine core is SIGKILLed.

Exit code 0 = all assertions passed; nonzero with a FAIL summary otherwise.
"""

import argparse
import json
import re
import sys

import psutil

RECOVERY_MARKERS = [
    ("detection", re.compile(r"replica \S+ confirmed dead")),
    ("rebuild", re.compile(r"route [AB]|relaunched|recovered via route")),
    ("re-serving", re.compile(r"recovered via route [AB]|SERVING")),
]
STEP_MARKERS = re.compile(r"step:\s*(\d+)/(\d+)|global[_ ]step[=:]\s*(\d+)")
FINAL_STEP_RE = re.compile(r"total_training_steps", re.IGNORECASE)


def read_log(path):
    with open(path, errors="replace") as f:
        return f.read()


def verify_processes_gone(record_path):
    with open(record_path) as f:
        record = json.load(f)
    gone, alive = [], []
    for entry in record["killed"]:
        try:
            proc = psutil.Process(entry["pid"])
            if abs(proc.create_time() - entry["create_time"]) <= 1e-6:
                alive.append(entry["pid"])
            else:
                gone.append(entry["pid"])  # PID was recycled; the original died.
        except psutil.NoSuchProcess:
            gone.append(entry["pid"])
    return gone, alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="trainer driver log file")
    parser.add_argument("--record", default=None, help="injector identity record (injection_record.json)")
    parser.add_argument("--expect", choices=["recover", "crash"], default="recover")
    parser.add_argument("--final-step", type=int, default=None, help="expected final training step number")
    args = parser.parse_args()

    log = read_log(args.log)
    failures = []
    steps = []

    for match in STEP_MARKERS.finditer(log):
        for group in match.groups():
            if group is not None:
                steps.append(int(group))

    if args.expect == "recover":
        for label, pattern in RECOVERY_MARKERS:
            if not pattern.search(log):
                failures.append(f"missing {label} marker in the driver log")
        if args.final_step is not None:
            if not steps or max(steps) < args.final_step:
                seen = max(steps) if steps else None
                failures.append(f"job did not reach step {args.final_step} after the injection (max seen: {seen})")
        if args.record:
            gone, alive = verify_processes_gone(args.record)
            if alive:
                failures.append(f"killed engine-core process(es) still alive with matching identity: {alive}")
            print(f"VERIFY: killed engine processes confirmed gone: {gone}")
        print(f"VERIFY: highest training step observed in log: {max(steps) if steps else None}")
    else:
        # Control group: the current behavior is that the job dies on engine death.
        crashed = any(p.search(log) for _, p in RECOVERY_MARKERS) is False and (
            "ActorDiedError" in log or "EngineDeadError" in log or "Traceback" in log
        )
        if not crashed:
            failures.append("control group (restart disabled) unexpectedly survived without engine-failure markers")

    if failures:
        print("VERIFY: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"VERIFY: PASS (expect={args.expect})")


if __name__ == "__main__":
    main()
