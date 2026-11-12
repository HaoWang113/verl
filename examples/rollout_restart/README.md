# Rollout-Restart E2E Demo (Phase 1)

Demonstrates the rollout-side independent restart design
(`reports/rollout_restart_design_v1.md`) on a single Ascend 910B node:
**4 NPUs Megatron training + 4 NPUs vLLM standalone rollout**, one vLLM
EngineCore process SIGKILLed mid-rollout, and the recovery verified end to end.

## What the demo shows

| Leg | `rollout.restart.enabled` | Expected outcome |
|---|---|---|
| `recover` | `true` | Detection (request-path evidence + sync-entry sweep) → CAS fence → unified replica rebuild (five-segment teardown, same-pool respawn) → route A cold start from the checkpoint HF export → VERIFYING → re-registration → training continues to the final step; killed engine process really gone |
| `control` | `false` | The same injection kills the job — the current upstream behavior, demonstrated live |

## Usage

```bash
# full demo (both legs)
bash examples/rollout_restart/run_e2e.sh

# one leg at a time
PHASE=recover  bash examples/rollout_restart/run_e2e.sh
PHASE=control  bash examples/rollout_restart/run_e2e.sh
```

Required environment (defaults overridable via env vars, see the script head):

- `MODEL_PATH` — HF model weights (default `Qwen/Qwen3.5-9B`; any Qwen-family
  model supported by both Megatron-Bridge and vllm-ascend works; swap to a
  smaller one if 4-NPU training is memory-tight)
- `TRAIN_FILE` / `VAL_FILE` — parquet prompt datasets (gsm8k format)

Artifacts land in a fresh `/tmp/rollout_restart_demo.*` directory: driver logs,
the injector's identity record (pid + create_time of the killed EngineCore),
and checkpoint roots per leg.

## Components

- `run_e2e.sh` — parameterized launcher (trainer + injector + verifier per leg)
- `inject_engine_kill.py` — waits for a committed checkpoint (`ckpt_contents.json`
  for Megatron saves), settles into the rollout window, SIGKILLs an EngineCore
  process, records its identity
- `verify_recovery.py` — asserts the recovery timeline markers in the driver
  log, that the job reached its final training step after the injection, and
  (by pid + create_time identity) that the killed process is gone

Unit tests (CPU, no accelerators): `tests/workers/rollout/rollout_restart/`.
Detailed test guidance for the dedicated test environment:
`reports/rollout_restart_test_guide.md`.
