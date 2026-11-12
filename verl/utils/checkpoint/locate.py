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
"""Checkpoint locator for the rollout restart route A (design §7.5).

Pure filesystem function, shared by the rollout restart supervisor and usable
by trainer resume tooling. The commit point of a checkpoint is its manifest —
``ckpt_contents.json`` for Megatron (rank 0 writes it last, layout schema v2)
and ``manifest.json`` for FSDP (written at the end of the save, restart-gated).
A crash landing mid-save leaves the manifest absent, so the descending scan
simply skips the half-written directory and falls back to the previous
complete one (design §6.3, scenario S4/S6).
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")


class CheckpointUnavailable(FileNotFoundError):
    """No valid (manifest-passing) checkpoint under the given root."""


def _step_number(name: str) -> int | None:
    match = _STEP_DIR_RE.match(name)
    return int(match.group(1)) if match else None


def _hf_export_ready(hf_dir: str) -> bool:
    """An engine-cold-startable HF export: config + tokenizer + safetensors shards."""
    if not os.path.isdir(hf_dir):
        return False
    names = set(os.listdir(hf_dir))
    if "config.json" not in names:
        return False
    if not any(name.endswith(".safetensors") for name in names):
        return False
    return any(name.startswith("tokenizer") for name in names) or os.path.isfile(
        os.path.join(hf_dir, "tokenizer_config.json")
    )


def _validate_step_dir(step_dir: str) -> str | None:
    """Return the HF export path if this step dir carries a committed, complete save."""
    # Megatron layout v2 variants: the manifest sits at actor/model/ or
    # directly under actor/ (rank-0 layout difference between save paths),
    # while the HF export lives under actor/model/huggingface/.
    for manifest_dir in (os.path.join("actor", "model"), "actor"):
        base = os.path.join(step_dir, manifest_dir)
        if not os.path.isfile(os.path.join(base, "ckpt_contents.json")):
            continue
        hf_dir = os.path.join(step_dir, "actor", "model", "huggingface")
        if _hf_export_ready(hf_dir):
            return hf_dir
    # FSDP layout: actor/huggingface/ committed by global_step_N/manifest.json
    if os.path.isfile(os.path.join(step_dir, "manifest.json")):
        hf_dir = os.path.join(step_dir, "actor", "huggingface")
        if _hf_export_ready(hf_dir):
            return hf_dir
    return None


def locate_latest_valid_checkpoint(root: str) -> tuple[str, int]:
    """Locate the latest valid checkpoint's HF export directory.

    Args:
        root: The checkpoint root containing ``global_step_N`` directories and
            ``latest_checkpointed_iteration.txt``.

    Returns:
        ``(model_path, global_step)`` — the HF export directory suitable for an
        engine cold start, and its step number.

    Raises:
        CheckpointUnavailable: no candidate passes validation (early window
            before the first save, transiently unreachable storage, or every
            candidate structurally broken — the caller degrades to route B).
    """
    if not root or not os.path.isdir(root):
        raise CheckpointUnavailable(f"checkpoint root does not exist or is not a directory: {root!r}")

    # The tracker's step is the primary candidate: if it validates, it wins even
    # when a fresher directory exists (tracker lag only makes the candidate one
    # version older — accepted per design D6). A broken tracker step falls back
    # to the descending directory scan.
    tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
    tracked_step = None
    if os.path.isfile(tracker):
        try:
            with open(tracker) as f:
                tracked_step = int(f.read().strip())
        except (OSError, ValueError):
            logger.warning("unreadable tracker file %s; falling back to directory scan", tracker)
    if tracked_step is not None:
        hf_dir = _validate_step_dir(os.path.join(root, f"global_step_{tracked_step}"))
        if hf_dir is not None:
            return hf_dir, tracked_step
        logger.info("tracker step %d failed validation; scanning directories", tracked_step)

    steps = sorted(
        (step for step, _name in ((_step_number(name), name) for name in os.listdir(root)) if step is not None),
        reverse=True,
    )
    for step in steps:
        if step == tracked_step:
            continue
        step_dir = os.path.join(root, f"global_step_{step}")
        if not os.path.isdir(step_dir):
            continue
        hf_dir = _validate_step_dir(step_dir)
        if hf_dir is not None:
            return hf_dir, step
        logger.info("checkpoint step %d failed validation (incomplete or uncommitted); scanning older", step)

    raise CheckpointUnavailable(f"no valid checkpoint with an HF export under {root!r}")


def read_manifest(step_dir: str) -> dict:
    """Read the commit manifest of a step dir (either backend's form)."""
    for name in ("actor/model/ckpt_contents.json", "actor/ckpt_contents.json", "manifest.json"):
        path = os.path.join(step_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    raise CheckpointUnavailable(f"no manifest found under {step_dir!r}")
