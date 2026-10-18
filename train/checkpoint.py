"""Checkpoints: save a run's complete state and resume it exactly where it stopped.

Why
    Colab sessions disconnect without warning. A Checkpoint on Google Drive lets a
    new session pick up the run instead of starting over.

What "exactly" requires
    Resuming only reproduces the uninterrupted run if EVERYTHING that influences the
    next step is restored, not just the model weights:
      * model weights
      * optimizer state (AdamW's running averages of gradients: without them the
        first steps after resuming behave differently from the original run)
      * GradScaler state (the fp16 loss scale)
      * the step counter (the learning-rate schedule depends on it)
      * the random number generators: python, numpy, torch (CPU and every GPU), which
        drive dropout, plus the data loader's own generator, which decides which
        windows are drawn next
      * the loss/validation history so far, so the final report covers the whole run

Safety
    A checkpoint is written to a temporary file and then renamed into place. Renaming
    is atomic, so a session that dies mid-write leaves an ignorable `.tmp` file and the
    previous checkpoint intact, never a half-written "latest" one.
"""
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

_NAME = re.compile(r"^ckpt_(\d+)\.pt$")


def _path(checkpoint_dir, step: int) -> Path:
    # Zero-padded so that file names also sort correctly as text.
    return Path(checkpoint_dir) / f"ckpt_{step:08d}.pt"


def _checkpoints(checkpoint_dir) -> list[tuple[int, Path]]:
    """All complete checkpoints in the folder as (step, path), oldest first."""
    folder = Path(checkpoint_dir)
    if not folder.is_dir():
        return []
    found = []
    for path in folder.iterdir():
        match = _NAME.match(path.name)  # excludes ".tmp" partial files and anything else
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def latest_checkpoint(checkpoint_dir) -> Path | None:
    """The checkpoint with the highest step number, or None if there is none."""
    found = _checkpoints(checkpoint_dir)
    return found[-1][1] if found else None


def save_checkpoint(checkpoint_dir, step: int, state: dict, keep_last: int = 2) -> Path:
    """Write `state` as the checkpoint for `step`, then delete all but the newest `keep_last`.

    A GPT-2 small checkpoint is ~1.5 GB (weights + two AdamW moments per weight), so
    keeping only a few bounds the space used on Drive.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    final = _path(checkpoint_dir, step)
    partial = final.with_name(final.name + ".tmp")
    torch.save(state, partial)
    os.replace(partial, final)  # atomic: either the old state or the complete new one
    for _, old in _checkpoints(checkpoint_dir)[:-keep_last]:
        old.unlink()
    return final


def load_checkpoint(path) -> dict:
    """Read a checkpoint onto the CPU.

    Everything is loaded on the CPU on purpose: the random-number states must be CPU
    tensors, and the model/optimizer `load_state_dict` calls move values to the right
    device themselves. weights_only=False is needed because the checkpoint also holds
    plain python/numpy objects; only load checkpoints you wrote yourself.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def capture_rng_state() -> dict:
    """Snapshot every global random number generator."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Put every global random number generator back to a captured snapshot."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
