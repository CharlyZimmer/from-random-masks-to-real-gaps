"""Read best-epoch train/val loss out of a Lightning run's logs.

Used by both ``esdc`` and ``sentinel`` ``evaluate_gapfilling.py`` scripts to
back-fill the training-loss / validation-loss columns of the results tables
*without* re-running model inference (see ``--training_metrics_only``).

Why two sources:

* The **validation loss** of the kept checkpoint is stored inside the ``.ckpt``
  itself, as ``ModelCheckpoint`` callback state (``best_model_score`` — the
  monitored ``MAE_val`` at the epoch that produced the lowest value, which is
  exactly the checkpoint the eval used).
* The matching **training loss** is *not* in the checkpoint. Lightning only
  persists callback/optimizer state, not arbitrary logged scalars. It is read
  from the TensorBoard event files (``events.out.tfevents.*``) that sit next to
  the checkpoint in the same ``version_<n>/`` directory, taking the ``MAE_epoch``
  scalar at the same step as the best ``MAE_val_epoch``.

The metric key is generic: training logs ``f"{loss_type.upper()}"`` /
``f"{loss_type.upper()}_val"`` (``mae`` by default → ``MAE`` / ``MAE_val``), and
the ESDC runs additionally log Kelvin-rescaled variants ``"MAE Rescaled"`` /
``"MAE Rescaled_val"``.
"""
from __future__ import annotations

import os

import torch

# Tolerance when cross-checking the checkpoint's stored val score against the
# value found in the event files. They are the same number logged twice, so any
# real difference points at a mismatched epoch/run.
_VAL_MATCH_TOL = 1e-3


def checkpoint_identity(path: str) -> tuple[str, str]:
    """Return ``(version_dir_name, checkpoint_basename)`` for equality checks.

    Absolute checkpoint paths differ between the HPC (where the committed
    ``run_info.json`` was written) and wherever the refresh runs, but the
    ``.../version_<n>/checkpoints/<file>.ckpt`` tail is stable, and the basename
    encodes ``epoch=`` and ``MAE_val=`` so it uniquely pins the run + epoch.
    """
    path = str(path).rstrip("/")
    basename = os.path.basename(path)
    ckpt_dir = os.path.dirname(path)
    version_dir = os.path.dirname(ckpt_dir)
    return os.path.basename(version_dir), basename


def assert_same_checkpoint(resolved_ckpt: str, recorded_ckpt: str) -> None:
    """Abort unless ``resolved_ckpt`` is the same checkpoint as ``recorded_ckpt``.

    ``--training_metrics_only`` skips inference, so the numbers it writes must
    come from the exact checkpoint the committed evaluation used. Raises
    ``SystemExit`` (non-zero) with both paths on any mismatch.
    """
    got = checkpoint_identity(resolved_ckpt)
    want = checkpoint_identity(recorded_ckpt)
    if got != want:
        raise SystemExit(
            "ERROR: --training_metrics_only would read from a different "
            "checkpoint than the one used for the committed evaluation.\n"
            f"  resolved now      : {resolved_ckpt}\n"
            f"                      (version={got[0]}, file={got[1]})\n"
            f"  run_info.json says : {recorded_ckpt}\n"
            f"                      (version={want[0]}, file={want[1]})\n"
            "Pass --checkpoint explicitly or fix the log_dir so the two agree."
        )


def read_checkpoint_val_loss(ckpt_path: str) -> tuple[int | None, float, str]:
    """Return ``(epoch, val_loss, monitor)`` from a Lightning checkpoint.

    ``val_loss`` is the ``ModelCheckpoint`` ``best_model_score`` (the monitored
    metric at the kept epoch). Falls back to parsing ``MAE_val=`` from the
    filename if the callback state has no score.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch")

    monitor = "MAE_val"
    score = None
    for state in ck.get("callbacks", {}).values():
        if isinstance(state, dict) and state.get("best_model_score") is not None:
            monitor = state.get("monitor", monitor)
            score = float(state["best_model_score"])
            break

    if score is None:
        import re

        m = re.search(r"MAE_val=([0-9]+(?:\.[0-9]+)?)", os.path.basename(ckpt_path))
        if not m:
            raise ValueError(
                f"No best_model_score in checkpoint callbacks and no MAE_val= in "
                f"filename: {ckpt_path}"
            )
        score = float(m.group(1))

    return epoch, score, monitor


def training_loss_metrics(ckpt_path: str, kelvin: bool = False) -> dict:
    """High-level: return the metrics.json fragment for one checkpoint.

    ``{"train_loss_normalized": ..., "val_loss_normalized": ...}`` always, plus
    ``train_loss_kelvin`` / ``val_loss_kelvin`` when ``kelvin`` is set and the
    run logged the rescaled variants (ESDC). Raises on any problem — callers on
    the normal eval path should wrap this and downgrade to a warning.
    """
    epoch, val_loss, monitor = read_checkpoint_val_loss(ckpt_path)
    version_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    losses = read_best_epoch_losses(version_dir, epoch, val_loss, monitor)

    payload = {
        "train_loss_normalized": losses["train_loss_normalized"],
        "val_loss_normalized": losses["val_loss_normalized"],
    }
    if kelvin:
        tk, vk = losses["train_loss_kelvin"], losses["val_loss_kelvin"]
        payload["train_loss_kelvin"] = round(tk, 4) if tk is not None else None
        payload["val_loss_kelvin"] = round(vk, 4) if vk is not None else None
    return payload


def _scalars_by_step(ea, tag: str) -> dict[int, float]:
    return {ev.step: ev.value for ev in ea.Scalars(tag)}


def read_best_epoch_losses(
    version_dir: str,
    ckpt_epoch: int | None,
    val_loss: float,
    monitor: str = "MAE_val",
) -> dict:
    """Read best-epoch train/val loss from the TensorBoard events in ``version_dir``.

    ``version_dir`` is ``.../version_<n>/`` — the parent of ``checkpoints/`` and
    the directory the ``TensorBoardLogger`` wrote its event files into.

    Selection: among the logged epoch-end steps, take the one whose ``epoch``
    scalar equals ``ckpt_epoch`` and whose ``{base}_val_epoch`` is closest to
    ``val_loss``; if the ``epoch`` scalar is unavailable, fall back to the step
    whose ``{base}_val_epoch`` is closest to ``val_loss``.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"version dir not found: {version_dir}")

    ea = EventAccumulator(version_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    base = monitor[:-4] if monitor.endswith("_val") else "MAE"
    train_tag = f"{base}_epoch"
    val_tag = f"{base}_val_epoch"
    if train_tag not in tags or val_tag not in tags:
        raise ValueError(
            f"Event files in {version_dir} have no '{train_tag}'/'{val_tag}' "
            f"scalars (found: {sorted(tags)}). Cannot recover training loss."
        )

    train_ev = _scalars_by_step(ea, train_tag)
    val_ev = _scalars_by_step(ea, val_tag)

    sel = None
    if ckpt_epoch is not None and "epoch" in tags:
        epoch_at_step = {s: int(round(v)) for s, v in _scalars_by_step(ea, "epoch").items()}
        cands = [s for s in val_ev if epoch_at_step.get(s) == ckpt_epoch]
        if cands:
            sel = min(cands, key=lambda s: abs(val_ev[s] - val_loss))
    if sel is None:
        sel = min(val_ev, key=lambda s: abs(val_ev[s] - val_loss))

    if abs(val_ev[sel] - val_loss) > _VAL_MATCH_TOL:
        print(
            f"WARNING: closest logged {val_tag} ({val_ev[sel]:.6f} at step {sel}) "
            f"differs from the checkpoint's stored score ({val_loss:.6f}). "
            f"Reporting the event-file values anyway — check the run."
        )

    train_step = sel if sel in train_ev else min(train_ev, key=lambda s: abs(s - sel))

    result = {
        "train_loss_normalized": float(train_ev[train_step]),
        "val_loss_normalized": float(val_ev[sel]),
        "best_epoch": ckpt_epoch,
        "matched_step": int(sel),
        "source_events": os.path.abspath(version_dir),
    }

    for out_key, tag in (
        ("train_loss_kelvin", f"{base} Rescaled_epoch"),
        ("val_loss_kelvin", f"{base} Rescaled_val_epoch"),
    ):
        if tag in tags:
            by_step = _scalars_by_step(ea, tag)
            ref = train_step if out_key.startswith("train") else sel
            result[out_key] = float(by_step[ref]) if ref in by_step else None
        else:
            result[out_key] = None

    return result
