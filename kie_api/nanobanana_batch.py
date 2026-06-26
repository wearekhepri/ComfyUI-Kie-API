"""Parallel batch runner for Nano Banana Pro (KEPRI / wearekhepri fork).

ComfyUI processes a list input serially (one node execution per element). KIE
runs remotely (no local GPU), and each job spends almost all of its time asleep
in the recordInfo poll loop — so firing many jobs at once via threads collapses
total wall-clock time from sum(jobs) to roughly max(jobs).

This reuses the upstream `run_nanobanana_image_job` unchanged (validation,
upload, createTask, poll, download, retry-on-429). We only add the fan-out.

Concurrency is bounded by `max_concurrency`; KIE's limit is on the submission
rate (~20 new requests / 10s per account), not on running tasks, and 429 is
already retried per-job — keep the default conservative since dev shares the
account budget with prod.
"""

from __future__ import annotations

import concurrent.futures

import torch

from .log import _log
from .nanobanana import run_nanobanana_image_job


def run_nanobanana_pro_batch(
    images_list: list[torch.Tensor],
    prompts: list[str],
    aspect_ratio: str,
    resolution: str,
    output_format: str,
    log: bool,
    poll_interval_s: float,
    timeout_s: int,
    max_concurrency: int,
) -> list[torch.Tensor]:
    """Run one Nano Banana Pro job per image, in parallel, preserving order.

    Args:
        images_list: per-job reference image tensors (e.g. one per folder image).
        prompts: per-image prompts; if shorter than images_list, the last prompt
            is broadcast to the remaining jobs (so a single prompt applies to all).
    Returns:
        A list of (1, H, W, 3) tensors aligned with `images_list`. A failed job
        yields a small black placeholder so one failure never kills the batch.
    """
    n = len(images_list)
    workers = max(1, min(int(max_concurrency), n))
    _log(log, f"[KIE Batch] Starting {n} job(s), up to {workers} in parallel...")

    def _one(i: int) -> torch.Tensor:
        prompt = prompts[i] if i < len(prompts) else prompts[-1]
        return run_nanobanana_image_job(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            log=log,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            images=images_list[i],
        )

    results: list[torch.Tensor | None] = [None] * n
    failures: list[tuple[int, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_one, i): i for i in range(n)}
        done = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            done += 1
            try:
                results[i] = future.result()
                _log(log, f"[KIE Batch] {done}/{n} done (job {i + 1} ok)")
            except Exception as exc:  # one bad job must not kill the whole batch
                failures.append((i, str(exc)))
                results[i] = torch.zeros((1, 64, 64, 3))
                _log(True, f"[KIE Batch] {done}/{n} — job {i + 1} FAILED: {exc}")

    if failures:
        failed_ids = [i + 1 for i, _ in failures]
        _log(True, f"[KIE Batch] Completed with {len(failures)} failure(s): jobs "
                   f"{failed_ids} (black placeholder returned for those).")
    else:
        _log(log, f"[KIE Batch] All {n} job(s) completed successfully.")

    return [r if r is not None else torch.zeros((1, 64, 64, 3)) for r in results]
