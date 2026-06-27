"""Multi-reference single job for Nano Banana Pro (KEPRI / wearekhepri fork).

The upstream node takes references as ONE ComfyUI IMAGE tensor ([B, H, W, 3]),
so every reference must share the same H×W (ComfyUI can't batch ragged sizes).
Nano Banana Pro itself uploads each reference separately and accepts up to 8 of
ANY size — ideal for mixing e.g. a necklace shot + a model shot + a style ref.

This helper takes a LIST of tensors (one per socket, each its own size), uploads
each, and fires ONE Nano Banana Pro task with all of them in `image_input`. It
composes the same upstream pieces as `run_nanobanana_image_job` (validation,
upload, createTask, poll, download, retry-on-429) — nothing upstream is modified.
"""

from __future__ import annotations

import concurrent.futures
import time

import torch

from .auth import _load_api_key
from .credits import _log_remaining_credits
from .http import TransientKieError
from .images import _download_image, _image_bytes_to_tensor
from .jobs import _poll_task_until_complete
from .log import _log
from .nanobanana import (
    ASPECT_RATIO_OPTIONS,
    MODEL_NAME,
    OUTPUT_FORMAT_OPTIONS,
    PROMPT_MAX_LENGTH,
    RESOLUTION_OPTIONS,
    _create_nano_banana_task,
)
from .results import _extract_result_urls
from .upload import _image_tensor_to_png_bytes, _truncate_url, _upload_image
from .validation import _validate_prompt

MAX_REFERENCES = 8


def run_nanobanana_multi_image_job(
    prompt: str,
    ref_tensors: list[torch.Tensor],
    aspect_ratio: str = "auto",
    resolution: str = "1K",
    output_format: str = "png",
    log: bool = True,
    poll_interval_s: float = 5.0,
    timeout_s: int = 600,
    retry_on_fail: bool = True,
    max_retries: int = 2,
    retry_backoff_s: float = 3.0,
) -> torch.Tensor:
    """Run one Nano Banana Pro job with several separately-uploaded references.

    Args:
        ref_tensors: list of ComfyUI IMAGE tensors ([B, H, W, 3]); each socket's
            image, any size. Slices are uploaded individually, capped at 8 total.
    Returns:
        A torch tensor of shape (1, H, W, 3), float values in [0, 1].
    """
    _validate_prompt(prompt, max_length=PROMPT_MAX_LENGTH)
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        raise RuntimeError("Invalid aspect_ratio. Use the pinned enum options.")
    if resolution not in RESOLUTION_OPTIONS:
        raise RuntimeError("Invalid resolution. Use the pinned enum options.")
    if output_format not in OUTPUT_FORMAT_OPTIONS:
        raise RuntimeError("Invalid output_format. Use the pinned enum options.")

    # Flatten the provided sockets into a flat list of single-image tensors.
    slices: list[torch.Tensor] = []
    for t in ref_tensors:
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            raise RuntimeError("Each image input must be a tensor batch.")
        if t.dim() != 4 or t.shape[-1] != 3:
            raise RuntimeError("Each image input must have shape [B, H, W, 3].")
        for idx in range(t.shape[0]):
            slices.append(t[idx])

    if len(slices) > MAX_REFERENCES:
        _log(log, f"More than {MAX_REFERENCES} references provided "
                  f"({len(slices)}); only the first {MAX_REFERENCES} will be used.")
        slices = slices[:MAX_REFERENCES]

    attempts = max(max_retries + 1 if retry_on_fail else 1, 1)
    backoff = retry_backoff_s if retry_backoff_s >= 0 else 0.0

    for attempt in range(1, attempts + 1):
        start_time = time.time()
        try:
            api_key = _load_api_key()

            image_urls: list[str] = []
            if slices:
                _log(log, f"Uploading {len(slices)} reference image(s)...")
            for i, sl in enumerate(slices):
                png_bytes = _image_tensor_to_png_bytes(sl)
                url = _upload_image(api_key, png_bytes)
                image_urls.append(url)
                _log(log, f"Reference {i + 1} upload success: {_truncate_url(url)}")

            payload = {
                "model": MODEL_NAME,
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "output_format": output_format,
                    "image_input": image_urls,
                },
            }

            _log(log, "Creating Nano Banana Pro task (multi-reference)...")
            task_id, create_response_text = _create_nano_banana_task(api_key, payload)
            _log(log, f"createTask response (elapsed={time.time() - start_time:.1f}s): {create_response_text}")
            _log(log, f"Task created with ID {task_id}. Polling...")

            record_data = _poll_task_until_complete(
                api_key, task_id, poll_interval_s, timeout_s, log, start_time,
            )
            result_urls = _extract_result_urls(record_data)
            _log(log, f"Result URLs: {result_urls}")

            image_bytes = _download_image(result_urls[0])
            image_tensor = _image_bytes_to_tensor(image_bytes)
            _log(log, "Image downloaded and decoded.")
            _log_remaining_credits(log, record_data, api_key, _log)
            return image_tensor
        except TransientKieError:
            if not retry_on_fail or attempt >= attempts:
                raise
            _log(log, f"Retrying (attempt {attempt + 1}/{attempts}) after {backoff}s")
            time.sleep(backoff)
            continue


def run_nanobanana_multi_image_batch(
    ref_tensors: list[torch.Tensor],
    prompt: str,
    num_variations: int,
    max_concurrency: int,
    aspect_ratio: str = "auto",
    resolution: str = "1K",
    output_format: str = "png",
    log: bool = True,
    poll_interval_s: float = 5.0,
    timeout_s: int = 600,
) -> list[torch.Tensor]:
    """Run the SAME multi-reference prompt N times in parallel (best-of-N).

    Every variation uses the identical `ref_tensors` set and prompt; only KIE's
    server-side stochasticity makes them differ. Returns a list of N tensors; a
    failed variation yields a small black placeholder so one failure never kills
    the set.
    """
    n = max(1, int(num_variations))
    workers = max(1, min(int(max_concurrency), n))
    _log(log, f"[KIE Multi] Starting {n} variation(s), up to {workers} in parallel...")

    def _one(_i: int) -> torch.Tensor:
        return run_nanobanana_multi_image_job(
            prompt=prompt,
            ref_tensors=ref_tensors,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            log=log,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
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
                _log(log, f"[KIE Multi] {done}/{n} done (variation {i + 1} ok)")
            except Exception as exc:  # one bad variation must not kill the set
                failures.append((i, str(exc)))
                results[i] = torch.zeros((1, 64, 64, 3))
                _log(True, f"[KIE Multi] {done}/{n} — variation {i + 1} FAILED: {exc}")

    if failures:
        failed_ids = [i + 1 for i, _ in failures]
        _log(True, f"[KIE Multi] Completed with {len(failures)} failure(s): "
                   f"variation(s) {failed_ids} (black placeholder returned).")
    else:
        _log(log, f"[KIE Multi] All {n} variation(s) completed successfully.")

    return [r if r is not None else torch.zeros((1, 64, 64, 3)) for r in results]
