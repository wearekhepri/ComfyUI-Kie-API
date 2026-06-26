"""KEPRI / wearekhepri fork addition: parallel batch node for KIE Nano Banana Pro.

The only thing this fork changes vs upstream: a batch node that processes a
list/folder of images in PARALLEL instead of ComfyUI's serial one-by-one.
Auth, jobs, upload, polling — all reused unchanged from the upstream package.

Wire the "Batch Images (Folder Loader)" (or any list of images) into `images`;
each image becomes the reference for one Nano Banana Pro job, all fired
concurrently (bounded by `max_concurrency`).
"""

from __future__ import annotations

from .kie_api.nanobanana import (
    ASPECT_RATIO_OPTIONS,
    OUTPUT_FORMAT_OPTIONS,
    RESOLUTION_OPTIONS,
)
from .kie_api.nanobanana_batch import run_nanobanana_pro_batch


def _first(value, default=None):
    """With INPUT_IS_LIST, every arg is delivered as a list — take the scalar."""
    if isinstance(value, list):
        return value[0] if value else default
    return value


class KIE_NanoBananaPro_Batch:
    HELP = """
KIE Nano Banana Pro (Batch)

Process a LIST of images in parallel through Nano Banana Pro. One job per input
image (each image = its reference), all submitted concurrently — far faster than
ComfyUI's serial list processing for API-backed nodes.

Inputs:
- prompt: Text prompt. Single = applied to every image; list = per-image.
- images: List/batch of reference images (e.g. Batch Images Folder Loader).
- aspect_ratio / resolution / output_format: same as the single node.
- max_concurrency: Max jobs in flight at once (default 6). KIE limits the
  submission RATE (~20 new requests / 10s per account), not running tasks, and
  429 is auto-retried — but the account is shared with prod, so stay moderate.
- poll_interval_s / timeout_s: status poll cadence and per-job max wait.
- log: Console logging on/off.

Outputs:
- IMAGE (list): one result per input image, in order. A failed job returns a
  small black placeholder so one bad image never kills the whole batch.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt": ("STRING", {"multiline": True})},
            "optional": {
                "images": ("IMAGE",),
                "aspect_ratio": ("COMBO", {"options": ASPECT_RATIO_OPTIONS, "default": "auto"}),
                "resolution": ("COMBO", {"options": RESOLUTION_OPTIONS, "default": "1K"}),
                "output_format": ("COMBO", {"options": OUTPUT_FORMAT_OPTIONS, "default": "png"}),
                "max_concurrency": ("INT", {"default": 6, "min": 1, "max": 32}),
                "poll_interval_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0}),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "log": ("BOOLEAN", {"default": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_batch"
    CATEGORY = "kie/api"

    def generate_batch(
        self,
        prompt,
        images=None,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        max_concurrency=6,
        poll_interval_s=5.0,
        timeout_s=600,
        log=True,
    ):
        # INPUT_IS_LIST: scalars arrive wrapped in single-element lists.
        aspect_ratio = _first(aspect_ratio, "auto")
        resolution = _first(resolution, "1K")
        output_format = _first(output_format, "png")
        max_concurrency = int(_first(max_concurrency, 6))
        poll_interval_s = float(_first(poll_interval_s, 5.0))
        timeout_s = int(_first(timeout_s, 600))
        log = bool(_first(log, True))

        # images: list of per-job reference tensors (folder loader → N images).
        if images is None:
            images_list = []
        elif isinstance(images, list):
            images_list = images
        else:
            images_list = [images]

        # prompt: list (per-image) or single (broadcast to all).
        prompts = prompt if isinstance(prompt, list) else [prompt]

        if not images_list:
            raise RuntimeError(
                "KIE Nano Banana Pro (Batch): no input images. Connect a list/batch "
                "of images (e.g. the Batch Images Folder Loader)."
            )

        results = run_nanobanana_pro_batch(
            images_list=images_list,
            prompts=prompts,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            log=log,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            max_concurrency=max_concurrency,
        )
        return (results,)


NODE_CLASS_MAPPINGS = {"KIE_NanoBananaPro_Batch": KIE_NanoBananaPro_Batch}
NODE_DISPLAY_NAME_MAPPINGS = {"KIE_NanoBananaPro_Batch": "KIE Nano Banana Pro (Batch)"}
