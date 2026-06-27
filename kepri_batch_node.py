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
from .kie_api.nanobanana_multi import run_nanobanana_multi_image_batch


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


class KIE_NanoBananaPro_BestOfN:
    HELP = """
KIE Nano Banana Pro (Best-of-N)

Run the SAME prompt + SAME reference image(s) N times in parallel, to pick the
best result. Nano Banana Pro's API has no "number of images" parameter (one task
= one image), so N independent jobs are fired concurrently and their outputs
collected. Each job is stochastic on KIE's side, so the N results differ.

Use it when you have one good collage/prompt and want several candidates to
choose from — not for processing different images (that's the (Batch) node).

Inputs:
- prompt: Text prompt, applied identically to every variation.
- image: Reference image(s) for the edit — the SAME set is sent to every
  variation (e.g. your placed-necklace collage). Optional: omit for pure
  text-to-image best-of-N.
- num_variations: How many candidates to generate (default 4).
- aspect_ratio / resolution / output_format: same as the single node.
- max_concurrency: Max jobs in flight (default 4). KIE limits the submission
  RATE (~20 new requests / 10s per account, shared with prod) and 429 is
  auto-retried — keep it moderate.
- poll_interval_s / timeout_s: status poll cadence and per-job max wait.
- log: Console logging on/off.

Outputs:
- IMAGE (list): the N candidates, in job order. A failed job returns a small
  black placeholder so one failure never kills the set.

Note: the reference image is uploaded once per job (N uploads for N variations).
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt": ("STRING", {"multiline": True})},
            "optional": {
                "image": ("IMAGE",),
                "num_variations": ("INT", {"default": 4, "min": 1, "max": 16}),
                "aspect_ratio": ("COMBO", {"options": ASPECT_RATIO_OPTIONS, "default": "auto"}),
                "resolution": ("COMBO", {"options": RESOLUTION_OPTIONS, "default": "1K"}),
                "output_format": ("COMBO", {"options": OUTPUT_FORMAT_OPTIONS, "default": "png"}),
                "max_concurrency": ("INT", {"default": 4, "min": 1, "max": 32}),
                "poll_interval_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0}),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "log": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_variations"
    CATEGORY = "kie/api"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Each run is a fresh set of stochastic candidates, and the remote result
        # must never be cached (NaN != NaN → ComfyUI always re-executes).
        return float("nan")

    def generate_variations(
        self,
        prompt,
        image=None,
        num_variations=4,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        max_concurrency=4,
        poll_interval_s=5.0,
        timeout_s=600,
        log=True,
    ):
        num_variations = max(1, int(num_variations))
        # Same reference set sent to every variation (None = text-to-image).
        images_list = [image] * num_variations
        results = run_nanobanana_pro_batch(
            images_list=images_list,
            prompts=[prompt],  # single prompt → broadcast to all variations
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            log=bool(log),
            poll_interval_s=float(poll_interval_s),
            timeout_s=int(timeout_s),
            max_concurrency=min(int(max_concurrency), num_variations),
        )
        return (results,)


class KIE_NanoBananaPro_MultiImage:
    HELP = """
KIE Nano Banana Pro (Multi-Image)

Send several SEPARATE reference images (each its own size) + one prompt to Nano
Banana Pro. Unlike the upstream (Image) node — which takes one IMAGE socket and
so needs all references pre-batched at the same size — each socket here is
uploaded independently, so you can mix e.g. a necklace shot, a model shot and a
style reference without resizing them to match.

Batch / best-of-N built in: set num_variations > 1 to run the SAME references +
prompt several times in parallel and pick the best.

Inputs:
- prompt: Text prompt (required).
- image_1 / image_2 / image_3 / image_4: reference images, any size, all
  optional. Connect the ones you need (Nano Banana Pro blends up to 8 total; a
  socket carrying a batch counts each frame).
- num_variations: How many candidates to generate per run (default 1 = one
  image; set 4 for best-of-N).
- max_concurrency: Max variations in flight (default 4). KIE limits the
  submission RATE (~20 req / 10s per account, shared with prod); 429 auto-retried.
- aspect_ratio / resolution / output_format: same as the single node.
- poll_interval_s / timeout_s / log: as usual.

Outputs:
- IMAGE (list): the generated candidate(s). A failed variation returns a small
  black placeholder so one failure never kills the set.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt": ("STRING", {"multiline": True})},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "num_variations": ("INT", {"default": 1, "min": 1, "max": 16}),
                "aspect_ratio": ("COMBO", {"options": ASPECT_RATIO_OPTIONS, "default": "auto"}),
                "resolution": ("COMBO", {"options": RESOLUTION_OPTIONS, "default": "1K"}),
                "output_format": ("COMBO", {"options": OUTPUT_FORMAT_OPTIONS, "default": "png"}),
                "max_concurrency": ("INT", {"default": 4, "min": 1, "max": 32}),
                "poll_interval_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0}),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "log": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_multi"
    CATEGORY = "kie/api"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Remote, non-deterministic result — never serve a cached output.
        return float("nan")

    def generate_multi(
        self,
        prompt,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        num_variations=1,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        max_concurrency=4,
        poll_interval_s=5.0,
        timeout_s=600,
        log=True,
    ):
        ref_tensors = [t for t in (image_1, image_2, image_3, image_4) if t is not None]
        if not ref_tensors:
            raise RuntimeError(
                "KIE Nano Banana Pro (Multi-Image): connect at least one image "
                "(image_1…image_4), or use the (Image) node for text-to-image."
            )
        num_variations = max(1, int(num_variations))
        results = run_nanobanana_multi_image_batch(
            ref_tensors=ref_tensors,
            prompt=prompt,
            num_variations=num_variations,
            max_concurrency=min(int(max_concurrency), num_variations),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            log=bool(log),
            poll_interval_s=float(poll_interval_s),
            timeout_s=int(timeout_s),
        )
        return (results,)


NODE_CLASS_MAPPINGS = {
    "KIE_NanoBananaPro_Batch": KIE_NanoBananaPro_Batch,
    "KIE_NanoBananaPro_BestOfN": KIE_NanoBananaPro_BestOfN,
    "KIE_NanoBananaPro_MultiImage": KIE_NanoBananaPro_MultiImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KIE_NanoBananaPro_Batch": "KIE Nano Banana Pro (Batch)",
    "KIE_NanoBananaPro_BestOfN": "KIE Nano Banana Pro (Best-of-N)",
    "KIE_NanoBananaPro_MultiImage": "KIE Nano Banana Pro (Multi-Image)",
}
