# KIE Nano Banana Pro (Batch)

Process a **list of images in parallel** through Nano Banana Pro, instead of
ComfyUI's default serial (one-by-one) list processing.

This is a **wearekhepri fork addition** — it reuses the upstream Nano Banana Pro
logic unchanged (auth, upload, `createTask`, polling, retry-on-429) and only
adds concurrent fan-out via a thread pool. Useful for dev/test batches where the
single node is slow because each image waits for the previous one to finish.

Why it works: KIE runs remotely (no local GPU) and each job is mostly spent
*waiting* in the status poll, so firing many at once collapses total time from
the sum of all jobs to roughly the slowest single job.

---

## Inputs

- **prompt** — Text prompt. A single prompt is applied to every image; a list
  (per-image) is used position-by-position.
- **images** — A list/batch of reference images. Wire the **Batch Images (Folder
  Loader)** here; each image becomes the reference for one job.
- **aspect_ratio** — 1:1, 16:9, … (same options as the single node).
- **resolution** — 1K / 2K / 4K.
- **output_format** — png / jpg.
- **max_concurrency** — Max jobs in flight at once (default **6**). KIE limits
  the *submission rate* (~20 new requests / 10s **per account**), not the number
  of running tasks, and 429 is auto-retried. Keep it moderate if the account is
  shared with production.
- **poll_interval_s** — How often each job checks task status (default 5s).
- **timeout_s** — Max wait per job before it fails (default 600s).
- **log** — Console logging on/off.

## Outputs

- **IMAGE (list)** — One result per input image, **in order**. If a job fails,
  a small black placeholder is returned for that slot so a single bad image
  never aborts the whole batch (the failure is logged with its index).

## Notes

- Credits are consumed per image (one job = one generation).
- Watch progress / failures at <https://kie.ai/logs>.
- Same API key handling as the rest of the pack (`config/kie_key.txt`).
