# Install

Not on PyPI yet. From source:

```bash
git clone https://github.com/olliemnicholls/paw-toolkit
cd paw-toolkit
uv sync --dev          # or: pip install -e .
uv run pytest -q       # no GPU, no network, no API key
```

## The real backend

For a real backend, install the `real` extra and get an API key from
[programasweights.com/settings](https://programasweights.com/settings):

```bash
uv sync --extra real   # or: pip install 'paw-kit[real]'
export PAW_API_KEY=paw_sk_...
```

`paw-kit[real]` pulls the official upstream SDK, which is what `ProgramAsWeightsBackend`
runs on. It resolves from PyPI directly; the `--extra-index-url` in upstream's own README
is not needed (verified 2026-09-09 against `programasweights==0.4.4`). No API key is
required to *run* an already-compiled program, only to compile a new one.

Two things to know before the first real call:

- The `llama-cpp-python` wheel this pulls from PyPI is **CPU-only**. On this repo's date
  normaliser that meant ~5.9 s per call against ~65 ms once a CUDA build was in place. The
  build notes are at the end of
  [`measurements/README.md`](../measurements/README.md#if-inference-is-unexpectedly-slow-seconds-not-milliseconds).
- The first call to any program downloads the ~600 MB base model into the SDK's cache.

Run `paw-kit doctor` before the first real call. It checks the SDK import, the API key,
the CUDA state of `llama-cpp-python`, the compile service's health (including whether it
has GPU workers behind a healthy `200`), and the local model cache, each with a one-line
remedy. With `--offline` it skips the network checks; with `--adapter a.paw` it also says
whether that adapter can run with no network at all. Without the SDK installed the
SDK-dependent checks are reported as skipped warnings rather than failures.

## The measurement extra

`paw-kit[measure]` is a separate, optional extra pulling PyTorch/transformers. It exists
only to reproduce `scripts/measure_schema_real_model.py`, which backs the
constrained-decoding numbers in [`measurements/`](../measurements/README.md). It is **not**
a backend and buys you no inference. (It was called `[torch]`, and before that it was what
`[real]` installed; both were misleading, so it is now named for what it actually does.)

## A note on `uv run`

`uv run` without `--no-sync` re-syncs the environment to the lockfile, which evicts
anything installed outside it, including the upstream SDK and a hand-built CUDA
`llama-cpp-python`. If you have installed either by hand, run tools with
`uv run --no-sync`.
