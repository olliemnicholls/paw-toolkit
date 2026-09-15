# Install

Not on PyPI yet. From source:

```bash
git clone https://github.com/olliemnicholls/paw-toolkit
cd paw-toolkit
uv sync                # or: pip install -e .
uv run paw-kit demo    # no GPU, no network, no API key
```

## The real backend

For a real backend, install the `real` extra and get an API key from
[programasweights.com/settings](https://programasweights.com/settings):

```bash
uv sync --extra real   # or: pip install -e '.[real]'
export PAW_API_KEY=paw_sk_...
```

`paw-kit[real]` pulls the official upstream SDK, which is what `ProgramAsWeightsBackend`
runs on. It resolves from PyPI directly; the `--extra-index-url` in upstream's own README
is not needed. No API key is required to *run* an already-compiled program, only to
compile a new one.

`paw-kit[real]` also pulls `llguidance`, the engine `ProgramAsWeightsBackend` uses for
grammar-constrained decoding (on by default; pass `constrained_decoding=False` to opt
out). Its prebuilt wheels cover Python 3.11-3.13 on Linux (glibc >= 2.31), macOS and
Windows; on anything else, installing it needs a Rust toolchain to build from source.
On a host where it is not importable, `ProgramAsWeightsBackend` warns once and falls
back to unconstrained decoding with post-hoc validation after generation -- it does not
fail to install or fail at inference time.

Three things to know before the first real call:

- The `llama-cpp-python` wheel this pulls from PyPI is **CPU-only**, which is roughly 90x
  slower per call than a CUDA build. See [GPU support](#gpu-support) below.
- The first call to any program downloads the ~600 MB base model into the SDK's cache.
- Grammar-constrained decoding guarantees output *shape* (it parses as your schema), not
  field-value correctness -- post-generation validation and fallback still run regardless.

Run `paw-kit doctor` before the first real call. It checks the SDK import, whether
`llguidance` is importable, the API key, the CUDA state of `llama-cpp-python`, the
compile service's health, and the local model cache, each with a one-line remedy. With `--offline` it skips the network checks; with
`--adapter a.paw` it also says whether that adapter can run with no network at all.
Without the SDK installed the SDK-dependent checks are reported as skipped warnings
rather than failures.

## The judge extra

`paw-test judge` sends cases to an LLM judge and needs the `anthropic` client:

```bash
uv sync --extra judge   # or: pip install -e '.[judge]'
export ANTHROPIC_API_KEY=...
```

## GPU support

Check whether the installed `llama-cpp-python` can use your GPU:

```bash
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

If that prints `False`, the wheel has no CUDA support compiled in. Two fixes, in order of
preference:

1. **Prebuilt CUDA wheel** (no compiler needed):
   ```bash
   pip install "llama-cpp-python==<version>" \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   If you then get `OSError: libcudart.so.12: cannot open shared object file`, the
   machine has a GPU driver but no CUDA *toolkit*. Add the runtime libraries standalone
   and point the loader at them:
   ```bash
   pip install "nvidia-cuda-runtime-cu12==12.1.*" "nvidia-cublas-cu12==12.1.*"
   export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))')/lib:$(python -c 'import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))')/lib:$LD_LIBRARY_PATH"
   ```
2. **Build from source** with `CMAKE_ARGS="-DGGML_CUDA=on"` if there is no prebuilt wheel
   for your CUDA version. On Ubuntu 24.04 with CUDA 12.1 the default `gcc` (13.x) is too
   new for `nvcc`: either install `g++-12`/`gcc-12` and add
   `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12` to `CMAKE_ARGS`, or use CUDA 12.4 or
   later, which supports GCC 13.

If PyTorch is installed in the same environment, prefer the source build: torch can pull
in its own newer `nvidia-*` CUDA packages, and a prebuilt wheel that finds those at load
time instead of the system CUDA can crash on GPU init.

## A note on `uv sync`

`uv sync` performs an exact sync by default, which evicts anything installed outside
the lockfile, including a hand-built CUDA `llama-cpp-python`. If you have installed
custom packages or builds by hand, sync with `uv sync --inexact`.

Under exact-sync semantics, `uv sync --extra real` and `uv sync --extra judge` are
mutually evicting: running `uv sync --extra judge` uninstalls the `real` extra
(and vice versa). To install both, pass them together in one command:

```bash
uv sync --extra real --extra judge   # or pass --inexact
```

