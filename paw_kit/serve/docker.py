"""Docker deployment exporter for PAW-Kit adapters."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import requires as _pkg_requires
from importlib.metadata import version as _pkg_version
from pathlib import Path
import re
import shutil
from typing import Union

# PAW-DOCKER-01: the exact set of filenames export_docker_scaffold writes into
# output_dir itself. An adapter_name colliding with one of these lets a crafted
# adapter path (e.g. "./Dockerfile") overwrite a just-generated deployment file via
# the copy-adapter step below. requirements.txt (PAW-DOCKER-03) is a generated file
# too, so it must be reserved the same way as the other four.
_RESERVED_OUTPUT_FILENAMES = frozenset(
    {"Dockerfile", ".dockerignore", "docker-compose.yml", "README.md", "requirements.txt"}
)

_VALID_BACKENDS = frozenset({"mock", "real"})

# `real` -- see pyproject.toml's [project.optional-dependencies] -- is the extra that
# actually pulls in the upstream `programasweights` SDK. A container built to serve
# --backend real needs it; a container serving --backend mock does not (the mock
# backend is pure-Python, no model, no SDK).
_REAL_BACKEND_EXTRA = "real"

# Measured up to ~110s worst case for a cold ~600MB base-model download plus load (see
# measurements/README.md); 180s leaves headroom above that rather than matching it
# exactly. The mock backend loads nothing, so its container needs no such headroom.
_REAL_HEALTHCHECK_START_PERIOD = "180s"
_MOCK_HEALTHCHECK_START_PERIOD = "10s"

# PAW-DEPS-01's floors: the lowest version each dependency is declared compatible
# with in pyproject.toml, used only if this environment's installed metadata can't
# be read (e.g. an unusual install layout) — so a generated requirements.txt is
# always pinned to *something* concrete rather than silently falling back to an
# unpinned line.
_REQUIREMENT_FALLBACK_VERSIONS = {
    "paw-kit": "0.1.0",
    "fastapi": "0.110.0",
    "uvicorn": "0.28.0",
    "httpx": "0.27.0",
    "pydantic": "2.6",
    "typer": "0.12",
    "pyyaml": "6.0",
    "interegular": "0.3.3",
}

# X-10: fallback names used only if `_resolved_dependency_names` below can't read
# `paw-kit`'s own installed metadata at all (see its docstring) -- kept in sync with
# `_REQUIREMENT_FALLBACK_VERSIONS` above, which the previous hand-written pin tuple
# (`fastapi`, `uvicorn`, `httpx` only) was not: pyproject.toml gained
# `interegular`/`pydantic`/`pyyaml`/`typer` over time and this tuple was never
# updated to match, so every generated requirements.txt silently left four
# dependencies unpinned.
_FALLBACK_DEPENDENCY_NAMES = ("fastapi", "uvicorn", "httpx", "pydantic", "typer", "pyyaml", "interegular")


def _pinned_requirement(package: str, extra: "str | None" = None) -> str:
    """PAW-DOCKER-03: pin to the exact version installed in *this* environment — the
    one that actually compiled/exported the adapter — rather than an unpinned bare
    package name that lets every future container build resolve a different,
    untested dependency set on its own schedule.

    `extra`, if given, is folded into the requirement name (`paw-kit[real]==...`)
    rather than the pinned version, which is `importlib.metadata`'s version for the
    base distribution regardless of which extras are active in this environment.
    """
    try:
        resolved = _pkg_version(package)
    except PackageNotFoundError:
        resolved = _REQUIREMENT_FALLBACK_VERSIONS[package]
    name = f"{package}[{extra}]" if extra else package
    return f"{name}=={resolved}"


def _resolved_dependency_names(package: str) -> "list[str]":
    """X-10: the exact *unconditioned* runtime dependency names this environment's
    installed `package` distribution declares (`importlib.metadata.requires`), read
    from the resolved environment rather than a hand-maintained tuple that can
    silently drift from pyproject.toml's own `[project.dependencies]` list -- which
    is exactly what happened: the previous tuple named only `fastapi`/`uvicorn`/
    `httpx`, and `interegular`/`pydantic`/`pyyaml`/`typer` were added to
    pyproject.toml without this list ever being updated to match, so every
    generated requirements.txt left four real dependencies to float unpinned.

    Extras-gated entries (a `Requires-Dist` value containing `;`, e.g.
    `"programasweights<0.5.0,>=0.4.4; extra == 'real'"`) are excluded -- this is the
    *bare* install's dependency set; `--backend real`'s extra is folded into the
    `paw-kit[real]==` line separately by the caller, matching the pre-existing
    behaviour, and `programasweights` itself is pinned directly, best-effort, only
    when this environment actually has it installed (see `_requirements_txt_content`).
    """
    try:
        requirement_strings = _pkg_requires(package) or []
    except PackageNotFoundError:
        return []
    names: "list[str]" = []
    for requirement in requirement_strings:
        if ";" in requirement:
            continue
        name = re.split(r"[<>=!~\s\[]", requirement, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return sorted(set(names))


# X-10: both the base Python image and the uv installer image previously used a
# mutable reference -- `python:3.12-slim` is a floating alias that can silently
# repoint to a new build any time upstream pushes one, and `uv:latest` is the same
# problem in its most literal form. Both are pinned by immutable digest (in addition
# to a human-readable tag, for uv) rather than a tag alone: a digest cannot be
# repointed even by the upstream registry itself, unlike any tag. Resolved directly
# against the public registries on 2026-09-12; periodic manual refresh is expected
# as new base-image/uv releases ship, the same way a lockfile is expected to be
# re-resolved periodically rather than never.
_PYTHON_BASE_IMAGE = (
    "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
)
_UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.12.13"
    "@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6"
)

# {backend}: "mock" or "real", threaded into the CMD's --backend flag.
# {warm_clause}: ', "--warm"' for real, '' for mock (there is nothing to warm on mock).
# {healthcheck_comment} / {cmd_comment}: backend-specific prose, see
# _backend_dockerfile_comments below.
# {start_period}: 180s for real (cold model download/load), 10s for mock.
# {python_image} / {uv_image}: see _PYTHON_BASE_IMAGE / _UV_IMAGE above (X-10).
DOCKERFILE_TEMPLATE = """# Generated by PAW-Kit Docker Exporter
FROM {python_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PORT=8000

# PAW-SERVE-01: no default API key is baked into the image (that would leak a
# secret into every image layer/registry). Supply one at *runtime* instead, e.g.:
#   docker run -e PAW_API_KEY=<your-secret> ...
# paw-serve already reads PAW_API_KEY from the process environment; if it is
# absent, the container falls back to generating and logging an ephemeral token
# (see `docker logs`) rather than serving anonymously.

WORKDIR /app

# Install uv package manager
COPY --from={uv_image} /uv /uvx /bin/

# PAW-DOCKER-03: install from a generated, pinned requirements.txt (exact versions,
# pulled from the environment that ran the exporter) instead of an unpinned inline
# package list, so every image build resolves the same dependency set rather than
# whatever happens to be latest on PyPI that day.
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r requirements.txt

# Create unprivileged application user
RUN addgroup --system app && adduser --system --ingroup app app

# Copy adapter artifact into container
COPY --chown=app:app {adapter_filename} /app/{adapter_filename}

USER app

# Expose HTTP service port
EXPOSE 8000

{healthcheck_comment}
HEALTHCHECK --interval=30s --timeout=5s --start-period={start_period} --retries=3 \\
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1

{cmd_comment}
CMD ["paw-serve", "/app/{adapter_filename}", "--host", "0.0.0.0", "--port", "8000", "--backend", "{backend}"{warm_clause}]
"""


def _backend_dockerfile_comments(backend: str) -> "tuple[str, str]":
    """Return (healthcheck_comment, cmd_comment) matching what the generated CMD
    actually serves -- see finding: a `--warm`/180s-start-period pair that described a
    real-model deployment while the CMD had no `--backend real` and shipped no real
    dependencies made the container's readiness check measure nothing."""
    if backend == "real":
        healthcheck_comment = (
            "# Health check (unauthenticated: /ready does not require PAW_API_KEY).\n"
            "# Checks readiness, not just liveness: /health returns ok before any model\n"
            "# is loaded, so a container could report \"healthy\" while its single\n"
            "# inference slot is still busy with the cold model load. This container\n"
            "# serves --backend real (ProgramAsWeightsBackend, the official SDK) --\n"
            "# --start-period covers a first-run cold download of the ~600MB base model\n"
            "# plus load, measured up to ~110s worst case (see measurements/README.md);\n"
            "# 180s leaves headroom above that measurement rather than matching it exactly."
        )
        cmd_comment = (
            "# Launch microservice. PAW_API_KEY, if set in the container's environment,\n"
            "# is picked up automatically by paw-serve; no CLI flag is needed here for it.\n"
            "# --backend real serves the adapter through the official ProgramAsWeights SDK\n"
            "# (an actual model, not a stub). --warm pays the cold-load cost (base model\n"
            "# download + load into memory) before the container reports ready, instead of\n"
            "# on whatever request happens to arrive first."
        )
    else:
        healthcheck_comment = (
            "# Health check (unauthenticated: /ready does not require PAW_API_KEY).\n"
            "# Checks readiness, not just liveness. This container serves --backend mock\n"
            "# (MockPAWBackend) -- a deterministic pure-Python stub, not a real model:\n"
            "# there is no base model to download or load, so --start-period is short\n"
            "# rather than sized for a cold ~600MB download. Re-export with --backend real\n"
            "# (the default) to deploy a container that actually runs a model."
        )
        cmd_comment = (
            "# Launch microservice. PAW_API_KEY, if set in the container's environment,\n"
            "# is picked up automatically by paw-serve. --backend mock serves the\n"
            "# deterministic mock backend, not a real model -- useful for demoing the API\n"
            "# surface or integration-testing this container itself, not for production\n"
            "# inference. --warm is omitted: there is no model to warm."
        )
    return healthcheck_comment, cmd_comment


def _requirements_txt_content(backend: str) -> str:
    """Render requirements.txt content, pinned per `_pinned_requirement`.

    `--backend real` needs the project's `real` extra (pulls in the upstream
    `programasweights` SDK, see pyproject.toml's optional-dependencies); `--backend
    mock` needs only bare `paw-kit` -- the mock backend ships with it.

    X-10: the dependency list itself is read from the resolved environment
    (`_resolved_dependency_names`) instead of a hand-written tuple -- see that
    function's docstring for why the previous tuple (`fastapi`/`uvicorn`/`httpx`
    only) had silently drifted from pyproject.toml's actual dependency list.
    """
    lines = ["# Generated by PAW-Kit Docker Exporter — exact versions, not ranges."]
    extra = _REAL_BACKEND_EXTRA if backend == "real" else None
    lines.append(_pinned_requirement("paw-kit", extra=extra))

    dependency_names = _resolved_dependency_names("paw-kit") or list(_FALLBACK_DEPENDENCY_NAMES)
    lines.extend(_pinned_requirement(pkg) for pkg in dependency_names)

    if backend == "real":
        # Best-effort: `programasweights` is pinned directly only when *this*
        # environment actually has it installed (e.g. the operator exported from an
        # environment that had already compiled/run against --backend real).
        # Otherwise there is nothing resolved to pin, and the `paw-kit[real]==` line
        # above already constrains pip to pyproject.toml's own declared range for it.
        try:
            resolved_paw_version = _pkg_version("programasweights")
        except PackageNotFoundError:
            resolved_paw_version = None
        if resolved_paw_version is not None:
            lines.append(f"programasweights=={resolved_paw_version}")

    return "\n".join(lines) + "\n"


# X-11: the generated README (see README_DOCKER_TEMPLATE below) tells the operator
# to put PAW_API_KEY in a `.env` file next to the compose file, but the previous
# .dockerignore never excluded `.env`/secrets from the build context -- a `docker
# build` run from that directory would have happily copied the secret into the
# image layer history via any `COPY .` (this exporter's own generated Dockerfile
# does not do that today, but nothing stopped an operator's own Dockerfile edits,
# or a future template change, from doing so silently).
DOCKERIGNORE_TEMPLATE = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.git/
.gitignore
.pytest_cache/
.coverage
dist/
build/
.env
.env.*
*.pem
*.key
secrets/
"""

COMPOSE_TEMPLATE = """# Generated by PAW-Kit Docker Exporter
version: '3.8'

services:
  paw-microservice:
    build:
      context: .
      dockerfile: Dockerfile
    # PAW-DOCKER-02: bind the published port to loopback only by default. The
    # container's own internal bind stays 0.0.0.0 (required for Docker's networking
    # to reach it at all) — this only controls which *host* interfaces the mapped
    # port is reachable from. Change to "8000:8000" only if external/LAN access is
    # actually intended, ideally behind a reverse proxy that terminates TLS.
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      # PAW-SERVE-01: forwards the host's PAW_API_KEY into the container so the
      # generated container inherits auth-by-default instead of a freshly minted
      # ephemeral token on every restart. Set it before running `docker compose up`,
      # e.g. `export PAW_API_KEY=$(openssl rand -base64 32)`, or add it to a `.env`
      # file next to this compose file. Leaving it unset falls back to an ephemeral
      # token logged to `docker compose logs`, not anonymous access.
      - PAW_API_KEY=${{PAW_API_KEY:-}}
    restart: unless-stopped
    healthcheck:
{compose_healthcheck_comment}
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"]
      interval: 30s
      timeout: 5s
      start_period: {start_period}
      retries: 3
"""


def _compose_healthcheck_comment(backend: str) -> str:
    if backend == "real":
        return (
            "      # See the matching Dockerfile HEALTHCHECK comment: /ready (readiness),\n"
            "      # not /health (liveness), and a start_period that covers a cold\n"
            "      # base-model download for --backend real."
        )
    return (
        "      # See the matching Dockerfile HEALTHCHECK comment: /ready (readiness),\n"
        "      # not /health (liveness). --backend mock has no model to load, so\n"
        "      # start_period is short."
    )


README_DOCKER_TEMPLATE = """# PAW Adapter Container Deployment

This directory contains containerization assets for `{adapter_filename}`.

## Authentication (PAW-SERVE-01)

The server authenticates by default. Set `PAW_API_KEY` before starting the container,
or it will generate and log a new ephemeral bearer token on every start/restart
(visible via `docker logs` / `docker compose logs`, and **rotating on every restart**
— clients holding the old token will start getting 401s). To use a stable key:

```bash
export PAW_API_KEY=$(openssl rand -base64 32)
```

### Build Image
```bash
docker build -t paw-adapter:{tag} .
```

### Run Container
```bash
docker run -p 127.0.0.1:8000:8000 -e PAW_API_KEY="$PAW_API_KEY" paw-adapter:{tag}
```
(PAW-DOCKER-02: bound to loopback only, matching `docker-compose.yml`'s default —
swap `127.0.0.1:8000:8000` for `8000:8000` only if external/LAN access is actually
intended, ideally behind a reverse proxy that terminates TLS.)

### Or Run via Docker Compose
```bash
docker compose up --build -d
```
(`docker-compose.yml` forwards `PAW_API_KEY` from your shell automatically — see its
`environment:` block.)

### Test Serving Endpoints

**OpenAI Client:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $PAW_API_KEY" \\
  -d '{{"messages": [{{"role": "user", "content": "Sample query"}}]}}'
```

**Anthropic Client:**
```bash
curl -X POST http://localhost:8000/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $PAW_API_KEY" \\
  -d '{{"messages": [{{"role": "user", "content": "Sample query"}}]}}'
```

**Health Check** (unauthenticated by design): `/health` is liveness (process is up);
`/ready` is readiness (the model has actually loaded and served one successful call --
this container's HEALTHCHECK polls `/ready`, and `--warm` in its CMD pays that cold-load
cost before the container reports ready, not on whatever request happens to arrive first):
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```
"""


def export_docker_scaffold(
    adapter_path: Union[str, Path],
    output_dir: Union[str, Path] = "./docker",
    copy_adapter: bool = True,
    backend: str = "real",
) -> Path:
    """Generate production Dockerfile and compose assets for deploying a .paw adapter.

    `backend` chooses which engine the generated container actually serves --
    "real" (default) for `ProgramAsWeightsBackend` (the official SDK, an actual
    model) or "mock" for `MockPAWBackend` (a deterministic stub, useful for demoing
    the API surface or testing the container itself, not for production inference).
    It is threaded into the generated CMD's `--backend` flag, the generated
    requirements.txt (the real extra is only pulled in for "real"), and the
    Dockerfile/compose comments and HEALTHCHECK `--start-period`/`start_period`, so
    none of those describe a backend the container does not actually run. Defaults to
    "real": a container that only ever serves the mock is a demo, not a deployment.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}")

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter file '{adapter_path}' does not exist.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adapter_name = adapter.name

    if not re.match(r"^[\w\-.]+$", adapter_name):
        raise ValueError(f"Adapter filename contains unsafe characters: {adapter_name!r}")

    # PAW-DOCKER-01: `adapter.name` is a Path.name and can never contain a path
    # separator, so a traversal sequence cannot reach this point — the real,
    # verified vector is an adapter filename that collides with one of the four
    # files this function writes below, letting the "copy adapter" step overwrite
    # a just-generated deployment file (e.g. `paw export docker ./Dockerfile
    # --out-dir ./deploy` would otherwise overwrite the generated Dockerfile).
    if adapter_name in _RESERVED_OUTPUT_FILENAMES:
        raise ValueError(
            f"Adapter filename {adapter_name!r} collides with a file "
            f"export_docker_scaffold generates itself {sorted(_RESERVED_OUTPUT_FILENAMES)}; "
            "rename the adapter file."
        )
    if not adapter_name.endswith(".paw"):
        raise ValueError(f"Adapter filename must have a '.paw' extension: {adapter_name!r}")

    tag_name = adapter.stem.lower().replace("_", "-")

    start_period = _REAL_HEALTHCHECK_START_PERIOD if backend == "real" else _MOCK_HEALTHCHECK_START_PERIOD
    warm_clause = ', "--warm"' if backend == "real" else ""
    healthcheck_comment, cmd_comment = _backend_dockerfile_comments(backend)

    # 1. Write Dockerfile
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        adapter_filename=adapter_name,
        backend=backend,
        warm_clause=warm_clause,
        healthcheck_comment=healthcheck_comment,
        cmd_comment=cmd_comment,
        start_period=start_period,
        python_image=_PYTHON_BASE_IMAGE,
        uv_image=_UV_IMAGE,
    )
    (out / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

    # 2. Write .dockerignore
    (out / ".dockerignore").write_text(DOCKERIGNORE_TEMPLATE, encoding="utf-8")

    # 3. Write docker-compose.yml
    compose_content = COMPOSE_TEMPLATE.format(
        compose_healthcheck_comment=_compose_healthcheck_comment(backend),
        start_period=start_period,
    )
    (out / "docker-compose.yml").write_text(compose_content, encoding="utf-8")

    # 4. Write README.md
    readme_content = README_DOCKER_TEMPLATE.format(
        adapter_filename=adapter_name,
        tag=tag_name,
    )
    (out / "README.md").write_text(readme_content, encoding="utf-8")

    # 5. Write requirements.txt (PAW-DOCKER-03)
    (out / "requirements.txt").write_text(_requirements_txt_content(backend), encoding="utf-8")

    # 6. Copy adapter file
    if copy_adapter:
        dest_adapter = out / adapter_name
        if adapter.resolve() != dest_adapter.resolve():
            shutil.copy2(adapter, dest_adapter)

    return out
