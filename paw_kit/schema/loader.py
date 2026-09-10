"""High-level model loader with schema validation and fail-open routing."""

import warnings
from typing import Any, Callable, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.schema.exceptions import PAWSchemaError
from paw_kit.schema.grammar import pydantic_to_regex

T = TypeVar("T", bound=BaseModel)

_DEFAULT_BACKEND: Optional[AbstractPAWBackend] = None

_NO_BACKEND_WARNING = (
    "paw_kit: no backend= was provided, so this is running on the built-in "
    "MockPAWBackend -- a deterministic rule/example-based stub, not a real model. "
    "Any adapter 'compiled' against it will keep returning that stub's synthetic "
    "output forever, silently, including after @compile_on_hit hot-swaps to it. "
    "Pass backend=ProgramAsWeightsBackend(...) (`pip install programasweights`, get an "
    "API key at https://programasweights.com/settings) or "
    "your own AbstractPAWBackend for real production inference."
)


def get_default_backend() -> AbstractPAWBackend:
    """Retrieve or initialize the active default backend.

    Warns (once, the first time this lazily constructs the singleton) since
    callers of `paw.load`/`@compile_on_hit` who omit `backend=` land here
    silently otherwise -- this is the mock backend, not a placeholder that
    becomes real hardware in production. See `_NO_BACKEND_WARNING`.
    """
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        warnings.warn(_NO_BACKEND_WARNING, UserWarning, stacklevel=3)
        _DEFAULT_BACKEND = MockPAWBackend()
    return _DEFAULT_BACKEND


def set_default_backend(backend: AbstractPAWBackend) -> None:
    """Override the default backend."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def load(
    adapter_path: str,
    response_model: Type[T],
    backend: Optional[AbstractPAWBackend] = None,
    fallback_provider: Optional[Callable[[str], Any]] = None,
) -> Callable[[str], T]:
    """Load a compiled PAW adapter and bind it to a strict Pydantic response schema.

    Compiles the schema to a regex and passes it to the backend as `grammar_constraint`,
    but no shipped backend applies it at decoding time (see the README's "what is real"
    table). What this function actually enforces is post-generation Pydantic validation:
    output that fails to parse as `response_model` is routed to `fallback_provider`, or
    raised as `PAWSchemaError` if none is configured.

    Args:
        adapter_path: Path to the .paw adapter artifact.
        response_model: Target Pydantic BaseModel class for validation.
        backend: PAW backend to execute inference. Uses default backend if None.
        fallback_provider: Optional fallback callable invoked if local inference fails.

    Returns:
        A typed callable accepting input string and returning a validated Pydantic model instance.
    """
    active_backend = backend or get_default_backend()
    try:
        grammar_regex = pydantic_to_regex(response_model, anchors=False)
    except PAWSchemaError:
        raise
    except Exception as exc:
        raise PAWSchemaError(
            f"Failed to compile grammar regex for {response_model.__name__}: {exc}"
        ) from exc

    def _execute(input_text: str) -> T:
        try:
            # 1. Run local inference with grammar constraint
            raw_output = active_backend.infer(
                adapter_path=adapter_path,
                input_text=input_text,
                grammar_constraint=grammar_regex,
            )

            # 2. Parse and validate output
            if isinstance(raw_output, response_model):
                return raw_output
            if isinstance(raw_output, dict):
                return response_model.model_validate(raw_output)
            if isinstance(raw_output, str):
                return response_model.model_validate_json(raw_output)
            raise ValueError(f"Unexpected backend output type: {type(raw_output)}")

        except Exception as exc:
            # 3. Fail-Open Safety: Fall back to teacher if configured
            if fallback_provider is not None:
                try:
                    fallback_raw = fallback_provider(input_text)
                    if isinstance(fallback_raw, response_model):
                        return fallback_raw
                    if isinstance(fallback_raw, dict):
                        return response_model.model_validate(fallback_raw)
                    if isinstance(fallback_raw, str):
                        return response_model.model_validate_json(fallback_raw)
                    raise ValueError(f"Unexpected fallback return type: {type(fallback_raw)}")
                except Exception as fallback_exc:
                    raise PAWSchemaError(
                        f"Both local execution and fallback failed for model {response_model.__name__}."
                    ) from fallback_exc

            # If no fallback configured, raise explicit schema error
            raise PAWSchemaError(
                f"Local execution failed validation against {response_model.__name__}: {exc}"
            ) from exc

    return _execute
