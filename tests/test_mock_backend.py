"""Unit tests for AbstractPAWBackend protocol and MockPAWBackend implementation."""

import json
import time
from pathlib import Path

import pytest
from paw_kit import AbstractPAWBackend, MockPAWBackend


def test_protocol_conformance() -> None:
    """Verify MockPAWBackend implements the AbstractPAWBackend interface."""
    backend = MockPAWBackend()
    assert isinstance(backend, AbstractPAWBackend)


def test_availability() -> None:
    """Verify mock backend is immediately available without hardware dependencies."""
    backend = MockPAWBackend()
    assert backend.is_available() is True


def test_mock_compilation_creates_artifact(tmp_path: Path) -> None:
    """Verify compilation creates artifact on disk and in memory in under 50ms."""
    backend = MockPAWBackend()
    adapter_path = str(tmp_path / "models" / "triage.paw")
    spec = "Classify customer ticket priority (low, med, high)."
    examples = [
        {"input": "Server is down", "output": '{"priority": "high"}'},
        {"input": "How do I change font?", "output": '{"priority": "low"}'},
    ]

    start_time = time.perf_counter()
    result_path = backend.compile(spec=spec, examples=examples, output_path=adapter_path)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert result_path == adapter_path
    assert Path(adapter_path).exists()
    assert elapsed_ms < 50.0  # Fast deterministic compile guarantee

    # Verify written JSON artifact
    with open(adapter_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["spec"] == spec
    assert data["examples_count"] == 2
    assert data["backend"] == "mock"


def test_mock_inference_examples_and_fallback(tmp_path: Path) -> None:
    """Verify mock inference routes to examples, registered rules, or fallback."""
    backend = MockPAWBackend()
    adapter_path = str(tmp_path / "date_normalizer.paw")
    examples = [
        {"input": "yesterday", "output": "2026-09-04"},
        {"input": "tomorrow", "output": "2026-09-06"},
    ]
    backend.compile(spec="Normalize dates", examples=examples, output_path=adapter_path)

    # 1. Matching example input
    assert backend.infer(adapter_path, "yesterday") == "2026-09-04"
    assert backend.infer(adapter_path, "tomorrow") == "2026-09-06"

    # 2. Registered rule takes precedence
    backend.register_rule(adapter_path, "February 30th", "INVALID")
    assert backend.infer(adapter_path, "February 30th") == "INVALID"

    # 3. Unmatched input falls back to deterministic mock response
    assert backend.infer(adapter_path, "some unknown text") == "[mock:some unknown text]"

    # 4. Configurable default response
    backend.set_default_response(adapter_path, "DEFAULT_VAL")
    assert backend.infer(adapter_path, "another unknown text") == "DEFAULT_VAL"


def test_mock_backend_disk_reload(tmp_path: Path) -> None:
    """Verify mock backend can reload an adapter saved to disk across backend instances."""
    adapter_path = str(tmp_path / "reloaded.paw")
    backend1 = MockPAWBackend()
    backend1.compile(
        spec="Test spec",
        examples=[{"input": "ping", "output": "pong"}],
        output_path=adapter_path,
    )

    # Separate backend instance without in-memory state
    backend2 = MockPAWBackend()
    assert backend2.get_adapter(adapter_path) is None

    # Inference should reload metadata from disk
    result = backend2.infer(adapter_path, "ping")
    assert result == "pong"
    assert backend2.get_adapter(adapter_path) is not None


def test_mock_backend_corrupted_file_and_reset(tmp_path: Path) -> None:
    """Verify corrupted artifact handling, uncompiled rule registration, and reset."""
    backend = MockPAWBackend()
    corrupt_file = tmp_path / "corrupt.paw"
    corrupt_file.write_text("invalid-json{", encoding="utf-8")

    # Should gracefully handle JSONDecodeError and fall back
    assert backend.infer(str(corrupt_file), "test") == "[mock:test]"

    # Test registering rule and default response on uncompiled path
    backend.register_rule("custom_path", "hello", "world")
    assert backend.infer("custom_path", "hello") == "world"

    backend.set_default_response("custom_path_2", "default_val")
    assert backend.infer("custom_path_2", "anything") == "default_val"

    # Test reset clears all state
    backend.reset()
    assert backend.get_adapter("custom_path") is None


def test_mock_backend_concurrent_compile_and_infer_PAW_BACKEND_03(tmp_path: Path) -> None:
    """Verify genuine multi-threaded compile()/infer() calls against many distinct
    adapter paths never raise (e.g. a dict-mutated-during-iteration RuntimeError) and
    every result is exactly what that adapter's own examples specify -- the audit's
    own theorized race, exercised with real concurrent threads, not simulated
    single-threaded."""
    from concurrent.futures import ThreadPoolExecutor

    backend = MockPAWBackend()
    num_adapters = 20
    calls_per_adapter = 15
    errors: list = []

    def worker(idx: int) -> None:
        try:
            adapter_path = str(tmp_path / f"adapter_{idx}.paw")
            backend.compile(
                spec=f"spec-{idx}",
                examples=[{"input": "ping", "output": f"pong-{idx}"}],
                output_path=adapter_path,
            )
            for _ in range(calls_per_adapter):
                result = backend.infer(adapter_path, "ping")
                if result != f"pong-{idx}":
                    errors.append(f"adapter {idx}: expected pong-{idx}, got {result}")
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(f"adapter {idx}: {exc!r}")

    with ThreadPoolExecutor(max_workers=num_adapters) as executor:
        futures = [executor.submit(worker, i) for i in range(num_adapters)]
        for f in futures:
            f.result(timeout=30)

    assert errors == []


def test_mock_backend_adapter_cache_bounded_lru_PAW_BACKEND_03(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the in-memory adapter cache evicts least-recently-used entries past its
    size cap, rather than retaining every distinct adapter_path ever compiled."""
    import paw_kit.backend.mock as mock_module

    monkeypatch.setattr(mock_module, "_MAX_CACHED_ADAPTERS", 3)
    backend = MockPAWBackend()

    for i in range(5):
        backend.compile(spec="s", examples=[], output_path=str(tmp_path / f"a{i}.paw"))

    assert len(backend._adapters) == 3
    # The three most recently compiled survive; the earliest two were evicted.
    assert backend.get_adapter(str(tmp_path / "a0.paw")) is None
    assert backend.get_adapter(str(tmp_path / "a4.paw")) is not None


def test_mock_backend_rejects_oversized_adapter_file_PAW_BACKEND_04(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a reloaded adapter file over the size cap is treated as absent rather
    than passed to json.load, which reads the whole file into memory before it can
    even attempt to parse it."""
    import paw_kit.backend.mock as mock_module

    monkeypatch.setattr(mock_module, "_MAX_ADAPTER_FILE_BYTES", 100)
    backend = MockPAWBackend()
    big_file = tmp_path / "big.paw"
    big_file.write_text(json.dumps({"examples": [{"input": "x", "output": "y"}], "rules": {}}) + " " * 200, encoding="utf-8")

    assert backend.infer(str(big_file), "x") == f"[mock:x]"
    assert backend.get_adapter(str(big_file)) is None


def test_mock_backend_rejects_malformed_adapter_shape_PAW_BACKEND_04(tmp_path: Path) -> None:
    """Verify a structurally-invalid adapter file (examples/rules of the wrong type)
    is treated as absent rather than trusted -- infer()'s matching logic assumes
    `examples` is a list of dicts and `rules` is a string-to-string mapping."""
    backend = MockPAWBackend()

    bad_examples = tmp_path / "bad_examples.paw"
    bad_examples.write_text(json.dumps({"examples": "not-a-list", "rules": {}}), encoding="utf-8")
    assert backend.infer(str(bad_examples), "x") == "[mock:x]"
    assert backend.get_adapter(str(bad_examples)) is None

    bad_rules = tmp_path / "bad_rules.paw"
    bad_rules.write_text(json.dumps({"examples": [], "rules": ["not", "a", "dict"]}), encoding="utf-8")
    assert backend.infer(str(bad_rules), "x") == "[mock:x]"
    assert backend.get_adapter(str(bad_rules)) is None

    bad_rule_values = tmp_path / "bad_rule_values.paw"
    bad_rule_values.write_text(json.dumps({"examples": [], "rules": {"k": 123}}), encoding="utf-8")
    assert backend.infer(str(bad_rule_values), "x") == "[mock:x]"
    assert backend.get_adapter(str(bad_rule_values)) is None
