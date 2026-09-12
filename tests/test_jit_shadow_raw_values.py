"""J-4 and J-10: `agreement_fn` must receive the teacher's and the adapter's real
(undecoded) values on both the shadow and audit paths, and a teacher-side coercion
failure must score `teacher_error`, excluded from the promotion window, not `error`
(which counts against the adapter). See bug-hunt-2026-09-11.md section 7 and this
track's Dependency check for the full rationale.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel
import pytest

from paw_kit import MockPAWBackend, compile_on_hit
import paw_kit.jit.decorator as decorator_module
from paw_kit.jit.agreement import field_tolerance_agreement
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER


class Triage(BaseModel):
    label: str
    urgency_score: int


class DictBackend(MockPAWBackend):
    """Returns a raw Python dict, not a JSON string -- exercises the no-response_model
    path where a custom backend/teacher naturally returns structured Python objects."""

    def __init__(self, urgency: int) -> None:
        super().__init__()
        self.urgency = urgency

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> Dict[str, Any]:
        return {"label": "x", "urgency_score": self.urgency}


def _make(
    tmp_path: Path,
    name: str,
    backend: MockPAWBackend,
    teacher_fn: Optional[Callable[[str], Any]] = None,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, int]]:
    calls = {"n": 0}
    spec = f"J-4/J-10 raw values ({name})"
    cache_dir = str(tmp_path / f"cache_{name}")
    fn = teacher_fn or (lambda text: f"teacher:{text}")

    def teacher(text: str) -> Any:
        calls["n"] += 1
        return fn(text)

    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
        shadow_queue_size=64,
    )
    params.update(kwargs)
    return compile_on_hit(**params)(teacher), calls


def _drain(wrapper: Any, timeout: float = 10.0) -> None:
    assert _GLOBAL_SHADOW_RUNNER.drain(wrapper.task_id, timeout, db_path=str(wrapper.db.db_path))


def _pairs(wrapper: Any) -> List[Dict[str, Any]]:
    with wrapper.db._lock:
        rows = wrapper.db._conn.execute(
            "SELECT * FROM shadow_pairs WHERE task_id = ? ORDER BY id ASC;", (wrapper.task_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- J-4: raw values reach agreement_fn, shadow phase -------------------------------


def test_j4_field_tolerance_agreement_promotes_on_shadow_phase_without_response_model(
    tmp_path: Path,
) -> None:
    """Teacher and adapter both return raw dicts (no response_model). Their
    urgency_score differs by exactly the tolerance -- must AGREE. Before J-4, both
    sides were always stringified before agreement_fn saw them, so
    field_tolerance_agreement's dict-shaped input never arrived and this fell
    through to string equality (a disagreement, since the JSON strings differ)."""
    backend = DictBackend(urgency=5)
    svc, _ = _make(
        tmp_path, "shadow-tolerance", backend,
        teacher_fn=lambda text: {"label": "x", "urgency_score": 4},
        shadow_window=1, shadow_threshold=1.0,
        agreement_fn=field_tolerance_agreement({"urgency_score": 1}),
    )
    svc("seed0")
    svc("seed1")
    svc("compare-me")
    _drain(svc)

    pairs = _pairs(svc)
    assert pairs, "expected a shadow comparison"
    assert pairs[-1]["verdict"] == "agree", (
        "field_tolerance_agreement must have received dicts, not stringified JSON, "
        "on the shadow path without a response_model"
    )
    assert svc.db.get_status(svc.task_id) == "ready"


# --- J-4: raw values reach agreement_fn, audit phase ---------------------------------


def test_j4_field_tolerance_agreement_receives_dicts_on_audit_path_without_response_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same property, but for the AUDIT path specifically -- Phase 0's own scope
    correction: the audit path's *adapter* value is stringified on the caller
    thread before it ever reaches shadow.py, so a teacher-only fix leaves this
    path a no-op even after fixing the shadow path alone."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = DictBackend(urgency=5)
    svc, _ = _make(
        tmp_path, "audit-tolerance", backend,
        teacher_fn=lambda text: {"label": "x", "urgency_score": 4},
        shadow_window=1, shadow_threshold=1.0,
        audit_window=1, audit_rate=0.5, demote_threshold=0.0,
        agreement_fn=field_tolerance_agreement({"urgency_score": 1}),
    )
    svc("seed0")
    svc("seed1")
    svc("promote-me")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready", "setup failed: task did not promote"

    svc("audit-me")
    _drain(svc)

    audit_pairs = [p for p in _pairs(svc) if p["phase"] == "audit"]
    assert audit_pairs, "expected an audit comparison"
    assert audit_pairs[-1]["verdict"] == "agree", (
        "field_tolerance_agreement must have received dicts, not stringified JSON, "
        "on the audit path without a response_model"
    )
    assert svc.db.get_status(svc.task_id) == "ready", "must not have demoted on a real agreement"


def test_j4_agreement_fn_receives_raw_python_types_not_strings(tmp_path: Path) -> None:
    """Directly pins the type: a custom agreement_fn records what it was actually
    handed. Both sides must be dicts, never str, when the teacher/adapter both
    return dicts and no response_model is configured."""
    seen: List[Tuple[type, type]] = []

    def recording_agreement(teacher: Any, adapter: Any) -> bool:
        seen.append((type(teacher), type(adapter)))
        return True

    backend = DictBackend(urgency=5)
    svc, _ = _make(
        tmp_path, "type-check", backend,
        teacher_fn=lambda text: {"label": "x", "urgency_score": 5},
        shadow_window=1, shadow_threshold=1.0,
        agreement_fn=recording_agreement,
    )
    svc("seed0")
    svc("seed1")
    svc("compare-me")
    _drain(svc)

    assert seen == [(dict, dict)], f"agreement_fn saw {seen}, expected raw dicts on both sides"


# --- J-10: teacher-side vs adapter-side coercion failures ---------------------------


def test_j10_teacher_side_coercion_failure_scores_teacher_error_not_error(
    tmp_path: Path,
) -> None:
    """The report's own reproduction, on the shadow path: a valid adapter, a
    teacher returning a plain string that fails to parse against response_model.
    Must be `teacher_error` (excluded from the promotion window), not `error`
    (which counts against the adapter)."""
    backend = MockPAWBackend()
    svc, _ = _make(
        tmp_path, "teacher-coerce", backend,
        teacher_fn=lambda text: "label=x",  # not valid JSON for Triage
        response_model=Triage, shadow_window=1, shadow_threshold=1.0,
    )
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    # Give the adapter a matching rule so its own output validates cleanly against
    # Triage -- isolating the teacher-side coercion failure specifically (the
    # adapter's own load() would otherwise fail first, on the `[mock:...]`
    # sentinel, and that failure is caught before the comparison block even runs).
    adapter_path = svc.db.get_task_routing(svc.task_id)[1]
    backend.register_rule(adapter_path or "", "x1", '{"label": "x", "urgency_score": 1}')

    svc("x1")
    _drain(svc)

    pairs = _pairs(svc)
    assert pairs, "expected a shadow comparison"
    assert pairs[-1]["verdict"] == "teacher_error", (
        f"a teacher-side coercion failure scored {pairs[-1]['verdict']!r}, expected "
        "'teacher_error' -- it must not be charged to the adapter"
    )
    assert pairs[-1]["error_type"] in ("ValueError", "ValidationError")
    assert svc.db.get_status(svc.task_id) == "shadow", "must not have promoted on an excluded row"


def test_j10_window_still_fills_despite_teacher_side_failures(tmp_path: Path) -> None:
    """A teacher-side coercion failure must be excluded from the window's
    denominator, exactly as it already is for a teacher *exception* (see
    test_teacher_error_on_audit_is_excluded_from_numerator_and_denominator in
    tests/test_jit_shadow.py for the pre-existing exception-based case this
    mirrors for a coercion failure instead). window=2: one excluded ("bad") row
    plus two genuine agreements must promote in exactly one window's worth of
    genuine comparisons -- if the excluded row wrongly consumed a window slot
    (scored `error` instead), the two genuine agreements would split across two
    incomplete windows instead and the task would still be in `shadow`."""
    backend = MockPAWBackend()
    state = {"valid": False}

    def flaky_teacher(text: str) -> str:
        return '{"label": "x", "urgency_score": 1}' if state["valid"] else "label=x"

    svc, _ = _make(
        tmp_path, "teacher-coerce-fills", backend, teacher_fn=flaky_teacher,
        response_model=Triage, shadow_window=2, shadow_threshold=1.0,
    )
    svc("seed0")
    svc("seed1")
    adapter_path = svc.db.get_task_routing(svc.task_id)[1]
    for text in ("bad", "good1", "good2"):
        backend.register_rule(adapter_path or "", text, '{"label": "x", "urgency_score": 1}')

    svc("bad")  # a teacher-side coercion failure: must be excluded, not spend a slot
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "shadow"

    state["valid"] = True
    svc("good1")
    svc("good2")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready", (
        "two genuine agreements must fill a window of 2 regardless of the earlier "
        "excluded row -- if it wrongly counted, the window would have already "
        "split across two incomplete draws instead"
    )
