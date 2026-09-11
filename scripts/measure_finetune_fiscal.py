"""Can the finetune compiler (`paw-ft-bs48`) learn an arbitrary rule the fast compiler
(`paw-4b-qwen3-0.6b`) cannot?

`measurements/README.md` has two prior finetune-vs-fast comparisons. On phone extraction
(easy, format-following) 132 of 134 outputs were byte-identical between the two compilers.
On ticket triage (hard, judgement) the finetune compiler scored 60.0% against the fast
compiler's 53.3% and a 91.7% teacher ceiling -- better, but nowhere near enough, and that
section closes by saying "a task the finetune compiler can do and the fast one cannot is
the next thing to look for."

This script looks for it. The hypothesis: the fast compiler is a single forward pass from
spec text to adapter weights, so it can only emit adapters of a kind its training covered;
the finetune compiler generates examples from the spec with a teacher and *trains* on
them, so it should win on an explicit, arbitrary procedure that is stated in the spec and
that the base model does not already know.

The task is fiscal-week labelling. A date in, a label `FY<year>-W<nn>` out, where the
fiscal year starts on the first Monday of February. It was picked because:

  * the rule is arbitrary -- no pretrained model knows this particular fiscal calendar,
    and the nearest thing it does know (ISO 8601 week numbers, January-anchored) gives a
    systematically *different* answer, so guessing from priors is visibly wrong;
  * the ground truth is computable by code, so there is no judge, no teacher ceiling and
    no label noise -- every number here is exact and the evaluation is free;
  * it needs real multi-step arithmetic (find a weekday-anchored epoch, subtract, divide,
    handle the wrap into the previous fiscal year), not pattern-matching.

Four arms. Three compiles, one API-only reference:

    A  fast compiler `paw-4b-qwen3-0.6b`, max_spec_examples=0
    B  fast compiler, the 8 folding examples folded into the spec text
    C  finetune compiler `paw-ft-bs48`, the same 8 examples, via compile_async polling
    D  `claude-haiku-4-5-20251001` zero-shot on the same spec, temperature 0, no compile

D is a frontier-model reference for how hard the rule is. It is explicitly *not* a
ceiling: ground truth here is exact, so 100% is the ceiling and D can be beaten.

Evaluation set: 300 dates drawn deterministically (fixed seed) across 2024-01-01 to
2027-12-31, oversampling the ten days either side of each fiscal-year boundary, formatted
round-robin across four unambiguous formats (ISO, long, day-first, weekday-prefixed).
Folding pool: 8 examples, disjoint from the evaluation set, covering all four formats and
three boundary cases. Both are committed as a JSON fixture.

Prerequisites:
    PAW_API_KEY, ANTHROPIC_API_KEY in the environment.

Usage:
    uv run python scripts/measure_finetune_fiscal.py --label 3080
    uv run python scripts/measure_finetune_fiscal.py --label 3080 --no-run --arms C
    uv run python scripts/measure_finetune_fiscal.py --label 3080 --skip-compile --arms A,B,C,D
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paw_kit.backend.programasweights import ProgramAsWeightsBackend  # noqa: E402

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

REFERENCE_MODEL = "claude-haiku-4-5-20251001"
REFERENCE_TEMPERATURE = 0.0
# B-3 (bug hunt 2026-09-11): arm D used to be called with a hardcoded 400-token cap,
# recorded nowhere, while the three compiled arms ran unbounded. The rule here needs real
# multi-step arithmetic, so Haiku reasons out loud and ran past that budget on 39 of 300
# cases; each was scored as a failure with no label, which put arm D at 85.7% when it was
# exactly right on 257 of the 261 it finished (98.5%). The baseline was deflated ~13
# points, which overstates how close the finetune compiler (49.0%) is to the frontier.
#
# 2000, not 30. `measure_finetune_lookup.py` has a constant of the same name whose value
# is 30 and whose docstring says why: its answer is six characters and there is nothing to
# reason about. Copying that value here would truncate essentially every arm-D answer
# instead of 13% of them.
REFERENCE_MAX_TOKENS = 2000

FAST_COMPILER = "paw-4b-qwen3-0.6b"
FINETUNE_COMPILER = "paw-ft-bs48"

N_FOLD = 8
N_EVAL = 300
SEED = 20260910
RANGE_START = dt.date(2024, 1, 1)
RANGE_END = dt.date(2027, 12, 31)
BOUNDARY_WINDOW = 10  # days either side of each fiscal-year start that get oversampled

FORMATS = ["iso", "long", "day_first", "weekday_prefixed"]

# ---------------------------------------------------------------------------- spec
#
# Plain prose, one statement of the rule plus two worked examples. Both compilers and the
# reference model get this byte-for-byte identical text; arms B and C additionally get the
# 8 folding examples appended by the backend's own `_render_spec_with_examples`.
SPEC = (
    "Label a calendar date with the fiscal week it falls in.\n"
    "\n"
    "The fiscal calendar works like this. A fiscal year begins on the first Monday of "
    "February and ends on the Sunday before the next fiscal year begins. The fiscal year "
    "is named after the calendar year its February start falls in: the fiscal year that "
    "begins on the first Monday of February 2026 is fiscal year 2026. Weeks are numbered "
    "starting from 1: the seven days beginning on that first Monday of February are week "
    "1, the next seven days are week 2, and so on. Because the fiscal year runs from "
    "February to February, a date that falls in January, or in February before that "
    "year's first Monday, belongs to the PREVIOUS fiscal year, and its week number "
    "continues that previous year's count -- so those dates land in the high forties or "
    "low fifties. A fiscal year has 52 weeks, or 53 when the extra day pushes it over, so "
    "the last week of a fiscal year is week 52 or week 53.\n"
    "\n"
    "Note that this is NOT the ISO 8601 week number, which counts from January. A fiscal "
    "week number is always the number of whole weeks since the first Monday of February, "
    "plus one.\n"
    "\n"
    "The input is a single date, written in any of these forms: 2026-03-03, "
    "March 3, 2026, 3 Mar 2026, or Tuesday 3 March 2026.\n"
    "\n"
    "Output exactly one label and nothing else, in the form FY<year>-W<nn>, where <year> "
    "is the four-digit fiscal year and <nn> is the two-digit week number, zero-padded "
    "below ten.\n"
    "\n"
    "Worked example one. The date is 3 Mar 2026. The first Monday of February 2026 is "
    "2 February 2026, so fiscal year 2026 began that day. 3 March 2026 is 29 days after "
    "2 February 2026. 29 divided by 7 is 4 whole weeks with 1 day left over, so it is in "
    "week 4 + 1 = 5. The label is FY2026-W05.\n"
    "\n"
    "Worked example two. The date is 2026-01-20. That is in January, so it belongs to the "
    "previous fiscal year, 2025. Fiscal year 2025 began on the first Monday of February "
    "2025, which is 3 February 2025. 20 January 2026 is 351 days after 3 February 2025. "
    "351 divided by 7 is 50 whole weeks with 1 day left over, so it is in week 50 + 1 = "
    "51. The label is FY2025-W51.\n"
)

LABEL_RE = re.compile(r"FY\d{4}-W\d{2}")
STRICT_LABEL_RE = re.compile(r"^FY\d{4}-W\d{2}$")


# ------------------------------------------------------------------- ground truth

def first_monday_of_february(year: int) -> dt.date:
    """The first Monday on or after 1 February of `year` -- the fiscal-year epoch."""
    d = dt.date(year, 2, 1)
    return d + dt.timedelta(days=(0 - d.weekday()) % 7)


def fiscal_year_of(date: dt.date) -> int:
    """The fiscal year `date` belongs to."""
    return date.year if date >= first_monday_of_february(date.year) else date.year - 1


def fiscal_year_weeks(fiscal_year: int) -> int:
    """How many weeks fiscal year `fiscal_year` contains (52 or 53)."""
    start = first_monday_of_february(fiscal_year)
    end = first_monday_of_february(fiscal_year + 1)
    return (end - start).days // 7


def fiscal_label(date: dt.date) -> str:
    """`FY<year>-W<nn>` for `date`. This is the ground truth; no model is involved."""
    fy = fiscal_year_of(date)
    week = (date - first_monday_of_february(fy)).days // 7 + 1
    return f"FY{fy}-W{week:02d}"


def is_boundary(date: dt.date) -> bool:
    """True if `date` is within BOUNDARY_WINDOW days of any fiscal-year start."""
    for year in (date.year - 1, date.year, date.year + 1):
        if abs((date - first_monday_of_february(year)).days) <= BOUNDARY_WINDOW:
            return True
    return False


# ------------------------------------------------------------------- formatting

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render_date(date: dt.date, fmt: str) -> str:
    """Render `date` in one of the four unambiguous input formats.

    No numeric-only forms: `03/04/2026` is day-first in some places and month-first in
    others, and a date the *task* cannot resolve is not a fair test of the adapter.
    Note that `weekday_prefixed` hands the model the weekday for free, which is exactly
    the fact the rule turns on -- the per-format scoring below is there to show whether
    any arm exploits that.
    """
    if fmt == "iso":
        return date.isoformat()
    if fmt == "long":
        return f"{_MONTHS[date.month - 1]} {date.day}, {date.year}"
    if fmt == "day_first":
        return f"{date.day} {_MONTHS_ABBR[date.month - 1]} {date.year}"
    if fmt == "weekday_prefixed":
        return (f"{_WEEKDAYS[date.weekday()]} {date.day} "
                f"{_MONTHS[date.month - 1]} {date.year}")
    raise ValueError(f"unknown format {fmt!r}")


# ------------------------------------------------------------------- fixture

# Hand-picked so the folding pool covers all four formats (two each) and three boundary
# cases: the fiscal-year start itself, the day before it (which wraps to the previous
# fiscal year's final week), and a second year's start. These eight dates are removed
# from the evaluation pool before it is drawn, so the two sets are disjoint by
# construction rather than by luck.
FOLDING_DATES: List[Tuple[dt.date, str]] = [
    (dt.date(2024, 2, 5), "iso"),               # first Monday of Feb 2024 -> FY2024-W01
    (dt.date(2024, 2, 4), "long"),              # the Sunday before it -> FY2023-W52
    (dt.date(2026, 2, 2), "weekday_prefixed"),  # first Monday of Feb 2026 -> FY2026-W01
    (dt.date(2025, 3, 15), "day_first"),
    (dt.date(2025, 11, 20), "iso"),
    (dt.date(2024, 12, 1), "long"),
    (dt.date(2027, 1, 9), "day_first"),         # January -> previous fiscal year
    (dt.date(2026, 6, 23), "weekday_prefixed"),
]


def build_fixture(out_path: Path) -> Dict[str, Any]:
    """Build the 300-case evaluation set and the 8-case folding pool, deterministically.

    No API calls: the labels are computed by `fiscal_label`, so this fixture is exact and
    costs nothing to rebuild.
    """
    rng = random.Random(SEED)

    folding = []
    for date, fmt in FOLDING_DATES:
        folding.append({
            "date": date.isoformat(),
            "format": fmt,
            "input": render_date(date, fmt),
            "expected": fiscal_label(date),
            "boundary": is_boundary(date),
        })

    reserved = {d for d, _ in FOLDING_DATES}

    all_dates = []
    d = RANGE_START
    while d <= RANGE_END:
        if d not in reserved:
            all_dates.append(d)
        d += dt.timedelta(days=1)

    boundary_dates = [d for d in all_dates if is_boundary(d)]
    other_dates = [d for d in all_dates if not is_boundary(d)]

    # Oversample: every day in the +/-10 window around each of the four in-range fiscal
    # starts goes in (81 of them after the 3 reserved for folding), and the remaining 219
    # are drawn uniformly from the rest of the range.
    chosen = list(boundary_dates)
    chosen += rng.sample(other_dates, N_EVAL - len(chosen))
    rng.shuffle(chosen)

    evaluation = []
    for i, date in enumerate(chosen):
        fmt = FORMATS[i % len(FORMATS)]
        evaluation.append({
            "id": f"d{i:03d}",
            "date": date.isoformat(),
            "format": fmt,
            "input": render_date(date, fmt),
            "expected": fiscal_label(date),
            "boundary": is_boundary(date),
            "fiscal_year": fiscal_year_of(date),
            "week": int(fiscal_label(date).split("-W")[1]),
        })

    fixture = {
        "task": "fiscal_week_label",
        "spec": SPEC,
        "rule": (
            "A fiscal year begins on the first Monday of February and is named for that "
            "calendar year. Weeks are numbered from 1 starting that Monday. Dates before "
            "the first Monday of February belong to the previous fiscal year and continue "
            "its week count, whose final week is 52 or 53."
        ),
        "ground_truth": "computed by scripts/measure_finetune_fiscal.py:fiscal_label -- no model, no judge",
        "seed": SEED,
        "range": [RANGE_START.isoformat(), RANGE_END.isoformat()],
        "boundary_window_days": BOUNDARY_WINDOW,
        "fiscal_year_starts": {
            str(y): first_monday_of_february(y).isoformat() for y in range(2023, 2029)
        },
        "fiscal_year_lengths_weeks": {str(y): fiscal_year_weeks(y) for y in range(2023, 2029)},
        "formats": FORMATS,
        "n_evaluation": len(evaluation),
        "n_boundary": sum(1 for c in evaluation if c["boundary"]),
        "n_folding": len(folding),
        "folding_examples": folding,
        "evaluation": evaluation,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"[data] wrote {out_path} ({len(evaluation)} eval, "
          f"{fixture['n_boundary']} boundary, {len(folding)} folding)")
    return fixture


# ------------------------------------------------------------------- arms

def arm_specs(out_dir: Path) -> List[Dict[str, Any]]:
    return [
        {"arm": "A", "kind": "adapter", "compiler": FAST_COMPILER, "max_spec_examples": 0,
         "manifest": str(out_dir / "finetune_fiscal_A-paw-4b-qwen3-0.6b.paw"),
         "description": "fast compiler, no folded examples"},
        {"arm": "B", "kind": "adapter", "compiler": FAST_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_fiscal_B-paw-4b-qwen3-0.6b.paw"),
         "description": f"fast compiler, {N_FOLD} folded examples"},
        {"arm": "C", "kind": "adapter", "compiler": FINETUNE_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_fiscal_C-paw-ft-bs48.paw"),
         "description": f"finetune compiler, {N_FOLD} folded examples"},
        {"arm": "D", "kind": "api", "compiler": REFERENCE_MODEL, "max_spec_examples": 0,
         "manifest": None,
         "description": f"{REFERENCE_MODEL} zero-shot, temperature 0, no compile"},
    ]


def fold_examples(fixture: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"input": r["input"], "output": r["expected"]} for r in fixture["folding_examples"]]


def compile_arm(arm: Dict[str, Any], examples: List[Dict[str, str]], skip: bool) -> Dict[str, Any]:
    path = Path(arm["manifest"])
    if skip:
        if not path.exists():
            # Copied verbatim in spirit from scripts/measure_finetune_triage.py: `.paw`
            # manifests are gitignored, so a fresh worktree or a `git clean` makes them
            # vanish while every committed artifact still refers to them by program id.
            # Silently compiling a *different* adapter under a flag that promises not to
            # compile is worse than stopping.
            raise RuntimeError(
                f"--skip-compile was passed but arm {arm['arm']}'s manifest {path} does not "
                f"exist, so there is nothing to reuse. Refusing to silently compile a "
                f"different adapter under a flag that promises not to. Run "
                f"`--no-run --arms {arm['arm']}` first to (re)compile it; if the spec is "
                f"unchanged the service returns the same program from cache."
            )
        manifest = json.loads(path.read_text())
        print(f"[compile:{arm['arm']}] reusing {path} (program {manifest.get('program_id')}, "
              f"compile_wall_s={manifest.get('compile_wall_s')})")
        return manifest
    backend = ProgramAsWeightsBackend(
        compiler=arm["compiler"],
        max_spec_examples=arm["max_spec_examples"],
        public=False,  # the ProgramAsWeightsBackend default; stated, not assumed
    )
    use = examples if arm["max_spec_examples"] > 0 else []
    print(f"[compile:{arm['arm']}] {arm['compiler']} with {len(use)} example(s) -> {path}")
    t0 = time.perf_counter()
    backend.compile(SPEC, use, str(path))
    wall = time.perf_counter() - t0
    manifest = json.loads(path.read_text())
    print(f"[compile:{arm['arm']}] done in {wall:.1f}s, program {manifest.get('program_id')}, "
          f"folded={manifest.get('examples_folded_into_spec')}, "
          f"cache_hit={manifest.get('cache_hit')}")
    return manifest


def run_adapter_arm(arm: Dict[str, Any], evaluation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    backend = ProgramAsWeightsBackend(compiler=arm["compiler"],
                                      max_spec_examples=arm["max_spec_examples"])
    rows = []
    for i, case in enumerate(evaluation):
        t0 = time.perf_counter()
        try:
            raw = backend.infer(arm["manifest"], case["input"])
            err = None
        except Exception as exc:  # noqa: BLE001
            raw, err = "", f"{type(exc).__name__}: {exc}"
        rows.append(_row(case, raw, err, (time.perf_counter() - t0) * 1000.0))
        if (i + 1) % 50 == 0:
            print(f"  [{arm['arm']}] {i + 1}/{len(evaluation)}")
    return rows


def run_api_arm(arm: Dict[str, Any], evaluation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Arm D: the same spec text, zero-shot, temperature 0, one call per case.

    `temperature` goes through `extra_body`: the installed `anthropic` SDK (1.4.0)
    removed it from `Messages.create`'s typed signature, so passing it as a named
    argument raises `TypeError`. The wire API still honours the field. Same workaround
    as scripts/measure_finetune_triage.py.
    """
    if anthropic is None:
        raise RuntimeError("arm D needs the `anthropic` package")
    client = anthropic.Anthropic()
    rows = []
    for i, case in enumerate(evaluation):
        prompt = f"{SPEC}\n\nInput: {case['input']}\nOutput:"
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(
                model=REFERENCE_MODEL,
                max_tokens=REFERENCE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"temperature": REFERENCE_TEMPERATURE},
            )
            raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
            err = None
            # `stop_reason == "max_tokens"` is the direct evidence of a truncated answer,
            # as opposed to an answer that was simply wrong. Recorded per case so the
            # truncation count in the summary is an observation, not an inference from an
            # absent label. (B-3.)
            stop_reason = getattr(resp, "stop_reason", None)
        except Exception as exc:  # noqa: BLE001
            raw, err, stop_reason = "", f"{type(exc).__name__}: {exc}", None
        rows.append(_row(case, raw, err, (time.perf_counter() - t0) * 1000.0,
                         stop_reason=stop_reason))
        if (i + 1) % 50 == 0:
            print(f"  [{arm['arm']}] {i + 1}/{len(evaluation)}")
    return rows


def _row(case: Dict[str, Any], raw: str, err: Optional[str], latency_ms: float,
         stop_reason: Optional[str] = None) -> Dict[str, Any]:
    """Score one output against ground truth.

    Two notions of "parses": `strict_shape` is the whole output being exactly a label
    (what the spec asks for), `label_found` is a label appearing anywhere in the output
    (what a chattier arm produces). Exact match is judged on the *first label found*, so
    an arm that gets the answer right but wraps it in prose is not penalised twice --
    the strict-shape column is where that shows up.
    """
    stripped = (raw or "").strip()
    match = LABEL_RE.search(stripped)
    found = match.group(0) if match else None
    fy = week = None
    if found:
        fy = int(found[2:6])
        week = int(found[8:10])
    exp_fy = case["fiscal_year"]
    exp_week = case["week"]
    return {
        "id": case["id"],
        "date": case["date"],
        "format": case["format"],
        "input": case["input"],
        "expected": case["expected"],
        "boundary": case["boundary"],
        "raw": raw,
        "label": found,
        "error": err,
        "strict_shape": bool(STRICT_LABEL_RE.match(stripped)),
        "label_found": found is not None,
        "exact": found == case["expected"],
        "fy_correct": fy == exp_fy,
        "week_correct": week == exp_week,
        "week_off_by_one": week is not None and abs(week - exp_week) == 1,
        "week_delta": None if week is None else week - exp_week,
        "latency_ms": latency_ms,
        # None for the adapter arms, which have no upstream stop reason to report.
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_tokens",
    }


# ------------------------------------------------------------------- scoring

_METRICS = ["exact", "fy_correct", "week_correct", "week_off_by_one",
            "strict_shape", "label_found"]


def _rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-cut counts and rates, plus the answered/unanswered split.

    B-3: `exact` alone conflates "got the rule wrong" with "never produced a label",
    and for a token-capped reference arm those are different failures. `exact` keeps the
    whole cut as its denominator -- it is the number that answers "how often is this arm
    right?" -- and `exact_when_answered` reports the same count over the cases that
    produced a label at all. `truncated_n` is the count that explains the gap, taken from
    the API's own `stop_reason` rather than inferred from a missing label.
    """
    n = len(rows)
    counts = {m: sum(1 for r in rows if r[m]) for m in _METRICS}
    answered = [r for r in rows if r["label_found"]]
    exact_answered = sum(1 for r in answered if r["exact"])
    return {
        "n": n,
        "counts": counts,
        "rates_pct": {m: (counts[m] / n * 100.0 if n else 0.0) for m in _METRICS},
        "answered_n": len(answered),
        "unanswered_n": n - len(answered),
        "truncated_n": sum(1 for r in rows if r.get("truncated")),
        "error_n": sum(1 for r in rows if r["error"]),
        "exact_when_answered": exact_answered,
        "exact_when_answered_pct": (
            exact_answered / len(answered) * 100.0 if answered else None
        ),
    }


def score_arm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": _rates(rows)}
    out["by_format"] = {f: _rates([r for r in rows if r["format"] == f]) for f in FORMATS}
    out["by_boundary"] = {
        "boundary": _rates([r for r in rows if r["boundary"]]),
        "non_boundary": _rates([r for r in rows if not r["boundary"]]),
    }
    deltas: Dict[str, int] = {}
    for r in rows:
        key = "none" if r["week_delta"] is None else str(r["week_delta"])
        deltas[key] = deltas.get(key, 0) + 1
    out["week_delta_histogram"] = dict(sorted(
        deltas.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0)))
    fy_deltas: Dict[str, int] = {}
    for r in rows:
        if r["label"] is None:
            key = "none"
        else:
            key = str(int(r["label"][2:6]) - int(r["expected"][2:6]))
        fy_deltas[key] = fy_deltas.get(key, 0) + 1
    out["fy_delta_histogram"] = dict(sorted(
        fy_deltas.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0)))
    return out


def _latency(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "first_call_including_model_load": rows[0]["latency_ms"],
        "median": lat[len(lat) // 2],
        "mean_excluding_first": sum(r["latency_ms"] for r in rows[1:]) / max(1, len(rows) - 1),
        "min": lat[0], "max": lat[-1],
    }


# ------------------------------------------------------------------- suite

def write_suite(path: Path, evaluation: List[Dict[str, Any]], adapter_path: str) -> None:
    """A `paw-test` suite over the 300 cases: `expected` is the exact ground-truth label,
    plus a `regex_match` assertion on the label shape.

    The assertion vocabulary is regex_match / max_length / min_length / exact_match /
    not_contains (paw_kit/test/suite.py), and assertions are suite-wide rather than
    per-case, so the only rule expressible here is the *shape* of the label. Whether the
    label is the right one is `expected`, which `paw-test compare` surfaces per case and
    `paw-test check` does not score.
    """
    import yaml

    cases = [{"input": c["input"], "expected": c["expected"]} for c in evaluation]
    suite = {
        "task_name": "fiscal_week",
        "spec": SPEC,
        "adapter_path": adapter_path,
        "standard_cases": cases,
        "assertions": [
            {"rule": "regex_match", "pattern": r"FY\d{4}-W\d{2}"},
            {"rule": "max_length", "value": 200},
            {"rule": "not_contains", "value": "ERROR"},
        ],
        "fuzzing": {
            "inject_unicode": False,
            "empty_inputs": False,
            "whitespace_flood": False,
            "payload_extremes": False,
        },
    }
    path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=True, width=10000))
    print(f"wrote {path}")


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="3080")
    ap.add_argument("--out-dir", default="measurements")
    ap.add_argument("--data-path", default="measurements/finetune-fiscal-dates.json")
    ap.add_argument("--rebuild-data", action="store_true")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Reuse existing .paw manifests instead of compiling (no compile calls at all)")
    ap.add_argument("--arms", default="A,B,C,D", help="Comma-separated subset of arms to run")
    ap.add_argument("--suite-out", default="measurements/finetune-fiscal-suite.yaml")
    ap.add_argument("--no-run", action="store_true",
                    help="Build data and compile only; skip inference")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    if data_path.exists() and not args.rebuild_data:
        fixture = json.loads(data_path.read_text())
        print(f"[data] reusing {data_path} ({fixture['n_evaluation']} eval, "
              f"{fixture['n_boundary']} boundary, {fixture['n_folding']} folding)")
    else:
        fixture = build_fixture(data_path)

    evaluation = fixture["evaluation"]
    examples = fold_examples(fixture)

    wanted = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in arm_specs(out_dir) if a["arm"] in wanted]

    compiles_made = 0
    for arm in arms:
        if arm["kind"] != "adapter":
            arm["manifest_data"] = {}
            continue
        existed = Path(arm["manifest"]).exists()
        arm["manifest_data"] = compile_arm(arm, examples, args.skip_compile)
        if not (args.skip_compile and existed):
            compiles_made += 1
    print(f"[compile] compile calls made this run: {compiles_made}")

    if args.no_run:
        write_suite(Path(args.suite_out), evaluation,
                    str(out_dir / "finetune_fiscal_B-paw-4b-qwen3-0.6b.paw"))
        return 0

    summary_arms = []
    for arm in arms:
        print(f"[run:{arm['arm']}] {len(evaluation)} dates on {arm['compiler']}")
        rows = (run_api_arm(arm, evaluation) if arm["kind"] == "api"
                else run_adapter_arm(arm, evaluation))
        md = arm["manifest_data"]
        summary_arms.append({
            "arm": arm["arm"],
            "kind": arm["kind"],
            "description": arm["description"],
            "compiler": arm["compiler"],
            "max_spec_examples": arm["max_spec_examples"],
            "manifest": arm["manifest"],
            # Mirrored so the arms stay checkable without the gitignored .paw files.
            "program_id": md.get("program_id"),
            "compiler_snapshot": md.get("compiler_snapshot"),
            "examples_folded_into_spec": md.get("examples_folded_into_spec"),
            "compile_wall_s": md.get("compile_wall_s"),
            "public": md.get("public"),
            "cache_hit": md.get("cache_hit"),
            "status": md.get("status"),
            "slug": md.get("slug"),
            "spec_sha256": md.get("spec_sha256"),
            "full_spec_sha256": md.get("full_spec_sha256"),
            "compiled_at": md.get("compiled_at"),
            "manifest_version": md.get("manifest_version"),
            "errors": sum(1 for r in rows if r["error"]),
            "latency_ms": _latency(rows),
            "scores": score_arm(rows),
            "cases": rows,
        })

    summary = {
        "label": args.label,
        "task": "fiscal_week_label",
        "spec": SPEC,
        "ground_truth": "computed by code (fiscal_label); no judge, no teacher, exact",
        "reference_model": REFERENCE_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "reference_max_tokens": REFERENCE_MAX_TOKENS,
        # Every parameter that can change arm D's output, in one place. The 2026-09-10 run
        # recorded the model and the temperature but not the token cap -- which is the one
        # that decided 39 of its 300 results. (B-3.)
        "reference_sampling": {
            "model": REFERENCE_MODEL,
            "temperature": REFERENCE_TEMPERATURE,
            "max_tokens": REFERENCE_MAX_TOKENS,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "system": None,
            "prompt_template": "{SPEC}\n\nInput: {input}\nOutput:",
            "note": (
                "top_p/top_k/stop_sequences/system are not passed to the API at all; "
                "None records that, so the absence is stated rather than left to be "
                "inferred from the source. temperature goes through extra_body because "
                "anthropic 1.4.0 removed it from the typed signature."
            ),
        },
        "adapter_temperature": 0.0,
        "adapter_temperature_note": (
            "programasweights' PawFunction defaults to temperature=0.0 (greedy). "
            "ProgramAsWeightsBackend.infer exposes no temperature argument, so 0 here is "
            "the upstream default rather than something paw-kit sets or can change."
        ),
        "data_fixture": str(data_path),
        "seed": SEED,
        "n_evaluation": len(evaluation),
        "n_boundary": fixture["n_boundary"],
        "n_folding": fixture["n_folding"],
        "compiles_made_this_run": compiles_made,
        "arms": summary_arms,
    }

    for a in summary_arms:
        r = a["scores"]["overall"]["rates_pct"]
        b = a["scores"]["by_boundary"]
        print(f"\n[{a['arm']}] {a['description']}")
        o = a["scores"]["overall"]
        print(f"   exact {r['exact']:.1f}%  fy {r['fy_correct']:.1f}%  "
              f"week {r['week_correct']:.1f}%  week+/-1 {r['week_off_by_one']:.1f}%")
        ewa = o["exact_when_answered_pct"]
        print(f"   answered {o['answered_n']}/{o['n']} "
              f"(truncated {o['truncated_n']}, errors {o['error_n']})  "
              f"exact when answered " + ("n/a" if ewa is None else f"{ewa:.1f}%"))
        print(f"   strict shape {r['strict_shape']:.1f}%  label found {r['label_found']:.1f}%")
        print(f"   boundary exact {b['boundary']['rates_pct']['exact']:.1f}%  "
              f"non-boundary exact {b['non_boundary']['rates_pct']['exact']:.1f}%")
        print(f"   by format: " + "  ".join(
            f"{f} {a['scores']['by_format'][f]['rates_pct']['exact']:.1f}%" for f in FORMATS))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"finetune-fiscal-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")

    write_suite(Path(args.suite_out), evaluation,
                str(out_dir / "finetune_fiscal_B-paw-4b-qwen3-0.6b.paw"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
