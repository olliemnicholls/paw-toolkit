"""Does the finetune compiler (`paw-ft-bs48`) beat the fast compiler (`paw-4b-qwen3-0.6b`)
on a task the fast compiler actually *fails*?

`measurements/README.md`'s "Finetune compiler (`paw-ft-bs48`), for real" section ran the
finetune compiler once, on phone extraction, where an 8-example fast compile already sat
at 93% structural -- and got 132/134 byte-identical outputs. Its own stated scope limit
was that "the places `paw-ft-bs48` would plausibly earn its wall-time -- a task the fast
compiler genuinely fails at ... -- are exactly the places this test did not go."

This script goes there. The task is ticket triage: the same spec and `Triage` response
model as the recorded 60%-agreement run
(`measurements/triage-semantic-agreement-3080-20260909-002033.json`), which is the one
task in this repo where the fast compiler is known to be substantially wrong rather than
near-saturated.

Three arms, each compiled exactly once through `ProgramAsWeightsBackend` (so the manifest
and its lineage are real), all `public=False`:

    A  fast compiler `paw-4b-qwen3-0.6b`, max_spec_examples=0
    B  fast compiler, 8 folding examples folded into the spec text
    C  finetune compiler `paw-ft-bs48`, the same 8 examples, via compile_async polling

Evaluation set: the 20 recorded tickets plus 40 freshly generated ones = 60. A further 8
fresh tickets form the folding pool, used only as folded examples and never evaluated.
Every evaluation ticket is labelled *twice*, independently, by the teacher, so the
teacher's self-agreement is reported as the ceiling any adapter can be measured against.
All teacher calls use temperature 0.

Limitation stated up front: the 48 fresh tickets are teacher-generated, so the evaluation
set and its labels come from the same model family. See the README section.

Prerequisites:
    PAW_API_KEY, ANTHROPIC_API_KEY in the environment.

Usage:
    uv run python scripts/measure_finetune_triage.py --label 3080
    uv run python scripts/measure_finetune_triage.py --label 3080 --skip-compile
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from paw_kit import ProgramAsWeightsBackend

try:
    import anthropic
except ImportError:  # pragma: no cover
    print("pip install anthropic", file=sys.stderr)
    sys.exit(2)

TEACHER_MODEL = "claude-haiku-4-5-20251001"
TEACHER_TEMPERATURE = 0.0

# Identical to scripts/measure_triage_semantic_agreement.py -- this run is only
# comparable to the recorded 60% if the spec is byte-for-byte the same.
SPEC = (
    "Classify a customer support ticket into priority (low, medium, high, or critical), "
    "department (billing, technical, sales, or general), and urgency_score (integer 1-5)."
)

FAST_COMPILER = "paw-4b-qwen3-0.6b"
FINETUNE_COMPILER = "paw-ft-bs48"

N_FRESH = 48
N_FOLD = 8          # fresh tickets used only as folded examples
N_FRESH_EVAL = 40   # fresh tickets that go into the evaluation set
GEN_BATCH = 8

CATEGORIES = ["billing", "technical", "account", "shipping", "security", "general"]


class Triage(BaseModel):
    priority: str
    department: str
    urgency_score: int


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _recorded_tickets(path: Path) -> List[str]:
    """The 20 tickets from the recorded 60%-agreement run, read from its artifact."""
    data = json.loads(path.read_text())
    return [c["ticket"] for c in data["cases"]]


# ---------------------------------------------------------------- teacher calls

def _teacher_text(client: "anthropic.Anthropic", prompt: str, max_tokens: int,
                  temperature: Optional[float] = TEACHER_TEMPERATURE) -> str:
    """One teacher call at temperature 0.

    `temperature` is passed via `extra_body`, not as a named argument: the installed
    `anthropic` SDK (1.4.0) removed `temperature` from `Messages.create`'s typed
    signature, so `client.messages.create(..., temperature=0.0)` raises
    `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`.
    The wire API still honours the field, so `extra_body` sets it. This is the same
    break that stops `paw_kit/test/judge.py:277` (`anthropic_judge`) working at all --
    see the README section for the full write-up.
    """
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["extra_body"] = {"temperature": temperature}
    resp = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def label_ticket(client: "anthropic.Anthropic", ticket: str, nonce: str = "") -> Optional[Dict[str, Any]]:
    """One independent teacher label for one ticket, temperature 0.

    `nonce` distinguishes the two independent labelling passes. At temperature 0 an
    identical prompt would be near-deterministic (and, with prompt caching, is likely
    to return literally the same answer), which would make "teacher self-agreement"
    measure the API's determinism rather than the task's genuine ambiguity. The two
    passes therefore differ in how the question is framed while asking for exactly the
    same judgement -- see the README section's limitations.
    """
    prompt = (
        f"{SPEC}\n\nRespond with ONLY a JSON object with exactly the keys "
        f'"priority", "department", "urgency_score". No other text.\n\n'
        f"{nonce}Ticket: {ticket}"
    )
    try:
        text = _teacher_text(client, prompt, max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        print(f"    [teacher-error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    match = _JSON_RE.search(text)
    if not match:
        print(f"    [teacher-no-json] {text!r}", file=sys.stderr)
        return None
    try:
        return Triage(**json.loads(match.group(0))).model_dump()
    except Exception as exc:  # noqa: BLE001
        print(f"    [teacher-bad-json] {type(exc).__name__}: {text!r}", file=sys.stderr)
        return None


def _normalise(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _too_similar(candidate: str, existing: List[str]) -> bool:
    """Reject exact (normalised) repeats and near-duplicates by token overlap."""
    cn = _normalise(candidate)
    ctoks = set(cn.split())
    if not ctoks:
        return True
    for e in existing:
        en = _normalise(e)
        if cn == en:
            return True
        etoks = set(en.split())
        if not etoks:
            continue
        jac = len(ctoks & etoks) / len(ctoks | etoks)
        if jac >= 0.6:
            return True
    return False


GEN_SETTINGS = [
    "a consumer mobile banking app",
    "a B2B SaaS analytics platform",
    "an online electronics retailer",
    "a cloud file-storage service",
    "a subscription meal-kit company",
    "a developer API and hosting provider",
    "an online course marketplace",
    "a hardware IoT device vendor",
    "a travel booking site",
    "a payroll and HR software vendor",
]

GEN_VOICES = [
    "a frustrated long-time customer",
    "a calm enterprise IT administrator",
    "a confused first-time user",
    "a terse power user in a hurry",
    "a polite customer writing at length",
    "a customer writing in a language other than English",
]


def generate_tickets(client: "anthropic.Anthropic", recorded: List[str], n: int,
                     max_calls: int = 14) -> List[Dict[str, Any]]:
    """Generate `n` fresh, deduplicated tickets with the teacher, in batches.

    Deliberately NOT temperature 0. A first attempt asked for batches at temperature 0
    with a prompt that varied only in which categories it named; the teacher returned
    substantially the same tickets every time and deduplication stalled at 24/48 unique
    after 12 calls. Generation therefore runs at the API's default sampling temperature,
    with the product setting, customer voice and category mix varied per batch and the
    tickets collected so far quoted back as "do not repeat these". Labelling -- the part
    of this measurement that has to be reproducible -- is still pinned to temperature 0.
    """
    fresh: List[Dict[str, Any]] = []
    calls = 0
    batch_idx = 0
    while len(fresh) < n and calls < max_calls:
        cats = [CATEGORIES[(batch_idx * 5 + i) % len(CATEGORIES)] for i in range(GEN_BATCH)]
        setting = GEN_SETTINGS[batch_idx % len(GEN_SETTINGS)]
        voice = GEN_VOICES[batch_idx % len(GEN_VOICES)]
        avoid = [f["ticket"] for f in fresh][-24:]
        avoid_block = ""
        if avoid:
            avoid_block = (
                "\n\nDo NOT write anything resembling these tickets, which have already "
                "been collected -- pick different products, symptoms, wording and names:\n"
                + "\n".join(f"- {a[:110]}" for a in avoid)
            )
        prompt = (
            "You are helping build a test set for a customer-support ticket triage system.\n"
            f"The company this batch is for: {setting}.\n"
            f"Write {GEN_BATCH} realistic, varied customer support tickets, one for each of these "
            f"categories in order: {', '.join(cats)}.\n"
            f"At least two of them should read like {voice}.\n\n"
            "Requirements:\n"
            "- Write them as the customer would actually write them, not as summaries.\n"
            "- Spread the urgency widely: some trivial, some routine, some genuinely severe.\n"
            "- Vary the length: some a single short line, some a full paragraph.\n"
            "- Make two or three of them messy: typos, ALL CAPS, missing punctuation, or "
            "written in a language other than English.\n"
            "- Use concrete, specific and DIFFERENT details (order numbers, product names, "
            "error codes, dates) in every ticket.\n"
            "- Do not number them or add any commentary."
            + avoid_block
            + "\n\nRespond with ONLY a JSON array of "
            f"{GEN_BATCH} strings. No other text."
        )
        calls += 1
        batch_idx += 1
        try:
            # temperature=None -> API default sampling; see this function's docstring.
            text = _teacher_text(client, prompt, max_tokens=2500, temperature=None)
        except Exception as exc:  # noqa: BLE001
            print(f"  [gen-error] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print(f"  [gen-no-json] {text[:200]!r}", file=sys.stderr)
            continue
        try:
            items = json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            print(f"  [gen-bad-json] {text[:200]!r}", file=sys.stderr)
            continue
        added = 0
        for i, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                continue
            if len(fresh) >= n:
                break
            pool = recorded + [f["ticket"] for f in fresh]
            if _too_similar(item, pool):
                print(f"  [dedup] dropped {item[:60]!r}")
                continue
            fresh.append({
                "ticket": item.strip(),
                "generated_category": cats[i] if i < len(cats) else None,
                "generated_setting": setting,
                "gen_batch": batch_idx,
            })
            added += 1
        print(f"  [gen] batch {batch_idx} ({setting}): +{added}, have {len(fresh)}/{n} "
              f"({calls} generation calls)")
    return fresh


# ---------------------------------------------------------------- fixture

def build_fixture(client: "anthropic.Anthropic", recorded_path: Path, out_path: Path) -> Dict[str, Any]:
    recorded = _recorded_tickets(recorded_path)
    print(f"[data] {len(recorded)} recorded tickets from {recorded_path}")

    print(f"[data] generating {N_FRESH} fresh tickets with {TEACHER_MODEL} (temperature {TEACHER_TEMPERATURE})")
    fresh = generate_tickets(client, recorded, N_FRESH)
    if len(fresh) < N_FRESH:
        raise RuntimeError(f"only generated {len(fresh)}/{N_FRESH} unique fresh tickets")

    folding = fresh[:N_FOLD]
    fresh_eval = fresh[N_FOLD:N_FOLD + N_FRESH_EVAL]

    evaluation: List[Dict[str, Any]] = []
    for i, t in enumerate(recorded):
        evaluation.append({"id": f"rec-{i:02d}", "ticket": t, "source": "recorded",
                           "generated_category": None})
    for i, f in enumerate(fresh_eval):
        evaluation.append({"id": f"gen-{i:02d}", "ticket": f["ticket"], "source": "generated",
                           "generated_category": f["generated_category"]})

    print(f"[data] labelling {len(evaluation)} evaluation tickets TWICE (independently)")
    for j, row in enumerate(evaluation):
        row["teacher_label_1"] = label_ticket(client, row["ticket"], nonce="")
        row["teacher_label_2"] = label_ticket(
            client, row["ticket"],
            nonce="Classify the following support ticket.\n",
        )
        if (j + 1) % 10 == 0:
            print(f"  [label] {j + 1}/{len(evaluation)}")

    print(f"[data] labelling {len(folding)} folding tickets once")
    for row in folding:
        row["teacher_label"] = label_ticket(client, row["ticket"], nonce="")

    fixture = {
        "spec": SPEC,
        "teacher_model": TEACHER_MODEL,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "recorded_source": str(recorded_path),
        "limitation": (
            "The 48 fresh tickets were generated by the same teacher model that labels them "
            "(claude-haiku-4-5-20251001). The evaluation set is therefore not independent of "
            "the labeller: tickets may be unrepresentatively easy for this model to classify, "
            "and the teacher-vs-teacher ceiling reported here is a ceiling on agreement with "
            "*this* teacher, not on correctness."
        ),
        "n_evaluation": len(evaluation),
        "n_folding": len(folding),
        "folding_examples": folding,
        "evaluation": evaluation,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"[data] wrote {out_path}")
    return fixture


# ---------------------------------------------------------------- arms

def arm_specs(out_dir: Path) -> List[Dict[str, Any]]:
    return [
        {"arm": "A", "compiler": FAST_COMPILER, "max_spec_examples": 0,
         "manifest": str(out_dir / "finetune_triage_A-paw-4b-qwen3-0.6b.paw"),
         "description": "fast compiler, no folded examples"},
        {"arm": "B", "compiler": FAST_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_triage_B-paw-4b-qwen3-0.6b.paw"),
         "description": f"fast compiler, {N_FOLD} folded examples"},
        {"arm": "C", "compiler": FINETUNE_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_triage_C-paw-ft-bs48.paw"),
         "description": f"finetune compiler, {N_FOLD} folded examples"},
    ]


def fold_examples(fixture: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    for row in fixture["folding_examples"]:
        if row.get("teacher_label"):
            out.append({"input": row["ticket"], "output": json.dumps(row["teacher_label"])})
    return out


def compile_arm(arm: Dict[str, Any], examples: List[Dict[str, str]], skip: bool) -> Dict[str, Any]:
    path = Path(arm["manifest"])
    if skip:
        if not path.exists():
            # Do NOT fall through to a compile here. `.paw` manifests are gitignored
            # (`.gitignore:39-40`), so a `git clean` or a fresh worktree makes them
            # vanish while every committed artifact still refers to them by program id.
            # The old behaviour silently spent a compile call and substituted a *new*
            # adapter for the one the recorded numbers came from -- a measurement that
            # reports itself as `--skip-compile` while quietly re-compiling is worse
            # than one that stops. Compile the arm explicitly instead.
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
        # public=False is the ProgramAsWeightsBackend default and is left implicit
        # nowhere: state it, because these specs carry folded real ticket text.
        public=False,
    )
    use = examples if arm["max_spec_examples"] > 0 else []
    print(f"[compile:{arm['arm']}] {arm['compiler']} with {len(use)} example(s) -> {path}")
    t0 = time.perf_counter()
    backend.compile(SPEC, use, str(path))
    wall = time.perf_counter() - t0
    manifest = json.loads(path.read_text())
    print(f"[compile:{arm['arm']}] done in {wall:.1f}s, program {manifest.get('program_id')}, "
          f"folded={manifest.get('examples_folded_into_spec')}")
    return manifest


def run_arm(arm: Dict[str, Any], evaluation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    backend = ProgramAsWeightsBackend(compiler=arm["compiler"], max_spec_examples=arm["max_spec_examples"])
    rows = []
    for i, case in enumerate(evaluation):
        t0 = time.perf_counter()
        try:
            raw = backend.infer(arm["manifest"], case["ticket"])
            err = None
        except Exception as exc:  # noqa: BLE001
            raw, err = "", f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed = None
        parse_error = err
        if err is None:
            match = _JSON_RE.search(raw)
            if not match:
                parse_error = "no JSON object in output"
            else:
                try:
                    parsed = Triage(**json.loads(match.group(0))).model_dump()
                except Exception as exc:  # noqa: BLE001
                    parse_error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "id": case["id"],
            "ticket": case["ticket"],
            "raw": raw,
            "parsed": parsed,
            "parse_error": parse_error,
            "latency_ms": latency_ms,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{arm['arm']}] {i + 1}/{len(evaluation)}")
    return rows


# ---------------------------------------------------------------- scoring

def _score(preds: List[Optional[Dict[str, Any]]], golds: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Agreement of `preds` against `golds`. A None on either side counts as a
    non-agreement on every metric (never silently dropped)."""
    n = len(preds)
    counts = {"priority": 0, "department": 0, "urgency_exact": 0, "urgency_within_1": 0,
              "full_exact": 0, "full_urgency_within_1": 0}
    scored = 0
    for p, g in zip(preds, golds):
        if p is None or g is None:
            continue
        scored += 1
        pri = p["priority"].strip().lower() == g["priority"].strip().lower()
        dep = p["department"].strip().lower() == g["department"].strip().lower()
        ue = int(p["urgency_score"]) == int(g["urgency_score"])
        u1 = abs(int(p["urgency_score"]) - int(g["urgency_score"])) <= 1
        counts["priority"] += pri
        counts["department"] += dep
        counts["urgency_exact"] += ue
        counts["urgency_within_1"] += u1
        counts["full_exact"] += pri and dep and ue
        counts["full_urgency_within_1"] += pri and dep and u1
    return {
        "n": n,
        "n_scored": scored,
        "counts": counts,
        # Rates are over all n, not over n_scored: an unparseable output is a
        # disagreement, not a missing datapoint.
        "rates_pct": {k: (v / n * 100.0 if n else 0.0) for k, v in counts.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="3080")
    ap.add_argument("--out-dir", default="measurements")
    ap.add_argument("--data-path", default="measurements/finetune-triage-tickets.json",
                    help="JSON fixture of tickets + teacher labels; built if absent")
    ap.add_argument("--rebuild-data", action="store_true",
                    help="Regenerate and re-label the fixture even if it already exists (costs teacher calls)")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Reuse existing .paw manifests instead of compiling (no compile calls at all)")
    ap.add_argument("--arms", default="A,B,C", help="Comma-separated subset of arms to run")
    ap.add_argument("--recorded", default="measurements/triage-semantic-agreement-3080-20260909-002033.json")
    ap.add_argument("--suite-out", default="measurements/finetune-triage-suite.yaml")
    ap.add_argument("--no-run", action="store_true", help="Build data and compile only; skip GPU inference")
    ap.add_argument("--judge-compare-report", default=None,
                    help="Judge a `paw-test compare --json` report and exit (workaround for the "
                         "paw_kit/test/judge.py:277 SDK break; see judge_compare_report)")
    ap.add_argument("--judge-out", default="measurements/finetune-triage-judge-BC.json")
    args = ap.parse_args()

    if args.judge_compare_report:
        return judge_compare_report(Path(args.judge_compare_report), Path(args.judge_out))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    client = anthropic.Anthropic()

    if data_path.exists() and not args.rebuild_data:
        fixture = json.loads(data_path.read_text())
        print(f"[data] reusing {data_path} "
              f"({fixture['n_evaluation']} eval, {fixture['n_folding']} folding)")
    else:
        fixture = build_fixture(client, Path(args.recorded), data_path)

    evaluation = fixture["evaluation"]
    examples = fold_examples(fixture)
    print(f"[data] {len(examples)} folding examples available")

    wanted = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in arm_specs(out_dir) if a["arm"] in wanted]

    compiles_made = 0
    for arm in arms:
        existed = Path(arm["manifest"]).exists()
        arm["manifest_data"] = compile_arm(arm, examples, args.skip_compile)
        if not (args.skip_compile and existed):
            compiles_made += 1
    print(f"[compile] compile calls made this run: {compiles_made}")

    if args.no_run:
        write_suite(Path(args.suite_out), evaluation)
        return 0

    results = {}
    for arm in arms:
        print(f"[run:{arm['arm']}] {len(evaluation)} tickets on {arm['compiler']}")
        results[arm["arm"]] = run_arm(arm, evaluation)

    gold1 = [c.get("teacher_label_1") for c in evaluation]
    gold2 = [c.get("teacher_label_2") for c in evaluation]

    summary_arms = []
    for arm in arms:
        rows = results[arm["arm"]]
        preds = [r["parsed"] for r in rows]
        lat = sorted(r["latency_ms"] for r in rows)
        n_fail = sum(1 for r in rows if r["parsed"] is None)
        summary_arms.append({
            "arm": arm["arm"],
            "description": arm["description"],
            "compiler": arm["compiler"],
            "max_spec_examples": arm["max_spec_examples"],
            "manifest": arm["manifest"],
            "program_id": arm["manifest_data"].get("program_id"),
            "compiler_snapshot": arm["manifest_data"].get("compiler_snapshot"),
            "examples_folded_into_spec": arm["manifest_data"].get("examples_folded_into_spec"),
            "compile_wall_s": arm["manifest_data"].get("compile_wall_s"),
            # A-2: the manifest key that records the *requested* visibility was renamed
            # `public` -> `public_requested`, and `public_confirmed` (three-state: True /
            # False / None-with-a-reason) now carries what the server actually reported.
            # Read the new name with a legacy fallback so a summary built from a manifest
            # written before the rename still reports the value instead of silently None.
            "public_requested": arm["manifest_data"].get(
                "public_requested", arm["manifest_data"].get("public")),
            "public_confirmed": arm["manifest_data"].get("public_confirmed"),
            "public_confirmed_reason": arm["manifest_data"].get("public_confirmed_reason"),
            "cache_hit": arm["manifest_data"].get("cache_hit"),
            # Mirrored so the arms stay checkable without the gitignored .paw files.
            # full_spec_sha256 is the load-bearing one: it covers spec + folded
            # examples, so B and C sharing it is what proves they differ only in
            # compiler. status/slug/compiled_at complete the manifest's identity.
            "status": arm["manifest_data"].get("status"),
            "slug": arm["manifest_data"].get("slug"),
            "spec_sha256": arm["manifest_data"].get("spec_sha256"),
            "full_spec_sha256": arm["manifest_data"].get("full_spec_sha256"),
            "compiled_at": arm["manifest_data"].get("compiled_at"),
            "compiler_version": arm["manifest_data"].get("manifest_version"),
            "parse_failures": n_fail,
            "parse_failure_rate_pct": n_fail / len(rows) * 100.0,
            "latency_ms": {
                "first_call_including_model_load": rows[0]["latency_ms"],
                "median": lat[len(lat) // 2],
                "mean_excluding_first": (
                    sum(r["latency_ms"] for r in rows[1:]) / max(1, len(rows) - 1)
                ),
                "min": lat[0], "max": lat[-1],
            },
            "vs_teacher_label_1": _score(preds, gold1),
            "vs_teacher_label_2": _score(preds, gold2),
            "cases": rows,
        })

    teacher_ceiling = _score(gold1, gold2)

    summary = {
        "label": args.label,
        "spec": SPEC,
        "teacher_model": TEACHER_MODEL,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "adapter_temperature": 0.0,
        "adapter_temperature_note": (
            "programasweights' PawFunction defaults to temperature=0.0 (greedy) -- see "
            ".venv/.../programasweights/runtime_llamacpp.py:399. ProgramAsWeightsBackend.infer "
            "exposes no temperature argument, so 0 here is the upstream default rather than "
            "something paw-kit sets or can change."
        ),
        "data_fixture": str(data_path),
        "n_evaluation": len(evaluation),
        "n_folding": fixture["n_folding"],
        "limitation": fixture["limitation"],
        "compiles_made_this_run": compiles_made,
        "teacher_self_agreement": teacher_ceiling,
        "arms": summary_arms,
    }

    for a in summary_arms:
        r = a["vs_teacher_label_1"]["rates_pct"]
        print(f"\n[{a['arm']}] {a['description']}")
        print(f"   full agreement (all 3 exact):     {r['full_exact']:.1f}%")
        print(f"   full (urgency within 1):          {r['full_urgency_within_1']:.1f}%")
        print(f"   priority {r['priority']:.1f}%  department {r['department']:.1f}%  "
              f"urgency exact {r['urgency_exact']:.1f}%  urgency +/-1 {r['urgency_within_1']:.1f}%")
        print(f"   parse failures: {a['parse_failures']}/{a['vs_teacher_label_1']['n']}")
        print(f"   compile wall: {a['compile_wall_s']}s")
    tr = teacher_ceiling["rates_pct"]
    print(f"\n[teacher ceiling] full exact {tr['full_exact']:.1f}%  "
          f"full+/-1 {tr['full_urgency_within_1']:.1f}%  "
          f"priority {tr['priority']:.1f}%  department {tr['department']:.1f}%  "
          f"urgency exact {tr['urgency_exact']:.1f}%")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"finetune-triage-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")

    write_suite(Path(args.suite_out), evaluation)
    return 0


def judge_compare_report(report_path: Path, out_path: Path, model: str = "claude-haiku-4-5") -> int:
    """Judge a `paw-test compare --json` report using paw-kit's own `judge_outputs`.

    This exists only because `paw-test judge` cannot run at all against the installed
    `anthropic` SDK: `paw_kit/test/judge.py:277` passes `temperature=temperature` to
    `client.messages.create`, and `anthropic==1.4.0` removed `temperature` from that
    method's signature, so every judge call raises
    `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`.
    `judge_outputs` catches per-case exceptions, so the CLI does not crash -- it
    completes with `errored == total_cases` and zero usable verdicts.

    Reproduce (no API call is made; the TypeError precedes the request):
        python -c "from paw_kit.test.judge import anthropic_judge; anthropic_judge()('hi')"

    Everything else here -- the prompt, verdict parsing, the A/B difference block, the
    judge-vs-assertion disagreement block, the report shape -- is paw-kit's own code,
    called exactly the way `paw_kit/cli.py`'s `judge_cmd` calls it. Only the judge
    *callable* is substituted, passing temperature through `extra_body` so the judge is
    still pinned to 0 as the tool intends. This is a measurement workaround, not a fix
    to the tool.
    """
    from paw_kit.test.judge import JudgeInputRow, judge_disagreements, judge_outputs

    client = anthropic.Anthropic()

    def _judge(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"temperature": 0.0},
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    data = json.loads(report_path.read_text())
    if "rows" not in data or "adapter_a" not in data:
        print(f"{report_path} is not a compare report", file=sys.stderr)
        return 2

    rows_a = [JudgeInputRow(input=r["input"], output=r["output_a"], rule_passed=r.get("pass_a"))
              for r in data["rows"]]
    rows_b = [JudgeInputRow(input=r["input"], output=r["output_b"], rule_passed=r.get("pass_b"))
              for r in data["rows"]]
    note = ("temperature=0.0 via extra_body (anthropic 1.4.0 removed the named argument; "
            "paw_kit/test/judge.py:277 still passes it and therefore cannot run)")
    judge_id = f"anthropic/{model}/temperature=0.0"
    rep_a = judge_outputs(rows_a, _judge, spec=SPEC, temperature_note=note, judge_id=f"{judge_id}/A")
    rep_b = judge_outputs(rows_b, _judge, spec=SPEC, temperature_note=note, judge_id=f"{judge_id}/B")

    diffs = [(va, vb) for va, vb in zip(rep_a.verdicts, rep_b.verdicts) if va.verdict != vb.verdict]
    print(f"\nVerdict differs between A and B ({len(diffs)}/{rep_a.total_cases}):")
    for va, vb in diffs:
        print(f"  Input: {va.input[:100]}")
        print(f"    A: {va.output[:100]} -> {va.verdict} ({va.reason})")
        print(f"    B: {vb.output[:100]} -> {vb.verdict} ({vb.reason})")
    for side, rep in (("A", rep_a), ("B", rep_b)):
        dis = judge_disagreements(rep)
        print(f"\njudge disagrees with assertions ({side}) ({len(dis)}/{rep.total_cases}):")
        for v in dis:
            print(f"  {v.case_id}: rule_passed={v.rule_passed} judge={v.verdict} ({v.reason})")
    print(f"\nSummary: A pass {rep_a.pass_rate:.1f}% ({rep_a.pass_count}/{rep_a.total_cases}), "
          f"unparseable {rep_a.unparseable_count}, errored {rep_a.error_count}; "
          f"B pass {rep_b.pass_rate:.1f}% ({rep_b.pass_count}/{rep_b.total_cases}), "
          f"unparseable {rep_b.unparseable_count}, errored {rep_b.error_count}")
    out_path.write_text(json.dumps(
        {"adapter_a": rep_a.model_dump(), "adapter_b": rep_b.model_dump()}, indent=2))
    print(f"wrote {out_path}")
    return 0


def write_suite(path: Path, evaluation: List[Dict[str, Any]]) -> None:
    """Emit a suite.yaml for `paw-test compare` / `paw-test judge` over the 60
    evaluation tickets, with the first teacher label as `expected`."""
    import yaml

    cases = []
    for c in evaluation:
        lab = c.get("teacher_label_1")
        if not lab:
            continue
        cases.append({"input": c["ticket"], "expected": json.dumps(lab, separators=(", ", ": "))})
    suite = {
        "task_name": "finetune_triage",
        "spec": SPEC,
        "adapter_path": "measurements/finetune_triage_B-paw-4b-qwen3-0.6b.paw",
        "standard_cases": cases,
        # Structural only: `paw-test`'s assertion vocabulary is regex_match /
        # max_length / min_length / exact_match / not_contains (paw_kit/test/suite.py:23).
        # There is no rule that can check "priority is one of four values" or "this parses
        # as JSON matching a Pydantic model", so the semantic scoring stays in this
        # script and the suite checks shape.
        "assertions": [
            {"rule": "regex_match",
             "pattern": r'^\s*\{.*"priority".*"department".*"urgency_score".*\}\s*$'},
            {"rule": "regex_match",
             "pattern": r'"priority"\s*:\s*"(low|medium|high|critical)"'},
            {"rule": "regex_match",
             "pattern": r'"department"\s*:\s*"(billing|technical|sales|general)"'},
            {"rule": "regex_match", "pattern": r'"urgency_score"\s*:\s*[1-5]\b'},
            {"rule": "max_length", "value": 400},
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


if __name__ == "__main__":
    sys.exit(main())
