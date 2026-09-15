"""Semantic-correctness measurement: does the compiled adapter's output actually mean
the right thing, not just have the right shape?

# This script's judging logic is now also a shipped command: see paw_kit.test.judge / `paw-test judge`.

Every prior real test in this repo (measure_real_backend.py, measure_constrained_decoding.py,
measure_jit_speedup.py) checks *structural* validity: does the output match a regex, parse
as JSON, hit a length bound. None of them ask an independent judge whether the output is
actually a correct answer to the input under the spec. This script does that: it runs a
suite's standard_cases + fuzzed adversarial_probes through a freshly-compiled real adapter,
then asks Claude (as an independent judge, shown only the spec + input + adapter output --
never the suite's own `expected` field) whether the output is semantically correct.

This also runs suites with a deliberately terse, no-embedded-examples spec (the docs'
front-page style, not this toolkit's usual long carefully-scoped specs) with
max_spec_examples=0, to test what an under-specified spec actually produces once compiled
-- not what a longer spec would have produced.

Prerequisites:
    export PAW_API_KEY=paw_sk_...
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    uv run python scripts/measure_semantic_correctness.py measurements/spec-drafts/spec-1-json-repair.yaml \
        --compiler paw-4b-qwen3-0.6b --label 3080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from paw_kit import ProgramAsWeightsBackend, TestRunner, load_suite
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.runner import evaluate_assertion

try:
    import anthropic
except ImportError:
    # Deferred, not exit(2) here: this runs at import time, before this module's pure
    # functions (split_fold_and_eval, ...) are even defined.
    # tests/test_measurement_scripts.py imports this module by path specifically to
    # exercise those offline, with no anthropic installed and no network -- exiting here
    # made that impossible, so every test in that file failed importing the module rather
    # than testing anything (found 2026-09-11 reproducing a red CI run outside any dev
    # .venv that happens to have anthropic already installed).
    anthropic = None  # type: ignore[assignment]

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_PROMPT = """You are auditing a small compiled model's output for semantic correctness.

The model was compiled from this one-line specification, exactly as written (no other
instructions or examples were given to it beyond this sentence):

    SPEC: {spec}

Given this input:
    INPUT: {input!r}

The model produced this output:
    OUTPUT: {output!r}

Judge ONLY whether OUTPUT is a semantically correct, reasonable answer to INPUT under SPEC
-- ignore minor formatting differences (e.g. quote style, whitespace) unless the spec or
obvious intent requires an exact format. If the input is empty, garbage, or ambiguous under
the spec, judge whether OUTPUT is a *reasonable* thing to do with it (a sensible refusal or
best-effort answer both count as correct; silently returning something misleading does not).

Respond with EXACTLY one line: "YES: <reason, <15 words>" or "NO: <reason, <15 words>"."""


# --------------------------------------------------------------------- B-6: the fold split
#
# (a) `examples` used to be built from the whole of `suite.standard_cases`, and `TestRunner`
#     then scored those same cases -- 8 of 134 eval cases were memorisable from the adapter's
#     own prompt. (b) The adapter was named `{task}-{compiler}.paw` with no
#     `max_spec_examples` in it, so the `0` and `8` runs overwrote each other. That is how
#     the constrained-upstream section came to be measured against an 8-example adapter
#     while its narrative describes the terse-spec one: the committed
#     `phone_extractor-paw-4b-qwen3-0.6b.paw` has `examples_folded_into_spec: 8`.
#
# The folding pool is now drawn from the tail of `standard_cases`, those cases are removed
# from what gets scored, their ids are recorded in the artifact, and `main()` refuses to
# continue if any of them turns up in the scored results anyway.
#
# Note the report's suggested `few_shot_cases:` suite key is deliberately NOT the fix here:
# that is library code (`paw_kit/test/suite.py`) and out of scope for this track.


def split_fold_and_eval(cases: list, max_spec_examples: int) -> tuple[list, list]:
    """Partition raw `standard_cases` dicts into (folded, scored). Pure.

    The folding pool is the **tail** of the eligible cases -- eligible meaning they carry
    an `expected`, since a case with no expected output cannot be a few-shot example. The
    tail rather than the head because the backend folds `examples[:limit]`: passing exactly
    the cases to be folded makes `folded_case_ids` an exact record of what went into the
    spec rather than a prediction about how the backend will truncate.

    `max_spec_examples <= 0` folds nothing and scores everything, which is the
    2026-09-09 terse-spec configuration and stays byte-identical in behaviour.
    """
    if max_spec_examples <= 0:
        return [], list(cases)
    eligible = [i for i, c in enumerate(cases) if c.get("expected")]
    fold_idx = set(eligible[-max_spec_examples:])
    folded = [cases[i] for i in sorted(fold_idx)]
    scored = [c for i, c in enumerate(cases) if i not in fold_idx]
    return folded, scored


def fold_case_ids(cases: list, folded: list) -> list:
    """Stable ids for the folded cases: their index in the original `standard_cases`."""
    by_id = {id(c): i for i, c in enumerate(cases)}
    return [f"standard_cases[{by_id[id(c)]}]" for c in folded]


def adapter_filename(task_name: str, compiler: str, max_spec_examples: int) -> str:
    """B-6(b): `max_spec_examples` belongs in the path.

    Without it a 0-example run and an 8-example run of the same task and compiler write to
    the same file, the second silently replacing the first, and any later script pointed at
    that path measures whichever ran last.
    """
    return f"{task_name}-{compiler}-fold{max_spec_examples}.paw"


def judge(client: "anthropic.Anthropic", spec: str, inp: str, output: str) -> tuple[bool, str]:
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=60,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(spec=spec, input=inp, output=output)}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    verdict = text.upper().startswith("YES")
    reason = text.split(":", 1)[1].strip() if ":" in text else text
    return verdict, reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="path to a suite.yaml")
    ap.add_argument("--compiler", default="paw-4b-qwen3-0.6b")
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--max-spec-examples", type=int, default=0,
                     help="0 = pure terse spec, no folded examples (default: match docs' own style)")
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    if anthropic is None:
        print("pip install anthropic", file=sys.stderr)
        return 2

    if not os.environ.get("PAW_API_KEY"):
        print("PAW_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    raw = json.loads(json.dumps({}))  # noop, keeps structure obvious below
    suite_dict = None
    import yaml
    with open(args.suite) as f:
        suite_dict = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(out_dir / adapter_filename(
        suite_dict["task_name"], args.compiler, args.max_spec_examples))
    suite_dict["adapter_path"] = adapter_path
    suite_dict.setdefault("active_learning", {})["teacher_model"] = JUDGE_MODEL

    # B-6(a): hold the folding pool out of the suite that gets scored, before the suite
    # object is built, so the runner never sees those cases at all.
    all_standard_cases = list(suite_dict.get("standard_cases") or [])
    folded_cases, scored_cases = split_fold_and_eval(
        all_standard_cases, args.max_spec_examples)
    folded_case_ids = fold_case_ids(all_standard_cases, folded_cases)
    folded_inputs = {c["input"] for c in folded_cases}
    suite_dict["standard_cases"] = scored_cases
    print(f"[fold] standard_cases={len(all_standard_cases)} "
          f"folded={len(folded_cases)} scored={len(scored_cases)} "
          f"folded_case_ids={folded_case_ids}")
    if folded_inputs & {c["input"] for c in scored_cases}:
        raise RuntimeError(
            "a folded case is still in the scored set, which is the B-6(a) leak: "
            f"{sorted(folded_inputs & {c['input'] for c in scored_cases})}"
        )

    tmp_suite_path = out_dir / f"_tmp_{suite_dict['task_name']}_suite.yaml"
    tmp_suite_path.write_text(yaml.dump(suite_dict))
    suite = load_suite(str(tmp_suite_path))
    tmp_suite_path.unlink()

    backend = ProgramAsWeightsBackend(compiler=args.compiler, max_spec_examples=args.max_spec_examples)
    if not backend.is_available():
        print("programasweights SDK not importable", file=sys.stderr)
        return 2

    # BUG, found 2026-09-09 while reading a suspiciously-unchanged A/B result: this used
    # to hardcode `examples=[]` here regardless of --max-spec-examples, so every
    # "fewshot" A/B run silently compiled with zero examples folded in -- identical to
    # the terse-spec baseline it was supposed to be compared against. --max-spec-examples
    # only configured the backend's *cap*; with no examples ever passed in, there was
    # nothing for that cap to apply to. Fixed: build real examples from the suite's own
    # standard_cases.
    #
    # B-6(a), 2026-09-11: and those examples now come from the held-out tail only, so no
    # case the adapter was shown is also scored.
    examples = [{"input": c["input"], "output": c["expected"]} for c in folded_cases]
    examples_used = min(len(examples), args.max_spec_examples)
    print(f"[compile] task={suite.task_name!r} spec={suite.spec!r} "
          f"max_spec_examples={args.max_spec_examples} examples_available={len(examples)} "
          f"examples_folded_in={examples_used}")
    t0 = time.perf_counter()
    backend.compile(suite.spec, examples, adapter_path)
    compile_s = time.perf_counter() - t0
    manifest = json.loads(Path(adapter_path).read_text())
    print(f"[compile] done in {compile_s:.1f}s -> {adapter_path} "
          f"(program {manifest.get('program_id')}, "
          f"folded={manifest.get('examples_folded_into_spec')}, "
          f"public_requested={manifest.get('public_requested', manifest.get('public'))}, "
          f"public_confirmed={manifest.get('public_confirmed')})")

    # structural pass/fail via the real TestRunner (standard cases + fuzzer)
    runner = TestRunner(backend=backend)
    report = runner.run(suite)
    print(f"[structural] {report.passed_cases}/{report.total_cases} passed ({report.pass_rate:.0f}%)")

    client = anthropic.Anthropic()
    rows = []
    yes_count = 0
    divergences = []
    for r in report.results:
        sem_ok, reason = judge(client, suite.spec, r.input, r.output)
        yes_count += int(sem_ok)
        structurally_ok = r.passed
        row = {
            "input": r.input,
            "output": r.output,
            "structural_pass": structurally_ok,
            "semantic_pass": sem_ok,
            "judge_reason": reason,
            "failed_rules": r.failed_rules,
        }
        rows.append(row)
        tag = ""
        if structurally_ok and not sem_ok:
            tag = " <-- passed structurally, WRONG semantically"
            divergences.append(row)
        elif sem_ok and not structurally_ok:
            tag = " <-- failed structurally, semantically fine"
            divergences.append(row)
        print(f"  [{('OK' if sem_ok else 'BAD'):3s} sem | {('OK' if structurally_ok else 'BAD'):3s} struct] "
              f"{r.input[:40]!r:42s} -> {r.output[:40]!r:42s} ({reason}){tag}")

    leaked = [r for r in rows if r["input"] in folded_inputs]
    if leaked:  # pragma: no cover -- held out above; asserted so it cannot come back
        raise RuntimeError(
            f"{len(leaked)} folded case(s) appear in the scored results: "
            f"{[r['input'] for r in leaked]}"
        )

    summary = {
        "label": args.label,
        "task_name": suite.task_name,
        "spec": suite.spec,
        "max_spec_examples": args.max_spec_examples,
        "adapter_path": adapter_path,
        # B-6(b): the adapter's identity, mirrored into every artifact, so which adapter
        # produced which table is recoverable without the gitignored .paw file.
        "program_id": manifest.get("program_id"),
        # A-6: `examples_folded_into_spec` is now the count of examples that actually
        # reached the spec text, not the number offered to compile() (a malformed example
        # used to inflate it, and this artifact is where that inflated number was
        # published). `folded_example_ids` is mirrored alongside it so the artifact names
        # *which* examples those were, not just how many -- the ids are SHA-256 digests of
        # input+output, so this publishes no traced text.
        "examples_folded_into_spec": manifest.get("examples_folded_into_spec"),
        "folded_example_ids": manifest.get("folded_example_ids"),
        # A-2: the manifest key that records the *requested* visibility was renamed
        # `public` -> `public_requested`, and `public_confirmed` (three-state: True /
        # False / None-with-a-reason) now carries what the server actually reported.
        # Read the new name with a legacy fallback so a summary built from a manifest
        # written before the rename still reports the value instead of silently None.
        "public_requested": manifest.get("public_requested", manifest.get("public")),
        "public_confirmed": manifest.get("public_confirmed"),
        "public_confirmed_reason": manifest.get("public_confirmed_reason"),
        "spec_sha256": manifest.get("spec_sha256"),
        "full_spec_sha256": manifest.get("full_spec_sha256"),
        "compiler": args.compiler,
        "compile_wall_s": compile_s,
        # B-6(a): which cases were held out of the scored denominator, and the denominator.
        "standard_cases_total": len(all_standard_cases),
        "standard_cases_scored": len(scored_cases),
        "folded_case_ids": folded_case_ids,
        "folded_inputs": sorted(folded_inputs),
        "scored_rows_folded_into_spec": len(leaked),
        "judge_model": JUDGE_MODEL,
        "total_cases": report.total_cases,
        "structural_pass_rate": report.pass_rate,
        "semantic_pass_rate": (yes_count / report.total_cases * 100.0) if report.total_cases else None,
        "divergence_count": len(divergences),
        "divergences": divergences,
        "all_cases": rows,
    }
    print(f"\nstructural pass rate: {report.pass_rate:.0f}%")
    print(f"semantic pass rate:   {summary['semantic_pass_rate']:.0f}%")
    print(f"divergences (structural and semantic disagree): {len(divergences)}/{report.total_cases}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"semantic-{suite.task_name}-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
