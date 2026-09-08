"""Semantic-correctness measurement: does the compiled adapter's output actually mean
the right thing, not just have the right shape?

Every prior real test in this repo (measure_real_backend.py, measure_schema_real_model.py,
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
    print("pip install anthropic", file=sys.stderr)
    sys.exit(2)

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
    adapter_path = str(out_dir / f"{suite_dict['task_name']}-{args.compiler}.paw")
    suite_dict["adapter_path"] = adapter_path
    suite_dict.setdefault("active_learning", {})["teacher_model"] = JUDGE_MODEL

    tmp_suite_path = out_dir / f"_tmp_{suite_dict['task_name']}_suite.yaml"
    tmp_suite_path.write_text(yaml.dump(suite_dict))
    suite = load_suite(str(tmp_suite_path))
    tmp_suite_path.unlink()

    backend = ProgramAsWeightsBackend(compiler=args.compiler, max_spec_examples=args.max_spec_examples)
    if not backend.is_available():
        print("programasweights SDK not importable", file=sys.stderr)
        return 2

    print(f"[compile] task={suite.task_name!r} spec={suite.spec!r} max_spec_examples={args.max_spec_examples}")
    t0 = time.perf_counter()
    backend.compile(suite.spec, [], adapter_path)  # no examples folded in -- terse-spec test
    compile_s = time.perf_counter() - t0
    print(f"[compile] done in {compile_s:.1f}s -> {adapter_path}")

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

    summary = {
        "label": args.label,
        "task_name": suite.task_name,
        "spec": suite.spec,
        "max_spec_examples": args.max_spec_examples,
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
