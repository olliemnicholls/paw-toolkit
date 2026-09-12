"""Is the finetune compiler's advantage about *computation*, or about *spec-defined
mappings the base model does not have*?

`measurements/README.md` ends on the fiscal-week section: on a rule the base model does
not know, the finetune compiler `paw-ft-bs48` scored 49.0% exact against the fast
compiler `paw-4b-qwen3-0.6b`'s 11.3% (no examples) and 10.3% (8 folded examples), and the
fast compiler was shown to emit a small fixed vocabulary of labels regardless of input --
arm B answered `FY2024-W43`, one of the folded examples' own answers, 129 times out of
300. That located a gap, but on a task whose content is multi-step arithmetic. Two
readings survive it:

  (i) the gap is about *computation* -- the fast compiler's single forward pass cannot
      produce an adapter that computes, and would do fine on a rule with no arithmetic;
 (ii) the gap is about *spec-defined mappings the base model lacks* -- anything stated in
      the spec and absent from pretraining, arithmetic or not.

This script separates them with a task that has no arithmetic at all: an arbitrary
region-code lookup table. Thirty real country names map to six made-up internal codes
(`RG-K7`, `RG-M2`, `RG-Q9`, `RG-T4`, `RG-V1`, `RG-X6`). The assignment is a seeded
shuffle, so the grouping is not geographic, not alphabetical and not anything a
pretrained model could guess -- but every answer is a single table lookup: find the
country named in the sentence, read off its code. No dates, no counting, no division.

If arms A and B do well here, the fiscal-week gap was about computation (reading i). If
they collapse onto a fixed vocabulary of codes the way they collapsed onto W43, the
finding generalises to reading (ii).

Four arms. Three compiles, one API-only reference:

    A  fast compiler `paw-4b-qwen3-0.6b`, max_spec_examples=0
    B  fast compiler, the 8 folding examples folded into the spec text
    C  finetune compiler `paw-ft-bs48`, the same 8 examples, via compile_async polling
    D  `claude-haiku-4-5-20251001` zero-shot on the same spec, temperature 0, no compile

D is a reference for how hard the task is, not a ceiling: ground truth is exact, so 100%
is the ceiling. Unlike fiscal weeks, D's `max_tokens` can be tiny here -- the answer is
six characters -- so the truncation failure that depressed D's fiscal score cannot recur.

Evaluation set: 300 sentences, fixed seed. Every country appears exactly 10 times, once
in each of ten templates, so each template carries 30 cases and the per-country and
per-template cuts are balanced by construction. Some templates mention a city in a
*different* country than the one to be looked up; one writes the country name in lower
case. The country name always appears verbatim, so no case requires resolving a demonym
or an abbreviation -- the difficulty is the table, not the parsing.

Folding pool: 8 sentences covering 8 different countries and all 6 codes, built from four
phrasings that appear nowhere in the evaluation set, so the two sets are disjoint by
construction rather than by luck. The eight folded *countries* do appear in the
evaluation set (10 times each), which is what makes "does an arm only get the folded
countries right?" answerable.

Prerequisites:
    PAW_API_KEY, ANTHROPIC_API_KEY in the environment.

Usage:
    uv run python scripts/measure_finetune_lookup.py --label 3080 --no-run --arms A
    uv run python scripts/measure_finetune_lookup.py --label 3080 --skip-compile --arms A,B,C,D
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paw_kit.backend.programasweights import ProgramAsWeightsBackend  # noqa: E402

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

REFERENCE_MODEL = "claude-haiku-4-5-20251001"
REFERENCE_TEMPERATURE = 0.0
# The answer is six characters. 30 tokens is plenty, and it makes the fiscal section's
# "arm D reasoned past its token budget on 39 of 300 cases" failure structurally
# impossible here -- there is nothing to reason about.
REFERENCE_MAX_TOKENS = 30

FAST_COMPILER = "paw-4b-qwen3-0.6b"
FINETUNE_COMPILER = "paw-ft-bs48"

N_FOLD = 8
N_EVAL = 300
SEED = 20260910

CODES = ["RG-K7", "RG-M2", "RG-Q9", "RG-T4", "RG-V1", "RG-X6"]

# Thirty real countries, each with one real city used by the city-bearing templates.
# ASCII only: a diacritic in the input would be a second variable, and the question here
# is the table, not tokenisation.
COUNTRY_CITY: List[tuple] = [
    ("Japan", "Osaka"),
    ("Kenya", "Nairobi"),
    ("Portugal", "Porto"),
    ("New Zealand", "Auckland"),
    ("Brazil", "Rio de Janeiro"),
    ("Canada", "Montreal"),
    ("Norway", "Bergen"),
    ("Vietnam", "Da Nang"),
    ("Morocco", "Casablanca"),
    ("Poland", "Krakow"),
    ("Chile", "Valparaiso"),
    ("Ireland", "Cork"),
    ("Thailand", "Chiang Mai"),
    ("Egypt", "Alexandria"),
    ("Australia", "Brisbane"),
    ("Peru", "Arequipa"),
    ("Finland", "Tampere"),
    ("Nigeria", "Lagos"),
    ("Turkey", "Izmir"),
    ("Mexico", "Guadalajara"),
    ("Sweden", "Gothenburg"),
    ("India", "Pune"),
    ("Colombia", "Medellin"),
    ("Greece", "Thessaloniki"),
    ("Indonesia", "Surabaya"),
    ("Denmark", "Aarhus"),
    ("Argentina", "Rosario"),
    ("Philippines", "Cebu"),
    ("Hungary", "Debrecen"),
    ("Malaysia", "Penang"),
]

COUNTRIES = [c for c, _ in COUNTRY_CITY]
CITY_OF = dict(COUNTRY_CITY)

# Ten evaluation templates. `country` is always substituted verbatim except in
# `lowercase`, which lower-cases it. `other_city` is a city belonging to a *different*
# country, chosen deterministically -- those are the cases where geographic association
# pulls the wrong way.
TEMPLATES: List[Dict[str, Any]] = [
    {"id": "ship_city", "needs": ["city"],
     "text": "Ship this order to {city}, {country}."},
    {"id": "billing", "needs": [],
     "text": "Customer billing address is in {country}."},
    {"id": "invoice_office", "needs": ["city"],
     "text": "Invoice for the {city} office ({country}) attached."},
    {"id": "reseller", "needs": [],
     "text": "Our reseller in {country} needs the report."},
    {"id": "warehouse", "needs": [],
     "text": "Warehouse transfer: destination {country}."},
    {"id": "misleading_office", "needs": ["other_city"], "misleading": True,
     "text": "Our {other_city} office handled the call, but the customer is in {country}."},
    {"id": "lowercase", "needs": [], "lowercase": True,
     "text": "please route the shipment to {country}."},
    {"id": "support", "needs": [],
     "text": "Support ticket opened by a customer in {country}; escalate to the regional desk."},
    {"id": "misleading_route", "needs": ["other_city"], "misleading": True,
     "text": "Routed via {other_city}, final delivery {country}."},
    {"id": "depot", "needs": [],
     "text": "We are opening a second depot in {country} next quarter."},
]

# Four folding-only phrasings. None of these strings can collide with an evaluation case:
# the evaluation set uses the ten templates above and nothing else, so the folding pool is
# disjoint by construction. The trade-off is stated in the README -- the folded examples
# demonstrate the mapping in a phrasing the evaluation set never repeats, which tests
# transfer of the *table* rather than memorisation of a sentence.
FOLD_TEMPLATES: List[Dict[str, Any]] = [
    {"id": "fold_confirm", "needs": [],
     "text": "Please confirm the regional code for a shipment going to {country}."},
    {"id": "fold_registered", "needs": [],
     "text": "The account is registered in {country}."},
    {"id": "fold_dispatch", "needs": ["city"],
     "text": "Dispatch note: {city}, {country}."},
    {"id": "fold_partner", "needs": [],
     "text": "Our partner office in {country} raised the request."},
]

CODE_RE = re.compile(r"RG-[A-Z]\d")
STRICT_CODE_RE = re.compile(r"^RG-[A-Z]\d$")


# ------------------------------------------------------------------- the table

def build_table() -> Dict[str, str]:
    """Assign the 30 countries to the 6 codes with a fixed seed, five each.

    The shuffle is the point: the grouping must not be geographic, alphabetical, or
    anything else a pretrained model could infer without reading the table. Deterministic
    given SEED, so the table below is reproducible from this function alone.
    """
    rng = random.Random(SEED)
    shuffled = list(COUNTRIES)
    rng.shuffle(shuffled)
    table: Dict[str, str] = {}
    per_code = len(shuffled) // len(CODES)
    for i, country in enumerate(shuffled):
        table[country] = CODES[i // per_code]
    return table


TABLE = build_table()


def region_code(country: str) -> str:
    """Ground truth: a dict lookup. No model, no judge, no arithmetic."""
    return TABLE[country]


# ------------------------------------------------------------------- spec

def build_spec(table: Dict[str, str]) -> str:
    """The spec text every arm sees, byte-for-byte identical.

    The table goes in as a plain list, one country per line, grouped under its code --
    the most ordinary way anyone would write this down. Arms B and C additionally get the
    8 folding examples appended by the backend's own `_render_spec_with_examples`.
    """
    lines = [
        "Assign an internal region code to a sales message.",
        "",
        "Every country we operate in belongs to exactly one internal region. The regions "
        "are an internal grouping: they are not continents, not trading blocs, and not "
        "anything you can work out from geography. The only way to get the answer right "
        "is to look the country up in the table below.",
        "",
        "The table:",
        "",
    ]
    for code in CODES:
        members = [c for c in COUNTRIES if table[c] == code]
        lines.append(f"{code}")
        for m in members:
            lines.append(f"  {m}")
        lines.append("")
    lines += [
        "The input is a single short sentence from our order and support systems. It "
        "mentions exactly one of the thirty countries above, written out in full. It may "
        "also mention a city; the city is irrelevant, and it may well be a city in some "
        "other country. Match on the country name only. The country name may appear in "
        "lower case.",
        "",
        "Output exactly one region code and nothing else: one of RG-K7, RG-M2, RG-Q9, "
        "RG-T4, RG-V1, RG-X6. No explanation, no punctuation, no country name.",
        "",
        "Worked example one. The input is \"Ship this order to "
        f"{CITY_OF['Portugal']}, Portugal.\" The country is Portugal. "
        f"Portugal is listed under {table['Portugal']}. The output is {table['Portugal']}.",
        "",
        "Worked example two. The input is \"Our Osaka office handled the call, but the "
        "customer is in Kenya.\" Two places are mentioned, but Osaka is a city and Kenya "
        f"is the country. Kenya is listed under {table['Kenya']}. The output is "
        f"{table['Kenya']}.",
        "",
    ]
    return "\n".join(lines)


SPEC = build_spec(TABLE)

# ------------------------------------------------------- B-8a: spec/eval overlap
#
# The spec's worked example one renders byte-identical to evaluation case `r168`
# ("Ship this order to Porto, Portugal."), answer stated, so that one case is in every
# arm's prompt. `build_fixture`'s existing overlap assertion checks folding-vs-eval only.
#
# This is deliberately NOT folded into that RuntimeError. The overlap exists *now*, so
# raising on it would raise on every invocation, and the only ways to clear it are changing
# `SPEC` -- which changes the upstream compile-cache key and invalidates the committed
# 33.0/29.0/97.7/100% lookup table, requiring three paid recompiles -- or changing the
# evaluation template. So the overlap is computed, recorded, and asserted to be exactly the
# known one: it may not grow.
KNOWN_SPEC_EVAL_OVERLAP = ["r168"]


def spec_eval_overlap(spec: str, evaluation: List[Dict[str, Any]]) -> List[str]:
    """Ids of evaluation cases whose input appears verbatim inside `spec`. Pure."""
    return sorted(c["id"] for c in evaluation if c["input"] in spec)


# ------------------------------------------------------------------- fixture

def _render(template: Dict[str, Any], country: str, other_city: str) -> str:
    name = country.lower() if template.get("lowercase") else country
    return template["text"].format(
        country=name, city=CITY_OF[country], other_city=other_city)


def build_fixture(out_path: Path) -> Dict[str, Any]:
    """300 evaluation sentences and 8 folding sentences, deterministically.

    Every country gets each of the ten templates exactly once -- 30 countries x 10
    templates = 300 -- so the per-country cut is 10 cases each and the per-template cut is
    30 cases each, with no sampling noise in either. The order is then shuffled so no arm
    sees the countries in table order.
    """
    rng = random.Random(SEED + 1)

    evaluation: List[Dict[str, Any]] = []
    for country in COUNTRIES:
        for template in TEMPLATES:
            other_city = ""
            if "other_city" in template["needs"]:
                # A city from a different country, drawn deterministically. This is the
                # misleading half of the set: the sentence names a place associated with
                # some other region code.
                pool = [c for c in COUNTRIES if c != country]
                other_city = CITY_OF[rng.choice(pool)]
            evaluation.append({
                "country": country,
                "template": template["id"],
                "misleading_city": bool(template.get("misleading")),
                "lowercase": bool(template.get("lowercase")),
                "input": _render(template, country, other_city),
                "expected": region_code(country),
            })
    rng.shuffle(evaluation)
    for i, case in enumerate(evaluation):
        case["id"] = f"r{i:03d}"

    # Folding pool: 8 countries covering all 6 codes -- one per code, plus two extras.
    fold_countries: List[str] = []
    for code in CODES:
        members = sorted(c for c in COUNTRIES if TABLE[c] == code)
        fold_countries.append(rng.choice(members))
    remaining = sorted(c for c in COUNTRIES if c not in fold_countries)
    fold_countries += rng.sample(remaining, N_FOLD - len(fold_countries))

    folding = []
    for i, country in enumerate(fold_countries):
        template = FOLD_TEMPLATES[i % len(FOLD_TEMPLATES)]
        folding.append({
            "country": country,
            "template": template["id"],
            "input": _render(template, country, ""),
            "expected": region_code(country),
        })

    eval_inputs = {c["input"] for c in evaluation}
    overlap = [f["input"] for f in folding if f["input"] in eval_inputs]
    if overlap:  # pragma: no cover -- structurally impossible, asserted anyway
        raise RuntimeError(f"folding pool overlaps the evaluation set: {overlap}")

    # B-8a: a *reported* overlap, not an absence assertion. Raise only if it grows.
    spec_overlap = spec_eval_overlap(SPEC, evaluation)
    if spec_overlap != KNOWN_SPEC_EVAL_OVERLAP:
        raise RuntimeError(
            f"spec/eval overlap changed: {spec_overlap} != {KNOWN_SPEC_EVAL_OVERLAP}. "
            "The spec's worked examples render eval case(s) verbatim with the answer "
            "stated, so every arm has them in its prompt. If this grew, either the "
            "templates or the worked examples changed; fix that rather than widening the "
            "known set. Report B-8a."
        )

    fixture = {
        "task": "region_code_lookup",
        "spec": SPEC,
        "rule": (
            "Thirty countries are assigned to six made-up region codes by a seeded "
            "shuffle. The answer for a sentence is the code of the one country it names. "
            "Ground truth is a dict lookup -- no arithmetic anywhere in the task."
        ),
        "ground_truth": "computed by scripts/measure_finetune_lookup.py:region_code -- no model, no judge",
        "seed": SEED,
        "codes": CODES,
        "table": TABLE,
        "table_by_code": {code: [c for c in COUNTRIES if TABLE[c] == code] for code in CODES},
        "countries": COUNTRIES,
        "cities": CITY_OF,
        "templates": [{"id": t["id"], "text": t["text"],
                       "misleading": bool(t.get("misleading")),
                       "lowercase": bool(t.get("lowercase"))} for t in TEMPLATES],
        "fold_templates": [{"id": t["id"], "text": t["text"]} for t in FOLD_TEMPLATES],
        "n_evaluation": len(evaluation),
        "n_folding": len(folding),
        "n_misleading": sum(1 for c in evaluation if c["misleading_city"]),
        # B-8a: which evaluation cases the spec itself answers, recorded rather than
        # assumed absent.
        "spec_eval_overlap_ids": spec_overlap,
        "known_spec_eval_overlap": list(KNOWN_SPEC_EVAL_OVERLAP),
        "spec_eval_overlap_note": (
            "The spec's worked example one renders these evaluation case(s) byte-identically "
            "with the answer stated, so they are in every arm's prompt. Scores are reported "
            "both over all 300 cases and excluding them (see "
            "scores.excluding_spec_leak, denominator 299)."
        ),
        "folding_countries": fold_countries,
        "folding_codes": sorted({f["expected"] for f in folding}),
        "folding_examples": folding,
        "evaluation": evaluation,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"[data] wrote {out_path} ({len(evaluation)} eval, {len(folding)} folding, "
          f"{fixture['n_misleading']} misleading-city)")
    return fixture


# ------------------------------------------------------------------- arms

def arm_specs(out_dir: Path) -> List[Dict[str, Any]]:
    return [
        {"arm": "A", "kind": "adapter", "compiler": FAST_COMPILER, "max_spec_examples": 0,
         "manifest": str(out_dir / "finetune_lookup_A-paw-4b-qwen3-0.6b.paw"),
         "description": "fast compiler, no folded examples"},
        {"arm": "B", "kind": "adapter", "compiler": FAST_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_lookup_B-paw-4b-qwen3-0.6b.paw"),
         "description": f"fast compiler, {N_FOLD} folded examples"},
        {"arm": "C", "kind": "adapter", "compiler": FINETUNE_COMPILER, "max_spec_examples": N_FOLD,
         "manifest": str(out_dir / "finetune_lookup_C-paw-ft-bs48.paw"),
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
            # Same guard as scripts/measure_finetune_fiscal.py and
            # scripts/measure_finetune_triage.py: `.paw` manifests are gitignored, so a
            # fresh worktree or a `git clean` makes them vanish while every committed
            # artifact still refers to them by program id. Silently compiling a
            # *different* adapter under a flag that promises not to compile is worse than
            # stopping.
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
    removed it from `Messages.create`'s typed signature, so passing it as a named argument
    raises `TypeError`. The wire API still honours the field. Same workaround as
    scripts/measure_finetune_fiscal.py.
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
        except Exception as exc:  # noqa: BLE001
            raw, err = "", f"{type(exc).__name__}: {exc}"
        rows.append(_row(case, raw, err, (time.perf_counter() - t0) * 1000.0))
        if (i + 1) % 50 == 0:
            print(f"  [{arm['arm']}] {i + 1}/{len(evaluation)}")
    return rows


def _row(case: Dict[str, Any], raw: str, err: Optional[str], latency_ms: float) -> Dict[str, Any]:
    """Score one output against ground truth.

    `strict_shape` is the whole output being exactly a code (what the spec asks for);
    `code_found` is a code appearing anywhere in the output. Exact match is judged on the
    first code found, so an arm that answers correctly inside prose is not penalised
    twice -- that shows up in the strict-shape column instead.
    """
    stripped = (raw or "").strip()
    match = CODE_RE.search(stripped)
    found = match.group(0) if match else None
    return {
        "id": case["id"],
        "country": case["country"],
        "template": case["template"],
        "misleading_city": case["misleading_city"],
        "lowercase": case["lowercase"],
        "input": case["input"],
        "expected": case["expected"],
        "raw": raw,
        "code": found,
        "error": err,
        "strict_shape": bool(STRICT_CODE_RE.match(stripped)),
        "code_found": found is not None,
        "in_vocabulary": found in CODES if found else False,
        "exact": found == case["expected"],
        "latency_ms": latency_ms,
    }


# ------------------------------------------------------------------- scoring

_METRICS = ["exact", "strict_shape", "code_found", "in_vocabulary"]


def _rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    counts = {m: sum(1 for r in rows if r[m]) for m in _METRICS}
    return {
        "n": n,
        "counts": counts,
        "rates_pct": {m: (counts[m] / n * 100.0 if n else 0.0) for m in _METRICS},
    }


def score_arm(rows: List[Dict[str, Any]], folding_countries: List[str],
              spec_eval_overlap_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    overlap_ids = set(spec_eval_overlap_ids or ())
    out: Dict[str, Any] = {"overall": _rates(rows)}
    # B-8a: the same metrics with the spec-answered case(s) removed from the denominator.
    # Reported alongside `overall` rather than replacing it, so both numbers are available
    # and the difference is visible.
    out["excluding_spec_leak"] = _rates([r for r in rows if r["id"] not in overlap_ids])
    out["spec_eval_overlap_ids"] = sorted(overlap_ids)
    out["by_template"] = {
        t["id"]: _rates([r for r in rows if r["template"] == t["id"]]) for t in TEMPLATES}
    out["by_country"] = {
        c: {"n": sum(1 for r in rows if r["country"] == c),
            "exact": sum(1 for r in rows if r["country"] == c and r["exact"]),
            "expected_code": TABLE[c],
            "in_folding": c in folding_countries}
        for c in COUNTRIES}
    out["by_misleading"] = {
        "misleading_city": _rates([r for r in rows if r["misleading_city"]]),
        "plain": _rates([r for r in rows if not r["misleading_city"]]),
    }
    fold_rows = [r for r in rows if r["country"] in folding_countries]
    out["folded_countries"] = _rates(fold_rows)
    out["unfolded_countries"] = _rates([r for r in rows if r["country"] not in folding_countries])

    # The collapse question: how many distinct things does this arm ever say, and how
    # concentrated is its output? This is the column that answered the fiscal-week
    # section ("43 distinct labels for arm A, 31 for arm B, 130 for arm C").
    hist: Dict[str, int] = {}
    for r in rows:
        hist[r["code"] or "<none>"] = hist.get(r["code"] or "<none>", 0) + 1
    ordered = dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))
    n = len(rows) or 1
    top = list(ordered.items())
    out["output_distribution"] = ordered
    out["distinct_outputs"] = len([k for k in ordered if k != "<none>"])
    out["top_output_share_pct"] = (top[0][1] / n * 100.0) if top else 0.0
    out["top_two_output_share_pct"] = (sum(v for _, v in top[:2]) / n * 100.0) if top else 0.0
    # A uniform arm would answer each code 50 times out of 300; the ground truth is
    # exactly 50 per code, since 5 countries x 10 sentences each.
    out["expected_per_code_if_uniform"] = n / len(CODES)
    # How often each arm's answer is the code the *right* country belongs to, versus the
    # code of some other country mentioned in the sentence -- there is no other country
    # named, so this cannot rescue a wrong answer; kept as a sanity field.
    out["countries_fully_correct"] = sorted(
        c for c in COUNTRIES
        if out["by_country"][c]["n"] and out["by_country"][c]["exact"] == out["by_country"][c]["n"])
    out["countries_zero_correct"] = sorted(
        c for c in COUNTRIES if out["by_country"][c]["exact"] == 0)
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
    """A `paw-test` suite over the 300 cases.

    `expected` carries the exact ground-truth code. Note that `paw-test check` does not
    read it -- see the README section; the assertions below are all `check` can score, and
    they only constrain the *shape* of the output. `paw-test compare` does surface
    per-case differences between two adapters, which is what it is used for here.
    """
    import yaml

    cases = [{"input": c["input"], "expected": c["expected"]} for c in evaluation]
    suite = {
        "task_name": "region_code",
        "spec": SPEC,
        "adapter_path": adapter_path,
        "standard_cases": cases,
        "assertions": [
            {"rule": "regex_match", "pattern": r"RG-[A-Z]\d"},
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
    ap.add_argument("--data-path", default="measurements/finetune-lookup-regions.json")
    ap.add_argument("--rebuild-data", action="store_true")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Reuse existing .paw manifests instead of compiling (no compile calls at all)")
    ap.add_argument("--arms", default="A,B,C,D", help="Comma-separated subset of arms to run")
    ap.add_argument("--suite-out", default="measurements/finetune-lookup-suite.yaml")
    ap.add_argument("--no-run", action="store_true",
                    help="Build data and compile only; skip inference")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    if data_path.exists() and not args.rebuild_data:
        fixture = json.loads(data_path.read_text())
        print(f"[data] reusing {data_path} ({fixture['n_evaluation']} eval, "
              f"{fixture['n_folding']} folding)")
    else:
        fixture = build_fixture(data_path)

    evaluation = fixture["evaluation"]
    examples = fold_examples(fixture)
    folding_countries = fixture["folding_countries"]

    # B-8a. Recomputed here rather than trusted from the fixture, so a fixture written
    # before this field existed -- the committed `finetune-lookup-regions.json` is one --
    # still gets a correct `excluding_spec_leak` denominator, and so that the assertion
    # holds for a reused fixture and not only for a freshly built one.
    spec_overlap_ids = spec_eval_overlap(SPEC, evaluation)
    if spec_overlap_ids != KNOWN_SPEC_EVAL_OVERLAP:
        raise RuntimeError(
            f"spec/eval overlap changed: {spec_overlap_ids} != {KNOWN_SPEC_EVAL_OVERLAP}. "
            "See report B-8a and build_fixture's note."
        )
    print(f"[spec-leak] evaluation cases the spec itself answers: {spec_overlap_ids}")

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
                    str(out_dir / "finetune_lookup_B-paw-4b-qwen3-0.6b.paw"))
        return 0

    summary_arms = []
    for arm in arms:
        print(f"[run:{arm['arm']}] {len(evaluation)} sentences on {arm['compiler']}")
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
            # A-6: `examples_folded_into_spec` is now the count of examples that actually
            # reached the spec text, not the number offered to compile() (a malformed example
            # used to inflate it, and this artifact is where that inflated number was
            # published). `folded_example_ids` is mirrored alongside it so the artifact names
            # *which* examples those were, not just how many -- the ids are SHA-256 digests of
            # input+output, so this publishes no traced text.
            "examples_folded_into_spec": md.get("examples_folded_into_spec"),
            "folded_example_ids": md.get("folded_example_ids"),
            "compile_wall_s": md.get("compile_wall_s"),
            # A-2: the manifest key that records the *requested* visibility was renamed
            # `public` -> `public_requested`, and `public_confirmed` (three-state: True /
            # False / None-with-a-reason) now carries what the server actually reported.
            # Read the new name with a legacy fallback so a summary built from a manifest
            # written before the rename still reports the value instead of silently None.
            "public_requested": md.get("public_requested", md.get("public")),
            "public_confirmed": md.get("public_confirmed"),
            "public_confirmed_reason": md.get("public_confirmed_reason"),
            "cache_hit": md.get("cache_hit"),
            "status": md.get("status"),
            "slug": md.get("slug"),
            "spec_sha256": md.get("spec_sha256"),
            "full_spec_sha256": md.get("full_spec_sha256"),
            "compiled_at": md.get("compiled_at"),
            "manifest_version": md.get("manifest_version"),
            "errors": sum(1 for r in rows if r["error"]),
            "latency_ms": _latency(rows),
            "scores": score_arm(rows, folding_countries,
                                spec_overlap_ids),
            "cases": rows,
        })

    summary = {
        "label": args.label,
        "task": "region_code_lookup",
        "spec": SPEC,
        "table": TABLE,
        "ground_truth": "computed by code (region_code); no judge, no teacher, exact",
        "reference_model": REFERENCE_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "reference_max_tokens": REFERENCE_MAX_TOKENS,
        "adapter_temperature": 0.0,
        "adapter_temperature_note": (
            "programasweights' PawFunction defaults to temperature=0.0 (greedy). "
            "ProgramAsWeightsBackend.infer exposes no temperature argument, so 0 here is "
            "the upstream default rather than something paw-kit sets or can change."
        ),
        "data_fixture": str(data_path),
        "seed": SEED,
        "n_evaluation": len(evaluation),
        "n_folding": fixture["n_folding"],
        "folding_countries": folding_countries,
        "spec_eval_overlap_ids": spec_overlap_ids,
        "known_spec_eval_overlap": list(KNOWN_SPEC_EVAL_OVERLAP),
        "compiles_made_this_run": compiles_made,
        "arms": summary_arms,
    }

    for a in summary_arms:
        r = a["scores"]["overall"]["rates_pct"]
        s = a["scores"]
        print(f"\n[{a['arm']}] {a['description']}")
        print(f"   exact {r['exact']:.1f}%  strict shape {r['strict_shape']:.1f}%  "
              f"in vocabulary {r['in_vocabulary']:.1f}%")
        xsl = s["excluding_spec_leak"]
        print(f"   exact excluding the spec's own worked example(s) "
              f"({xsl['counts']['exact']}/{xsl['n']}): "
              f"{xsl['rates_pct']['exact']:.1f}%")
        print(f"   distinct outputs {s['distinct_outputs']}  "
              f"top share {s['top_output_share_pct']:.1f}%  "
              f"top-2 share {s['top_two_output_share_pct']:.1f}%")
        print(f"   folded countries {s['folded_countries']['rates_pct']['exact']:.1f}%  "
              f"unfolded {s['unfolded_countries']['rates_pct']['exact']:.1f}%")
        print(f"   misleading-city {s['by_misleading']['misleading_city']['rates_pct']['exact']:.1f}%  "
              f"plain {s['by_misleading']['plain']['rates_pct']['exact']:.1f}%")
        print(f"   distribution: {json.dumps(s['output_distribution'])}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"finetune-lookup-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")

    write_suite(Path(args.suite_out), evaluation,
                str(out_dir / "finetune_lookup_B-paw-4b-qwen3-0.6b.paw"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
