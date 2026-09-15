# Real-backend measurements

Output of `scripts/measure_real_backend.py` against the real `programasweights` service
and a real compiled program — not simulated, not mocked. Each JSON file is one run:
compile time, cold/warm inference latency (p50/p90/p99), and a `paw-test` pass rate on
the suite's standard + fuzzed cases.

## Results so far (`examples/date_normalizer/suite.yaml`, `paw-4b-qwen3-0.6b` compiler)

| Machine | GPU | warm p50 | warm p90 | calls | pass rate |
|---|---|---|---|---|---|
| RTX 3080 (desktop) | none (CPU build) | 5874 ms | 7162 ms | 10 | 71/82 (86.6%) |
| RTX 3080 (desktop) | RTX 3080 | 65 ms | 66 ms | 10 | 71/82 (86.6%) |
| A100 80GB (Imperial DoC cluster, `merlin`) | A100 | 89 ms | 91 ms | 50 | 71/82 (86.6%) |

All three runs compiled to the identical upstream `program_id` (the fast compiler is
deterministic) and failed the identical 11 cases every time — whitespace/control-character-only
inputs, where the model returns an empty string and the suite's blanket
"must look like `YYYY-MM-DD`" assertion then fails it. Same cause each time, not noise.

**The A100 was slower than the 3080, not faster.** This is a real result, not a bug: at
0.6B parameters and batch size 1, this workload is latency-bound (kernel-launch and
memory-access overhead per generation step), not throughput-bound — exactly the regime
where a data-center GPU's advantage doesn't show up. The A100 node was also shared with
other users' running jobs at measurement time, which may have added some contention.
Take it as "roughly hardware-independent for a model this size, maybe a shared A100 has
a bit of noise," not as "A100 is worse than a 3080" in general.

**Practical takeaway**: for this model size, a CPU or a consumer GPU (3080-class) is the
realistic target audience for this library, not a datacenter card — which is also who
most people running local micro-functions actually are. The one thing that *does* matter
enormously is GPU vs CPU at all (5874ms → 65ms, ~90x): see the CUDA-build notes below if
`llama_cpp.llama_supports_gpu_offload()` comes back `False` on your machine.

## JIT hot-swap: real teacher vs real compiled adapter

`scripts/measure_jit_speedup.py` wires `@compile_on_hit` to an actual Claude API call
(the "teacher") and `ProgramAsWeightsBackend` (the compiled local adapter) — the first
time either side of this comparison has been real. All prior demos (`paw-kit demo`,
`examples/triage_ticket/run.py`) fake both: a `time.sleep(0.2)` for "the API" and a
hardcoded dict for "the model".

One run, threshold=5, 20 total calls, `claude-haiku-4-5` as teacher, 3080:

| Phase | Latency |
|---|---|
| Calls 1-4 (real Claude API call) | ~930-1130ms each |
| Call 5 (Claude call + synchronous compile) | 5450ms |
| Call 6 (first local call: adapter download + model load into VRAM) | 7599ms |
| Calls 7-20 (steady-state local inference) | 84-96ms each |

**Correction (2026-09-08, caught by an Opus review):** this section originally claimed
"steady-state ≈ 21x" by dividing `pre_threshold.mean_ms` (1879.8) by `post_threshold.mean_ms`
straight from the script's own summary JSON — but `pre_threshold.mean_ms` averages
calls **1-5**, and call 5 is the one that includes the synchronous compile (5450ms), not
a clean teacher-only call. That's the same mistake the paragraph below criticizes the
script for making on the *other* side of the split (burying the one-time model-load cost
in the post-threshold average) — just committed here in the opposite direction. The
correct, honest split, straight from the per-call data in
`jit-speedup-3080-20260908-165914.json`:

- Teacher-only, calls 1-4: **987ms mean**
- Steady-state local, calls 7-20: **88ms mean**
- **Honest steady-state ratio: ~11.2x**, not 21x.

**Second correction (2026-09-11):** the account above of *how* 21x arose was itself wrong
in one detail — it named `post_threshold.mean_ms` as 88.4. The artifact's actual
`post_threshold.mean_ms` is **589.1**, not 88.4 (verified by reading the committed JSON
directly); it averages calls 6-20, and call 6 is the one-time adapter download and model
load (7599ms), contaminating that average in exactly the same way `pre_threshold.mean_ms`
was contaminated by call 5. Both of the artifact's own top-level split fields are
therefore unreliable, and so is its `speedup_x` field (3.19, computed from those two). The
987ms / 88ms split above was always correct — it comes from the per-call data, not from
either contaminated field — but the committed artifact itself has never contained a field
that holds the honest number. It now does: see
[`jit-speedup-3080-20260908-165914.recomputed.json`](jit-speedup-3080-20260908-165914.recomputed.json),
generated from this file's own `calls[]` by the same split this section describes, with
`teacher_only.mean_ms` 987.33, `steady_state.mean_ms` 88.37, `speedup_x` **11.17**.

Also **zero tokens billed** per call after hot-swap (vs 501 in / 185 out over the 5 traced
calls). The compiled model's classifications aren't identical to Claude's on every ticket
— expected: it was trained on 5 few-shot examples folded into the spec text, not
fine-tuned on a real dataset (see the upstream-limitation note in the main README). One
thing this test did *not* measure, and probably should have: whether the compiled
adapter's classifications are actually *right*, as opposed to merely fast and
schema-shaped — looking at the raw data, calls 7-20 return `priority=medium, urgency=3`
nine times out of fourteen (six of those `medium/technical/3`), which is at minimum worth
checking isn't near-degenerate output before quoting "11.2x" as if it were a
like-for-like speedup on real work. **This was checked directly — see "Semantic
correctness, for real" below: 60% full agreement with a fresh teacher call, and the
medium-clustering pattern noted here is real, not noise.**

> **Corrected 2026-09-09, public-material review.** The sentence above originally read
> "`medium/technical/3` nine times out of fourteen". Recounted from
> `jit-speedup-3080-20260908-165914.json`: the exact triple `medium/technical/3` appears
> **six** times in calls 7-20; it is `medium` priority with urgency `3` that appears nine
> times (six technical, two billing, one sales). The clustering claim stands; the count
> attached to it did not.

## Active learning, for real: a genuine spec gap, not a bug

`scripts/measure_active_learning.py` ran `run_active_learning_loop` against a live
Claude teacher, targeting the real 11/82 fuzzer failures above. Result: **0 repaired,
3/3 iterations exhausted, identical failures every time.**

This is not a broken feature — read the actual mechanism: `_query_teacher_safely`
validates every teacher-provided gold label against the suite's own assertions *before*
trusting it as training data (this is also the harness's prompt-injection defense on
adversarial-probe inputs). For these 11 inputs, all whitespace/control-character/emoji
garbage, Claude declines to invent a plausible-looking ISO date — so its answer never
passes the suite's `^\d{4}-\d{2}-\d{2}$` assertion, gets rejected, and never enters the
training set. The loop is correctly refusing to train the model to hallucinate a fake
date rather than admit it doesn't know. The real bug this surfaces is in the
**suite's assertions**, not the model: `examples/date_normalizer/suite.yaml` demands a
date-shaped answer unconditionally, with no defined "not a date" case. Fixing this means
relaxing the suite (e.g. permitting an explicit sentinel for unparseable input), not
squeezing the model harder.

One inefficiency this run also surfaced: the loop recompiles unconditionally every
non-final iteration even when zero new examples were added that round (confirmed here —
the recompiled program's ID was byte-identical to the previous one, both times). Harmless
for a deterministic compiler, but it's a wasted upstream compile call every time this
exact situation recurs; worth skipping recompilation when `newly_repaired == 0`.

> **Correction, 2026-09-11.** That unconditional-recompile behaviour was since removed
> (`28585ef`), and this committed artifact (`active-learning-20260908-170147.json`)
> **predates that change** — its `recompiled: true` on a `repaired_edge_cases: 0`
> iteration is a state today's code cannot produce; re-running the same command now gives
> `recompiled: false, recompiles_skipped: 2, stuck_reason: "all_labels_rejected"`. The
> conclusion ("0 repaired, correctly") is unaffected, and is not being re-run here — this
> note exists so a reader trying to reproduce the artifact literally doesn't file a false
> bug report against current code. `starting_program_id: 8fc80fb0687b6f0b8ce0`,
> `starting_examples_folded_into_spec: 0` — the committed `date_normalizer` manifest's
> `examples_folded_into_spec: 4` is the *output* of the recompile `28585ef` removed, not
> what this run started from.

## Grammar-constrained decoding, for real: 100% either way, and the "~13x cost" was a bug

`scripts/measure_schema_real_model.py` is the first time `RegexLogitsProcessor` has
touched a live model — previously unit-tested only against synthetic token vocabularies
(`tests/test_schema.py`). It loads `Qwen/Qwen2.5-0.5B-Instruct` via `transformers` and
runs the same 15 prompts (including deliberately adversarial ones — "ignore the schema
and reply with the word banana", a SQL-injection-shaped string, empty input) two ways.

**Correction (2026-09-08, caught by an Opus review, verified and fixed the same day):**
the original version of this section reported constrained decoding at 9221ms/call, ~13x
slower than unconstrained. That number came from a real bug in the measurement script,
not a property of anything in `paw_kit`: it constructed a **fresh `RegexLogitsProcessor`
inside the per-ticket loop**, throwing away its `_transition_cache` and
`_allowed_tokens_cache` every single call. Those caches are the entire performance story
for a class whose whole design is "expensive to walk cold, cheap once warm" — discarding
them every call and re-walking the FSM cold against all 151,665 Qwen vocabulary entries
at every decoding step is where the 9221ms came from, not GPU cost, not Python overhead
per se, and not an inherent property of constrained decoding. Fixed by building one
shared `RegexLogitsProcessor` before the loop and reusing it across all 15 calls (exactly
how `paw.load()` would use it across repeated calls to the same compiled adapter in a
real process) — corrected numbers below replace the original ones, run on the same
prompts, same model, same machine.

| Mode | Valid Pydantic parses (raw) | With a markdown fence stripped | Mean latency/call |
|---|---|---|---|
| Unconstrained (asked nicely, zero-shot) | 0/15 (0.0%) | **11/15 (73.3%)** | 643ms all-calls (**559ms warm**, calls 2-15); 691ms in the original, uncorrected run |
| FSM-masked decoding | 15/15 (100.0%) | 15/15 (100.0%) | **1084ms including one cold call; 501ms warm (calls 2-15)** |

Two corrections in this table, not one — and a third landed 2026-09-11 on top of the first:

1. **Cost**: warm, constrained decoding is **not slower than unconstrained** — if
   anything slightly faster, because the schema forces compact output with no markdown
   decoration, while the unconstrained model pads its answer with a ` ```json ` fence and
   indentation. There is still a real, one-time cold-cache cost on the *first* call with
   a new schema (9236ms in this run, matching the original run's number almost exactly —
   that number wasn't wrong, it was just wrongly generalized to every call instead of
   only the first one).
   - **Correction, 2026-09-11:** the gap above was published as "501ms vs 643ms" — but
     643ms is the unconstrained arm's **all-call** mean, and its own call 1 carries a
     1819ms one-time cost (the same framework warm-up the constrained arm's call 1 pays
     as 9236ms), so the unconstrained side of that comparison was warm-vs-**cold**, not
     warm-vs-warm — inflating the apparent advantage of constrained decoding by 2.4x.
     Recomputed directly from this section's own committed artifact, dropping call 1 of
     *both* arms: unconstrained warm mean **559.4ms**, constrained warm mean **501.4ms**,
     an honest gap of **58ms**, not 142ms. The conclusion is unchanged (constrained is not
     slower once warm) — only the stated size of the effect was wrong, in the direction
     that favoured the library's own claim.
2. **"0/15 unconstrained" was technically true but misleadingly framed.** Every one of
   those 15 failures is the same shape: correct JSON wrapped in a code fence, not garbage.
   With the fence stripped, 11/15 (73.3%) actually parse — the 4 real failures are a
   value outside the `Literal` (`"sales/general"`) and one truncation at the token
   budget. The defensible version of the "0%" claim is narrower than "the model can't
   produce valid JSON": it's that **`paw_kit.schema.load()` itself does no fence-stripping**
   (`paw_kit/schema/loader.py:92` calls `response_model.model_validate_json(raw_output)`
   directly), so 0/15 *is* what the library would actually do with these outputs today —
   that's a real, defensible claim about the library. "The model can't produce valid
   JSON" is not, and shouldn't be implied.

What's still true and unaffected by either correction: **constrained decoding was 15/15
valid both raw and fence-stripped, at both the wrong and the corrected latency.** The
"0.0% Pydantic syntax failures" invariant in `conductor/decisions.md` holds up against a
real model either way — that was never the part that was wrong.

**What's still genuinely untested here**: semantic correctness. All that's measured in
this section is *shape*, not *rightness* — the FSM guarantees the former by construction
and says nothing about the latter. **Now measured separately — see "Semantic
correctness, for real" below.**

`paw_kit/backend/real.py`'s docstring previously claimed `infer()` "applies
`RegexLogitsProcessor` if grammar_constraint is provided" — also false, also caught by
the same review, also fixed the same day: the import was dead and nothing in `paw_kit`
ever actually applied it. The wiring that made the numbers above possible lives entirely
in this script, not in the shipped library; `real.py`'s docstring now says so.

> **Note, 2026-09-15 (Track 15).** The numbers above stand as measured and the script
> that produced them, `scripts/measure_schema_real_model.py`, is **deleted**: it drove
> `RegexLogitsProcessor` directly against a bare HuggingFace Qwen2.5-0.5B-Instruct, and
> that class no longer exists. The committed artifact is the record. Two things about
> this section to carry forward rather than re-read literally:
>
> - **The 15/15-vs-0/15 contrast has no successor on a compiled PAW adapter.** It was
>   produced against an un-fine-tuned instruct model asked nicely for JSON. Re-run
>   through the shipped backend on an adapter compiled for its own schema, the same
>   comparison is 15/15 versus 15/15 and 60/60 versus 60/60, byte-identical — see the
>   2026-09-15 section at the end of this file. The null is the result.
> - **The engine that made these numbers had a hole this schema never reached.** The
>   character-level FSM walked a byte-level BPE vocabulary, so in both vocabularies built
>   for it, 1,456 tokens decoding to `U+FFFD` were admitted by the JSON string-content
>   class (1,447 not standalone-valid UTF-8). This section's `Triage` has three closed
>   `Literal` sets and no free string field, and 0 of its 121 live FSM states admit any of
>   them — which is why the result stands. Any schema here with a free `str` field would
>   not have. The replacement engine is byte-level and the hole is structurally closed.

## Semantic correctness, for real: does it mean the right thing, not just look right

> **What this measurement led to.** The 60% agreement figure below is why
> `@compile_on_hit` now has [shadow mode](../docs/shadow-mode.md): a compiled adapter is
> promoted only after a window of live comparisons clears a threshold, and with the shipped
> defaults this adapter would not have promoted. The measurement scripts are pinned to
> `shadow_window=0` so the runs recorded here stay reproducible. Shadow mode itself is
> now measured below, in "Shadow mode, for real": at the shipped defaults this adapter
> does not promote, and stalls after five windows.

Every test above this line checks *shape*: does the output match a regex, parse as JSON,
hit a length bound. None of them ask whether the output is actually a correct answer.
This section does, two ways: (1) re-scoring the ticket-triage adapter from the JIT test
above against a **fresh, independent teacher call** on the same ticket, and (2) three
brand-new specs, written the way the upstream docs' own front page actually models spec
writing — a single terse sentence, no input/output examples folded in — scored by an
independent Claude judge (`scripts/measure_semantic_correctness.py`) shown only the spec,
the input, and the adapter's output, never the suite's own `expected` field.

### Ticket-triage re-score: is "11.2x" comparing like-for-like work?

`scripts/measure_triage_semantic_agreement.py` compiles a fresh adapter for the exact
same spec and 5 folded-in examples as the JIT test, then for all 20 tickets compares the
adapter's classification against a **second, independent** teacher call on the same
ticket (not the originally-traced response — a live second opinion, so agreement isn't
just "did it memorize the trace").

| Metric | Result |
|---|---|
| Full agreement, all 20 scored tickets | 60% (12/20) — **see correction below: this figure leaks** |
| Full agreement, 15 tickets held out of the fold | **46.7% (7/15)** |
| Full agreement, 5 folded-into-the-spec tickets | 100% (5/5) |
| Urgency score within 1, alone (held out) | 86.7% (13/15) |

> **Correction, 2026-09-11.** The 60% figure folds `TICKETS[:5]` into the adapter's own
> spec text as few-shot examples, then scores all 20 tickets including those same 5 — so
> a fifth of the "agreement" measurement is the adapter reciting an answer it was handed
> verbatim. Confirmed byte-identical: all 5 folded tickets' adapter output matches the
> fresh teacher call exactly. The honest number is the 15 genuinely held-out tickets:
> **46.7%**, thirteen points below the figure this project has quoted since 2026-09-09.
> Re-derived directly from the committed `cases` array in
> `triage-semantic-agreement-3080-20260909-002033.json` — no re-run was needed to produce
> this number, only to un-conflate two slices that were always both present in the data.
> The disagreement analysis below still describes the same 8 raw mismatches; only the
> denominator used to call it "60%" was wrong. The script itself now folds from a
> disjoint pool (`FOLDING_TICKETS`, 5 new tickets) rather than the eval set, asserts the
> disjointness at runtime, and reports the held-out rate explicitly — so this leak cannot
> recur silently.

The disagreements are not random noise: in 5 of the 8 mismatches, the adapter says
`medium` where the fresh teacher call says `high` (SOC-2 report, password reset, iOS
crash, 403 error, EU 2FA/SMS) — the same "regression to medium" the raw JIT data hinted
at above, now confirmed by an independent comparison rather than inferred from one run's
output distribution. Two mismatches go the other way (adapter `high`, teacher `medium`,
on a non-profit-discount and an SLA question), so it isn't a uniform downward bias, but
a real central-tendency pull is visible. Caveat, honestly stated: the teacher itself is
not perfectly consistent call to call, so 46.7% is a ceiling on "the adapter is wrong," not
a floor — a disagreement means the two differ, not necessarily that the adapter is at
fault. Either way, **the JIT speedup and the semantic fidelity are two separate claims**;
11.2x describes the first, not the second.

### Terse, docs-style specs: what a one-line spec (as the docs actually show it) produces

The upstream docs' front-page example is `paw.compile("Fix malformed JSON: repair
missing quotes and trailing commas")` — one sentence, nothing else — even though the
SDK's own `compile()` docstring says "Include examples in the text." Every example this
toolkit ships (`date_normalizer`, `pii_scrubber`, `triage_ticket`) uses the *carefully
scoped, long* style instead. These three specs (`measurements/spec-drafts/`, designed by
a fresh-context Sonnet agent specifically to probe this gap) use the docs' own terse
style, compiled with `max_spec_examples=0` — no examples folded in, exactly what a new
user copying the front page would get.

| Spec (verbatim) | Cases | Structural pass | Semantic pass |
|---|---|---|---|
| `"Fix malformed JSON: repair missing quotes and trailing commas"` (the docs' own example) | 153 | 93.5% | 69.9% |
| `"Pull out the phone number from this text and format it consistently."` | 134 | **0.0%** | 89.6% |
| `"Decide if this product review is positive or negative."` | 133 | 92.5% | 89.5% |

**The phone-extractor row is the headline finding.** 0% structural pass looks alarming
until you look at the actual outputs: the compiled adapter reliably returns
`555-123-4567` (dash-separated); the test suite (written by the same design agent, before
compiling) assumed `(555) 123-4567` (parenthesized) as "the" consistent format, because
the spec never says which. The adapter picked *a* consistent format, just not the one a
human test-writer guessed — 89.6% of its outputs are judged semantically correct extractions.
This is not a defect in `paw_kit` or the compiler; it's the terse-spec ambiguity the docs'
own example invites, made concrete: **a test suite written against an assumption the spec
never actually pins down will show near-total "failure" that has nothing to do with
whether the compiled program works.**

That said, real defects did turn up, exactly where fuzzing is supposed to find them:

- **A genuine hallucination**, caught only by an out-of-training-distribution fuzz probe:
  given `"Dial 1-800-FLOWERS for delivery."`, the adapter returned `1-800-555-0123` —
  fabricated digits with no relationship to the input. `FLOWERS` is a vanity
  letters-for-digits number the spec never mentioned handling; the model didn't decline
  or flag it, it invented a plausible-looking wrong answer. This is the single clearest
  case across all four tests of "confidently wrong," and the only reason it surfaced is
  that a fuzzer put an unusual real-world phone-number style in front of the adapter.
- **Silent extension dropping and silent multi-number selection**: `"555-222-3333 ext.
  204"` loses the extension; `"555-123-4567 or 555-987-6543, whichever works"` silently
  returns the first number with no signal that a second one existed. Both are defensible
  choices in isolation, both are undocumented behavior an integrator would only discover
  by reading outputs one at a time.
- **Sentiment: the forced binary leaks a third label under real ambiguity.** The suite's
  own `not_contains: neutral` assertion (correctly) fails whenever the compiled adapter
  returns `"neutral"` for empty, gibberish, or genuinely balanced input — which it does,
  reliably, despite the spec allowing only two labels. Arguably the *more honest* answer
  for unclassifiable input is exactly this: refuse the binary rather than force a
  coin-flip. But it breaks the machine-checkable contract the spec promised, and nothing
  currently tells a caller "this output isn't one of the two values you asked for" short
  of the schema layer catching it.
- **One predicted failure that did *not* materialize**: the design agent hypothesized
  sarcasm would get misread by literal polarity words. It didn't —
  `"Oh great, ANOTHER product that breaks in a week. Love it."` was correctly classified
  `negative`, and a French-language review (`"Ce produit est incroyable, je l'adore !"`,
  never mentioned in the spec or any example) was correctly classified `positive`.
  Recorded here because a falsified hypothesis is as much a real result as a confirmed
  one — see the project's own stated preference for reporting blind spots over curating
  a clean narrative.

**A limitation of this methodology, found while reading the raw judge output, not
hidden after the fact**: a large share of the JSON-repair "semantic fail" verdicts (most
of the gap between 93.5% structural and 69.9% semantic) are the judge flagging that the
adapter *silently* stripped an invisible Unicode character (a right-to-left override,
a BOM, a null byte) while producing valid JSON — technically outside "repair missing
quotes and trailing commas," but not obviously wrong either. The judge treats any
undisclosed handling of invisible/control characters as suspicious by default, including
cases with no plausible security relevance (a bare BOM) alongside at least one with a
real one (a right-to-left-override character silently dropped, which in a real pipeline
could be masking a homograph/spoofing attempt rather than incidental whitespace). Read
the 69.9% and 89.5% numbers above as **an upper bound on how large the real defect rate
is, not a clean ground truth** — a stricter or more lenient judge prompt would move both
numbers without anything about the adapters changing. Full case-by-case output, including
every judge reason string, is in `measurements/semantic-*-3080-*.json`.

## Does folding examples into the spec text actually help? A real answer, on the second try

`paw_kit/backend/programasweights.py`'s own docstring names this as an open question:
the upstream compiler doesn't accept training examples directly, so traced/gold
examples can only reach it as few-shot demonstrations folded into the spec text
(`max_spec_examples`) — "whether it measurably helps is an open question... do not
assume it does." This section answers it, on the three terse specs above: same spec,
same compiler, `max_spec_examples=0` (already measured above) vs `max_spec_examples=8`
(all 8 available standard-case examples folded in).

**First attempt was wrong and is worth stating plainly rather than quietly fixing**:
`scripts/measure_semantic_correctness.py` originally called
`backend.compile(suite.spec, [], adapter_path)` — a hardcoded empty examples list,
regardless of `--max-spec-examples`. The flag configured the backend's cap; with
nothing ever passed in for it to cap, three "fewshot8" runs silently compiled with zero
examples folded in, identical to the baseline they were meant to be compared against.
Caught because the first result set looked suspiciously unchanged (phone-extractor
still 0% structural with the exact same dash-separated outputs) — confirmed via the
compile manifest's own `examples_folded_into_spec` field, which read `0` for a run
that had just been given `--max-spec-examples 8`. Fixed (build real examples from
`suite.standard_cases`, same convention as `measure_real_backend.py`) and re-run; the
numbers below are from the corrected script, confirmed via each manifest actually
reading `examples_folded_into_spec: 8`.

| Spec | Structural: 0 examples → 8 examples | Semantic: 0 examples → 8 examples |
|---|---|---|
| Phone extractor (`"...format it consistently"`) | **0.0% → 92.5%** (124/134) | 89.6% → 85.8% |
| Review sentiment (`"positive or negative"`) | 92.5% → **100.0%** | 89.5% → 88.7% |
| JSON repair (docs' own example) | 93.5% → 92.8% | 69.9% → 68.0% |

**Correction (2026-09-09, caught by the delegated finetune-compiler agent below cross-checking
this table against the artifact it cites):** this table originally read 93.3%/86.4%/93.2%/68.4%
for three of these six cells — each off by rounding to the wrong nearby number rather than the
actual `structural_pass_rate`/`semantic_pass_rate` fields in
`measurements/semantic-phone_extractor-3080-fewshot8-20260909-005717.json` and
`measurements/semantic-json_repair-3080-fewshot8-20260909-005723.json`. Corrected above; the
review-sentiment row (100.0%/88.7%) was already right. None of the qualitative conclusions below
change — the phone-extractor swing is 0.0%→92.5%, not 0.0%→93.3%, still the same dramatic result.

**Yes, dramatically, for format ambiguity; no, for everything else.** The phone
extractor and review-sentiment jumps are real and large: with 8 examples all showing
`(555) 123-4567`-style formatting, the adapter switched from reliably outputting
`555-123-4567` to reliably outputting the parenthesized format the examples
demonstrated — the exact ambiguity flagged earlier in this document as the reason the
0-example run failed structurally. Review sentiment's forced-binary leakage
(`"neutral"` on ambiguous input) also disappeared entirely once examples pinned the
output down. JSON repair, whose failures were mostly about invisible-Unicode handling
the docs' own two named defects (missing quotes, trailing commas) never covered, was
unmoved — the 8 examples don't demonstrate anything about control characters, so there
was nothing for them to fix.

**A new failure mode showed up specifically because of the examples, not despite
them**: given `"International line: +44 20 7946 0958"` (a real UK number, nothing like
the examples), the 8-example adapter returned `(555) 123-4567` — not a formatting
choice, the literal example number, verbatim, for an input that shares nothing with it.
The 0-example adapter never did this; its worst failures were dropping information
(an extension, a country code) or fabricating plausible-looking digits, not regurgitating
a training example wholesale. Folding in examples fixed the format-consistency problem
and introduced a memorization-under-distribution-shift one — both real, on the same
adapter, from the same fix. Worth knowing before treating "fold in examples" as a
free-standing recommendation rather than a trade-off.

## Fail-open safety under a real failure

Every existing test of "local compiled functions are never a single point of failure"
(`decisions.md`'s #1 invariant) mocks the exception (`raise Exception("boom")` from a
fake backend). `scripts/measure_fail_open.py` induces two real failures against the real
`ProgramAsWeightsBackend` instead:

1. **A real compiled adapter file goes missing** after a successful compile (disk
   cleanup, bad deploy, race condition). Real `FileNotFoundError`, real fallback:
   **confirmed** — the decorated function returned the teacher's result, not a crash.
   This is the clean result of the test.
2. **A deliberately invalid `PAW_API_KEY`** at compile time. This one came back
   **inconclusive**, honestly reported: the real upstream service did not reject the
   bogus key at either the compile or the download/inference step — `backend.infer()`
   called directly (bypassing the fail-open wrapper entirely, so there was no fallback
   to hide behind) returned real output, no exception. So this run never produced an
   actual failure to fall open *from* — meaning fail-open-on-compile-rejection is still
   genuinely untested against the real service.

   *Update, same day, one hypothesis checked and ruled out:* an Opus review guessed this
   might be an artifact of our own code — `ProgramAsWeightsBackend._paw()` caches the
   imported `programasweights` module via `importlib.import_module`, so maybe the SDK
   reads `PAW_API_KEY` once at import time and the later env-var swap never reached it.
   Checked against the installed SDK's actual source
   (`.venv/lib/python3.12/site-packages/programasweights/config.py`): `get_api_key()`
   reads `os.environ.get("PAW_API_KEY")` fresh on every call, with no caching at any
   layer between that and the HTTP request (`client.py:116-121` builds the `X-API-Key`
   header from it per-request). So the module-caching theory doesn't hold — the bogus
   key really was read fresh and really was sent to the server. That rules out the most
   plausible "it's our bug" explanation; it doesn't produce a positive explanation for
   why the server accepted it, and we stopped there deliberately rather than continuing
   to probe a third party's undocumented auth behavior. The compiled-program oddity
   noted below is unrelated to this specific question and also unconfirmed.

   Separately, the resulting compiled program, for this task/key combination, appears to
   have degenerated into memorizing the deliberately-uninformative training examples
   verbatim (constant output regardless of input) rather than following the spec the way
   the identically-shaped Phase 1 compile (valid key) did. Still just an observation, not
   a finding — no controlled valid-key comparison was run for this specific spec.

Net: fail-open on the most realistic failure mode (missing/corrupted local state) is
confirmed for real. Fail-open on a rejected compile request is not yet — nothing
available made the real service actually reject a request during this test session.

## Finetune compiler (`paw-ft-bs48`), for real

Everything above this line — 3 hardware configs, JIT hot-swap, active learning, grammar
decoding, fail-open, the three terse specs, the folded-examples A/B — used exactly one
compiler, `paw-4b-qwen3-0.6b` (the paper's single-forward-pass "pseudo-program" mapper).
The account also exposes `paw-ft-bs48`, the finetune compiler from "Compile by Training"
(arXiv:2609.04199), which `paw_kit` routes through `compile_async` + polling. It had
never been run. This section runs it, once, on the highest-signal suite available:

```bash
uv run python scripts/measure_semantic_correctness.py \
    measurements/spec-drafts/spec-2-phone-extractor.yaml \
    --compiler paw-ft-bs48 --label finetune-3080 --max-spec-examples 8
```

Same suite, same 8 folded-in examples, same judge, same 3080, as the `max_spec_examples=8`
fast-compiler run two sections up — so this is a clean three-way comparison on identical
input. Result: `measurements/semantic-phone_extractor-finetune-3080-20260909-010553.json`.

| Run | Compiler | Examples | Compile wall | Structural | Semantic |
|---|---|---|---|---|---|
| Terse baseline | `paw-4b-qwen3-0.6b` | 0 | ~1–5s | **0.0%** | 89.6% |
| Folded examples | `paw-4b-qwen3-0.6b` | 8 | ~1–5s | 92.5% (124/134) | 85.8% (115/134) |
| Finetune | `paw-ft-bs48` | 8 | **180.8s** | 93.3% (125/134) | 87.3% (117/134) |

It works end-to-end. No auth problem, no timeout, no payment or quota error; the async
job reached `status: ready` with a distinct `program_id` (`42db8135a1ac0089ce3d`, vs the
fast compiler's `1b070b72ff231d4710c3` for the same spec), and the resulting program
downloaded and ran locally like any other. Compile took **180.8s against the fast
compiler's 1.0–5.0s** on the same machine (`compile_wall_s` in the three hardware runs at
the top of this file) — roughly 36–175x, and squarely inside the "~2-5 min" the upstream
`list_compilers()` description advertises for this compiler. That is not a hang and not a
surprise; it is the advertised cost. (The 180.8 s figure was printed by the script and is
not persisted in the run's JSON — `compile_s` is not a field of that artifact, as the
deferred-topics log already notes — so unlike every other number in this section it
cannot be re-derived from a committed file. The manifest's `compiled_at` and the run's
timestamp are consistent with it, no more.)

**The headline is not in the percentages. 132 of the 134 outputs are byte-identical to
the fast compiler's.** Not similar — identical. Every one of the 8 standard cases, every
Unicode-injected variant, every empty/BOM/emoji probe. Two cases differ, and both are
out-of-distribution adversarial probes:

| Input | `paw-4b-qwen3-0.6b`, 8 ex. | `paw-ft-bs48`, 8 ex. |
|---|---|---|
| `"Dial 1-800-FLOWERS for delivery."` | `""` | `(800) 777-7777` |
| `"International line: +44 20 7946 0958"` | `(555) 123-4567` | `(207) 794-0958` |

Read those two rows carefully, because they are the only real evidence either way about
whether this is a different compile:

- The UK-number case is the memorization failure flagged in the folded-examples section
  above: the fast compiler returned `(555) 123-4567` — the literal example number, verbatim,
  for an input sharing nothing with it. **`paw-ft-bs48` does not do that.** It returns
  `(207) 794-0958`, which is the input's own digits (`20 7946 0958`) re-grouped into the
  demonstrated format. Still wrong — it drops `+44` and passes a London landline off as a
  US number — but wrong in a way that is *derived from the input* rather than copied from
  a demonstration. That is a genuinely different, and arguably better-behaved, failure.
- The vanity-number case goes the other way. The fast compiler declined (empty string);
  `paw-ft-bs48` emitted `(800) 777-7777`, fabricated digits with no relationship to
  `FLOWERS` (the keypad mapping would be `356-9377`). This is the same hallucination class
  as the 0-example run's `1-800-555-0123`, and it is the one case where the finetuned
  program is *worse*: it is the case that moved structural pass from 124 to 125, and the
  entire structural improvement in the table above is this single fabricated answer
  satisfying a regex. Reported as a defect, not a gain.

**The semantic column is noise, and this run happens to prove it.** Semantic pass went
85.8% → 87.3%, which looks like a small improvement. It is not: only 2 of 134 outputs
changed, and both were judged incorrect in both runs. Six cases flipped verdict, and all
six flipped on **byte-identical output text** — the judge said `(555) 666-7777` for
`"Office line: +1-555-666-7777"` was correct in one run and "removed country code +1
without justification" in the other; the zero-width-space-prefixed copy of that same input
flipped the opposite direction in the same pair of runs. That is a **4.5% (6/134)
test-retest flip rate on a fixed input/output pair**, measured for free here because the
adapter held still. Every semantic-pass number in this document is a `max_tokens=60`,
default-temperature Claude call; treat differences of ~2 points between any two rows as
indistinguishable from this. Logged in `conductor/deferred/index.md`.

**So: is `paw-ft-bs48` a genuinely different compile, or the same behavior on a slower
path?** On this task, honestly: **neither cleanly, and closer to the second than anyone
would hope.** It is demonstrably not the same program (different `program_id`, different
`compiler_kind` upstream — `finetune_lora` vs `mapper_lora`), and its two divergences are
not random: one replaces example-regurgitation with input-derived extraction, which is the
single most encouraging thing in this run. But 132/134 identical outputs and a structural
"gain" that consists entirely of one hallucination passing a regex is not a result anyone
should describe as "much higher accuracy" (the upstream description's phrase) on the
strength of this test. On a task where an 8-example fast compile already sits at 93%,
there is almost nothing left for a finetune to win, and three minutes of compile bought
about one probe's worth of behavioral difference.

Scope limits, stated rather than buried: **one suite, one task, one run, one seed.** The
places `paw-ft-bs48` would plausibly earn its wall-time — a task the fast compiler
genuinely fails at, a longer spec, more than 8 examples, an output format with real
structure — are exactly the places this test did not go. A single negative-to-neutral
result on a near-saturated task is weak evidence about the compiler in general, and is
not a reason to conclude the finetune path doesn't work. It is a reason not to reach for
it by default.

*Cost note, since this was the one compiler expected to carry real cost*: no
payment, quota, or rate error at any point, consistent with the unmetered-beta reading in
the cost note below. One `compile_async` job, ~3 minutes of upstream compute, 134 local
inferences, 134 Haiku judge calls.

**Bookkeeping discrepancy caught here, fixed at the source**: this section originally
flagged that the folded-examples table above reported the phone-extractor 8-example
structural rate as `93.3%` while the run artifact it cites
(`measurements/semantic-phone_extractor-3080-fewshot8-20260909-005717.json`) records
`92.537%` (124/134) — a rounding/transcription slip in the earlier table, not in this
one. Corrected at the source (the table now reads `92.5%`); this section's own
`92.5% (124/134)` fast-compiler row was already right throughout.

## Finetune compiler on a hard task (ticket triage): both compilers fail it

The section above ends by naming its own blind spot: `paw-ft-bs48` had only ever been run
on phone extraction, where an 8-example fast compile already sat at 93% and 132 of 134
outputs came back byte-identical, and "the places `paw-ft-bs48` would plausibly earn its
wall-time — a task the fast compiler genuinely fails at ... — are exactly the places this
test did not go." This section goes there, on ticket triage: the one task in this
document where the fast compiler is known to be *wrong* rather than near-saturated (the
60%-agreement re-score above).

**The finetune compiler is measurably better here, and it is nowhere near enough.**
`paw-ft-bs48` scores 60.0% full agreement against a 91.7% teacher ceiling, versus the best
fast-compiler arm's 53.3%. It closes 6.7 of the ~38-point gap — about a sixth — for 45x
the compile wall time, and it buys that by trading department accuracy away for priority
accuracy rather than by getting better at the task. On a task the finetune compiler fails, the
fast compiler also fails, somewhat worse. What this run demonstrates is that neither
compiler rescues this task; it does not show that the fast compiler matches the finetune
compiler in general, and a task the finetune compiler can do and the fast one cannot is
the next thing to look for.

> **Note, 2026-09-10 (earlier the same day):** this section was first published as a
> two-arm comparison because arm C could not be compiled at all. Async compile was refused
> service-side with HTTP 503 `durable_queue_unavailable`, twice, ~25 minutes apart, while
> `GET /api/v1/health` reported `{"status":"degraded", ...,
> "warnings":["redis_unavailable: using in-memory global rate limit fallback"]}`. The
> refusal is recorded in full under "What compile C actually did" below, because it is a
> real property of the service and not a footnote. At 18:03 the same day health returned
> `{"status":"healthy", "queue_depth":0, "warnings":[]}` and the compile went through on
> the first attempt.

```bash
# Data: 20 recorded tickets + 48 fresh, teacher-generated, deduplicated.
# 8 fresh -> folding pool; 40 fresh + 20 recorded -> 60-ticket evaluation set.
# Every evaluation ticket labelled TWICE, independently, at temperature 0.
# The fixture is committed; this run rebuilt nothing.

# Compile arm C on the finetune compiler (compile_async + polling, public=False).
uv run python scripts/measure_finetune_triage.py --label 3080 --no-run --arms C

# Arms A and B: manifests restored from the service cache (see "Provenance" below).
uv run python scripts/measure_finetune_triage.py --label 3080 --no-run --arms B
uv run python scripts/measure_finetune_triage.py --label 3080 --no-run --arms A

# Inference: 60 tickets per arm on the 3080, temperature 0, all three arms back to back.
uv run python scripts/measure_finetune_triage.py --label 3080 --skip-compile --arms A,B,C
```

Artifacts: `measurements/finetune-triage-tickets.json` (the committed data fixture: every
ticket, both teacher labels, the folding pool), `measurements/finetune-triage-3080-20260910-181612.json`
(the three-arm run), `measurements/finetune-triage-suite.yaml`,
`measurements/finetune-triage-compare-BC.json`, `measurements/finetune-triage-compare-AC.json`,
`measurements/finetune-triage-judge-BC.json`. The earlier two-arm run
(`measurements/finetune-triage-3080-20260910-145744.json`) and its A-vs-B compare/judge
reports are kept as-is.

### Provenance, and a warning the previous version of this section wrote about itself

The previous version of this section observed that the `.paw` manifests are gitignored
(`.gitignore:39-40`) and noted: "a measurement whose provenance lives only in an ignored
file is one `git clean` from unverifiable." **That happened, to this measurement, within
the same day.** Arm A's and arm B's manifests were gone from the working tree before this
run started; every committed artifact still referred to them by program id, and nothing
in the repo could have rebuilt them.

They were recoverable, but only by luck of how the service behaves: recompiling an
unchanged spec returns the *existing* program from cache rather than compiling a new one.
Both came back in ~1.2 s with `cache_hit: true` and **the same program ids the recorded
run used** — `a8be657c5ac492f0c296` (A) and `f0df1ccfe6083ed8f6f7` (B). The restore is
verified, not assumed: re-running all 60 tickets through the restored A and B reproduces
the recorded run's outputs **60/60 byte-identical** on both arms, and reproduces 38.3% and
53.3% full agreement exactly. `paw-test judge` independently returns 66.7% (40/60) for
arm B, the same figure the earlier A-vs-B judge run produced.

Two things follow. First, the provenance question is no longer hypothetical and should be
settled: every manifest field this section relies on — `program_id`, `compiler_snapshot`,
`examples_folded_into_spec`, `compile_wall_s`, `public`, `cache_hit`, and now
`spec_sha256`/`full_spec_sha256`/`status`/`compiled_at` — is mirrored per arm into the
committed run JSON, so the lineage is checkable without the `.paw` files. Second,
`scripts/measure_finetune_triage.py` used to treat `--skip-compile` with a missing
manifest as licence to compile a fresh adapter, silently, under a flag that promises not
to. That is fixed: it now raises rather than substituting a different adapter for the one
the recorded numbers came from.

The mirrored hashes are what make the B-vs-C comparison controlled. `full_spec_sha256`
covers the spec *and* its folded examples, and B and C share it exactly
(`c117b760662b7316…`), so those two arms differ in the compiler and in nothing else. Arm A
differs (`da85b897331dcfcf…`, no examples folded).

### The arms

| Arm | Compiler | Snapshot | Examples | Program id | Compiled |
|---|---|---|---|---|---|
| A | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 0 | `a8be657c5ac492f0c296` | yes (restored from cache) |
| B | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 8 folded | `f0df1ccfe6083ed8f6f7` | yes (restored from cache) |
| C | `paw-ft-bs48` | `paw-ft-bs48-20260530` | 8 folded | `3f49dbc83745469dd6a9` | **yes — fresh, `cache_hit: false`** |

All three were compiled `public=False` (the `ProgramAsWeightsBackend` default), which
matters here: the folded spec carries eight full ticket bodies.

### Results

Scored against the **first** teacher label. "Full agreement" is all three fields exact;
the "urgency ±1" column is the looser criterion the recorded 60% run used, kept so the
two are comparable. Latency is per call on the 3080, excluding the first call, with all
three arms run back to back in one session.

| Arm | Full agreement (all 3 exact) | Full (urgency ±1) | Priority | Department | Urgency exact | Urgency ±1 | Parse failures | Compile wall | Latency/call |
|---|---|---|---|---|---|---|---|---|---|
| A — fast, 0 examples | 38.3% (23/60) | 46.7% | 50.0% | 81.7% | 40.0% | 93.3% | 0/60 | 4.29 s | 89.6 ms (median 90.1) |
| B — fast, 8 examples | 53.3% (32/60) | 55.0% | 63.3% | **90.0%** | 60.0% | 93.3% | 0/60 | 4.94 s | 112.4 ms (median 113.2) |
| C — finetune, 8 examples | **60.0%** (36/60) | **61.7%** | **71.7%** | 83.3% | **68.3%** | **95.0%** | 0/60 | **223.3 s** | 112.7 ms (median 113.0) |
| **Teacher ceiling** (label 1 vs label 2) | **91.7%** | 96.7% | 96.7% | 100.0% | 91.7% | 100.0% | n/a | n/a | n/a |

Compile wall for A and B is the **original** first-compile time from the recorded run
(4.29 s, 4.94 s). This run re-fetched both from cache in ~0.88 s, which is a cache lookup
and not a compile; the run JSON records `cache_hit: true` for A and B and `false` for C,
and the 0.88 s figures are what its `compile_wall_s` fields contain. C's 223.3 s is a
genuine, uncached finetune compile — **45x arm B's compile** and about 250x the cached
lookup.

Against the *second* teacher label the ordering holds and C's margin is slightly wider:
A 35.0%, B 46.7%, C 56.7%. The gap between B and C (6.7 points on label 1, 10.0 on
label 2) is of the same order as the gap between the two label sets (≈3–7 points), so it
is real but not comfortably outside label noise — unlike the A-to-B gap (≈15 points),
which is. **One run, one seed; the 6.7-point headline should be read as "a few points",
not as a precise quantity.**

Per-call latency is the finding nobody should skip: **C is 112.7 ms against B's
112.4 ms.** The finetune compiler produces an adapter the same size and the same speed at
inference. Everything it costs, it costs once, at compile.

### Where arm C's 6.7 points come from

Per-case against B: **C fixes 8 of B's errors, breaks 4, and leaves 20 wrong the same
way.** Against A: fixes 14, breaks 1. The mechanism is visible in the output
distributions, and it is not "C is better at triage" — it is two separate movements that
partly cancel.

Priority:

| | medium | high | critical | low |
|---|---|---|---|---|
| A's outputs | **35** | 10 | 10 | 5 |
| B's outputs | 27 | 21 | 7 | 5 |
| C's outputs | 20 | 22 | **10** | 8 |
| Teacher (label 1) | 9 | 29 | 10 | 12 |

Department:

| | billing | technical | sales | general |
|---|---|---|---|---|
| A's outputs | 13 | 30 | 6 | 11 |
| B's outputs | 13 | **33** | 5 | 9 |
| C's outputs | 14 | 23 | **9** | **14** |
| Teacher (label 1) | 17 | 27 | 6 | 10 |

**1. C undoes the exact damage folding did to B.** The previous version of this section
identified three cases where folding made B *worse*, all `critical → high`, and traced it
to the folding pool's label skew (6 of its 8 teacher labels are `high`/4). C, folding the
byte-identical spec, gets all three right:

| Ticket | Teacher | A (0 ex.) | B (8 ex.) | C (finetune, 8 ex.) |
|---|---|---|---|---|
| `"Our production API gateway is throwing 502 Bad Gateway errors across all regions!"` | `critical`/5 | `critical`/5 | `high`/4 | **`critical`/5** |
| `"Urgent: our webhook deliveries have been failing silently for 3 hours, losing orders."` | `critical`/5 | `critical`/5 | `high`/4 | **`critical`/5** |
| `"Hi, I'm getting a 503 Service Unavailable error whenever I try to deploy my Node.js application to the us-east-2 region."` | `critical`/5 | `critical`/5 | `high`/4 | **`critical`/5** |

C emits `critical` exactly 10 times against the teacher's 10, where B emits it 7 times.
**This is the one place the finetune compiler clearly earns something the fast compiler
cannot do**: given the same 8 skewed examples, it does not inherit their ceiling. That is
a real and useful property — folding's headline failure mode in this document does not
reproduce on `paw-ft-bs48`.

**2. C also breaks the pull to `medium`** that both fast arms suffer, further than B did:
`medium` 35 (A) → 27 (B) → 20 (C), against a teacher that uses it 9 times. Four of the
eight fixes are severity corrections B got wrong:

| Ticket | Teacher | B | C |
|---|---|---|---|
| `"The iOS app crashes immediately upon opening on iOS 18 beta."` | `high`/4 | `medium`/3 | **`high`/4** |
| `"whenever i try to upload files larger than 500MB the browser just closes ... this is blocking my work"` | `high`/4 | `medium`/3 | **`high`/4** |
| `"Hola, tengo un problema con mi envío ... el rastreador dice que está perdido en tránsito ..."` | `high`/4 | `medium`/3 | **`high`/4** |
| `"Is there a discount for non-profit organizations on the enterprise tier?"` | `low`/1 | `medium`/3 | **`low`/1** |

That last one matters: C is the only arm that will say `low`/1 for a genuinely trivial
ticket. It emits `low` 8 times against B's 5 and the teacher's 12.

**3. It pays for that with department, which gets worse, not better.** Department is the
*objective* part of this task — the teacher agrees with itself on it 100% of the time —
and C is the arm that regresses on it: 90.0% (B) → 83.3% (C). All four of C's
regressions against B are department or over-eager de-escalation:

| Ticket | Teacher | B | C |
|---|---|---|---|
| `"Requesting a data export of all our account's usage logs for the last 12 months."` | `technical`/`medium`/3 | **correct** | `general`/`medium`/3 |
| `"Where's my order? Ordered the networking cables on March 15th with 2-day shipping and it never showed up ..."` | `general`/`high`/4 | **correct** | `billing`/`high`/4 |
| `"Quick question - do you support webhook integrations with Zapier? I'm trying to set up automated notifications when my deployments complete ..."` | `technical`/`medium`/3 | **correct** | `sales`/`low`/2 |
| `"Do you offer gluten-free options? My daughter has celiac disease and I want to make sure before I sign up."` | `sales`/`medium`/3 | **correct** | `sales`/`low`/1 |

The pattern is that C spreads department out — `technical` 33 → 23, `general` 9 → 14,
`sales` 5 → 9 — overshooting the teacher's distribution in the opposite direction from B.
B's error was collapsing everything into `technical`; C's is scattering. Net on the field
that has a 100% ceiling: **C is 6.7 points worse than the arm it beats overall.**

**4. Where all three still fail (20 cases) the failure is unchanged**, and it is severity
on unglamorous tickets:

| Ticket | Teacher | A | B | C |
|---|---|---|---|---|
| `"My password reset email is never arriving in my inbox or spam folder."` | `high`/4 | `medium`/3 | `medium`/3 | `medium`/3 |
| `"Getting a 403 Forbidden error when calling the /v2/export endpoint since this morning."` | `high`/4 | `medium`/3 | `medium`/3 | `medium`/3 |
| `"Database replication lag is exceeding 45 minutes on our primary PostgreSQL cluster."` | `critical`/5 | `high`/4 | `medium`/3 | `high`/4 |

Two of those three are the *same tickets* that failed in the recorded 60% run five
sections above, failing the same way against a fresh label, a fresh compile, a fresh
adapter and now a different compiler. Among the 20 cases all arms get wrong, C's error is
on `urgency_score` 17 times and on `priority` 15 times, against `department` only 7 —
the residual failure is severity calibration, and the finetune compiler does not fix it.

### Using paw-kit's own tools for the comparison

The per-case comparison was cross-checked with `paw-test`. Both tools have changed since
the previous version of this section was written (`27a0b56`), and **both changes landed
on exactly the two things this section complained about**, so the workarounds it describes
are no longer needed.

```bash
uv run paw-test compare \
    measurements/finetune_triage_B-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_triage_C-paw-ft-bs48.paw \
    measurements/finetune-triage-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-triage-compare-BC.json

uv run paw-test compare \
    measurements/finetune_triage_A-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_triage_C-paw-ft-bs48.paw \
    measurements/finetune-triage-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-triage-compare-AC.json

uv run paw-test judge measurements/finetune-triage-compare-BC.json \
    --suite measurements/finetune-triage-suite.yaml \
    --out measurements/finetune-triage-judge-BC.json
```

**`paw-test judge` now runs.** The previous version could not use it at all — every case
raised `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`
against `anthropic==1.4.0`, and because `judge_outputs` catches per-case exceptions the
CLI reported `pass rate 0.0% (0/60), errored 60` and **exited 0**, which reads like total
adapter failure rather than total tool failure. The verdicts in the previous version were
obtained by calling paw-kit's own `judge_outputs` with a substituted judge callable. This
run used the shipped CLI directly: **60/60 judged, 0 unparseable, 0 errored, exit 0.**
The companion fix — exiting non-zero when every case errors — is what would have made the
old failure legible; it is not exercised here because nothing errored.

Independent judge verdicts on the B-vs-C compare report: **B 66.7% (40/60), C 76.7%
(46/60)**, 0 unparseable, 0 errored. The judge and the semantic scoring agree on
direction and disagree on size — the judge puts C 10.0 points ahead where teacher
agreement puts it 6.7 ahead. Verdicts differ on 8 of 60 cases: C right where B is wrong on
7, the reverse on 1.

The judge-versus-assertion disagreement block remains the most useful output. Both arms
pass **60/60** on the suite's structural assertions while being 53.3% and 60.0% correct;
the judge overrules an assertion *pass* on 20 cases for B and 14 for C, e.g.
`"Department should be \"billing\" not \"general\"; order/shipping issues are
billing-related"` and `"Technical integration question misclassified as sales; should be
technical department"`. A 60/60 structural pass rate carries no information about whether
the triage is right, and the disagreement list is what makes that visible rather than
reassuring. Note also that the judge's own department opinions conflict with the teacher's
on several of those cases, which is a reminder that it is a second opinion, not ground
truth.

**`compare`'s new "equivalent" count fixes the metric that misled the previous run**, and
the A-vs-C pair demonstrates it precisely:

| Compare | Byte-identical | Equivalent after JSON normalisation |
|---|---|---|
| B vs C | 39/60 | 39/60 |
| A vs C | **0/60** | **35/60** |

Read with the old metric alone, A vs C says "0 identical" — two completely unrelated
programs. In fact they agree on 35 of 60 cases and differ only in JSON separators: A emits
`{"priority":"critical",...}` and C, like B, emits `{"priority": "critical", ...}`,
because both folded specs render their examples with `json.dumps`. B vs C shows the other
half of the point: when two adapters share a whitespace convention the two counts
coincide, and the normalisation costs nothing. The previous version argued a normalized
diff mode "would have turned this run's compare output from misleading into the most
informative table in the section"; it now exists and it does.

Two complaints from the previous version stand unfixed:

1. **The assertion vocabulary still cannot express this task's contract.** `paw-test`
   implements exactly `regex_match`, `max_length`, `min_length`, `exact_match`,
   `not_contains` (`paw_kit/test/suite.py:23`). There is no `one_of`, no `is_valid_json`,
   no schema rule, so "priority is one of four literals and urgency_score is an integer
   1–5" has to be written as a stack of hopeful regexes against the raw string, and "this
   parses into the `Triage` model" cannot be written at all. All three arms pass 60/60 on
   the resulting suite while being 38%, 53% and 60% correct. The semantic scoring in this
   section is therefore done in `scripts/measure_finetune_triage.py`, not in the suite —
   a gap for a toolkit whose whole premise is compiling *typed, structured* outputs.

2. **`paw-kit doctor`'s service check was not predictive in either direction** on the day
   arm C was refused. It warned `gpu_services is empty ... compiles are likely to fail or
   hang`; the fast compiler then compiled twice in ~5 s each. It did *not* warn about
   `redis_unavailable`, listed in the health payload as a rate-limit footnote, which was
   precisely what made async compile impossible. The one check that would have predicted
   that failure was present in the data and not surfaced. Worth noting that today's
   successful compile ran with `gpu_services: {}` in the health payload as well, so that
   warning has now been wrong in both directions on the same task.

A third is new, and it is about this script rather than `paw-test`:
`--skip-compile` silently compiled a *new* adapter when the manifest it was told to reuse
was missing. Given gitignored manifests, that is a live path to reporting numbers from
one adapter under another adapter's program id. Now fixed to raise.

### What compile C actually did — the 2026-09-10 refusal, kept for the record

Earlier the same day, on two attempts ~25 minutes apart:

```
RuntimeError: ProgramAsWeights compile service returned HTTP 503 after 2 attempt(s).
Run `paw-kit doctor` to check service health.
Response: {"detail":{"error":"durable_queue_unavailable",
           "message":"Async compile is unavailable until durable Redis is healthy.",
           "request_id":"5d6d3b52-c06"}}
```

`GET /api/v1/health` returned `{"status":"degraded", "gpu_services":{}, "queue_depth":0,
"warnings":["redis_unavailable: using in-memory global rate limit fallback"]}` throughout.
The request is rejected at queue admission, so no compile was ever queued and no GPU work
was requested. `precheck_compile` on `paw-ft-bs48` succeeded and cheerfully returned
`compiler_snapshot: paw-ft-bs48-20260530` in the same period — **precheck is not a
readiness signal for the finetune path.** (It is not an authentication signal either: an
absent key, a syntactically invalid key and the real key all return an identical 200.
Nothing short of an actual compile tells you whether `PAW_API_KEY` is good.) Note the
retry accounting: `_invoke_compile` retries a 5xx other than 504, so each of the two
attempts cost two HTTP calls.

At 18:03 health returned `{"status":"healthy","version":"0.4.0","gpu_services":{},
"queue_depth":0,"warnings":[]}` and the same command succeeded on the first attempt, no
retry, in 223.3 s. The failure was transient and entirely service-side; nothing in
paw-kit or in this measurement changed between the two.

Compile-endpoint accounting for the successful run: **one real compile** (arm C, 223.3 s,
`cache_hit: false`) and **two cache lookups** (arms A and B, ~1.2 s each, `cache_hit:
true`, returning pre-existing program ids). No GPU compile work was requested for A or B.

### What was expected, and where this differs

Expected, from the phone-extraction result: that a task with 38 points of headroom would
either show `paw-ft-bs48` earning its wall time, or show it failing the same way the fast
compiler does. The answer is a genuine third thing — it improves, by a modest amount, via
a mechanism that is not "understands the task better":

- **The finetune compiler does not inherit the folding pool's label skew.** This is the
  cleanest positive result in the section, and it is a compiler-level difference on a
  byte-identical spec: same 8 examples, same `full_spec_sha256`, and C keeps `critical`
  where B loses it. Folding's documented failure mode in this repo does not reproduce on
  `paw-ft-bs48`.
- **It does not close the gap.** 53.3% → 60.0% against a 91.7% ceiling leaves 31.7 points
  outstanding. The finetune compiler is on the same side of this task's difficulty as the
  fast one.
- **Its gain is a trade, not a lift.** Priority +8.3 and urgency-exact +8.3 are paid for
  with department −6.7, on the one field with a 100% ceiling. An adapter that got
  strictly better would not have that shape.
- **It is free at inference.** 112.7 ms vs 112.4 ms per call. The entire cost is 223 s of
  compile, once — 45x the fast compiler's. Whether that is worth 6.7 points is a
  deployment question with an obvious answer for a batch job and a much less obvious one
  otherwise.
- **The fast compiler's failure on triage is systematic and reproducible**, confirmed
  again: A and B reproduced their recorded outputs 60/60 byte-identical after a cache
  restore.

### Conclusion

**The finetune compiler closed 6.7 of the ~38.3-point gap to the teacher ceiling — about
one sixth of it — and it did not close it anywhere that makes the task work.** Arm C
reaches 60.0% full agreement against arm B's 53.3% and a 91.7% ceiling, leaving 31.7
points outstanding. The gain is concentrated in severity: priority 63.3% → 71.7% (gap to
ceiling 33.3 → 25.0) and urgency-exact 60.0% → 68.3% (gap 31.7 → 23.3), driven by two
specific behaviours — it recovers all three `critical` tickets that folding's skewed
example pool cost arm B, and it is the only arm willing to say `low`/1. Against that, it
*loses* 6.7 points on department (90.0% → 83.3%), the one field where the teacher agrees
with itself 100% of the time and where a competent adapter should be near-perfect; it
scatters `technical` into `general` and `sales` where B collapsed everything into
`technical`. On the 20 tickets all three arms still get wrong, C's errors are
`urgency_score` (17) and `priority` (15) far more than `department` (7) — the residual
failure is severity calibration on ordinary tickets like `"My password reset email is
never arriving"` and `"Getting a 403 Forbidden error ... since this morning"`, both
`high`/4 called `medium`/3 by every arm including this one. So: on a task the fast
compiler fails, the finetune compiler also fails. It costs 223.3 s of compile against
4.94 s (45x), is identical at inference (112.7 ms vs 112.4 ms), and buys a few points that
sit within shouting distance of the label-set noise (6.7 on label 1, 10.0 on label 2,
against ≈3–7 points of teacher-label variation). It is a real improvement and it is not
the improvement this task needs; nothing here supports reaching for `paw-ft-bs48` as a
fix for a task the fast compiler gets wrong.

### Limitations

- **One run, one seed, one machine** (RTX 3080), 60 tickets, one compile per arm. The
  A-to-B gap (≈15 points) is well outside the label-set noise (≈3–7 points); **the B-to-C
  gap (6.7 points) is not comfortably outside it** and should be read as "a few points".
  The direction is corroborated by the second label set (10.0 points) and by the
  independent judge (10.0 points), but the magnitude is not pinned down by this run.
- **The 48 fresh tickets were generated by the same model that labels them**
  (`claude-haiku-4-5-20251001`). The evaluation set is not independent of the labeller:
  tickets may be unrepresentatively easy for this model to classify, and the 91.7%
  "ceiling" is a ceiling on agreement with *this teacher*, not on correctness. The 20
  recorded tickets (hand-written, in the repo since 2026-09-08) are the only part of the
  set free of this.
- **Generation is not temperature 0.** It was tried first: at temperature 0 with a prompt
  varying only in category names, the teacher returned substantially the same tickets
  every batch and deduplication stalled at 24/48 unique after 12 calls. Generation
  therefore runs at default sampling with per-batch product/voice variation and an
  explicit "don't repeat these" list, and is *not* reproducible call-for-call — the
  committed fixture is what makes the run reproducible. All labelling calls, and the 120
  judge calls, are temperature 0.
- **The two teacher labels are not identically prompted.** Two temperature-0 calls with a
  byte-identical prompt would measure the API's determinism, not the task's ambiguity, so
  the second pass reframes the question while asking for the same judgement. The 91.7%
  figure therefore mixes genuine ambiguity with prompt sensitivity and should be read as a
  soft ceiling.
- **Adapter temperature is 0 by inheritance, not by choice.** `programasweights`'
  `PawFunction` defaults to `temperature=0.0`
  (`.venv/.../programasweights/runtime_llamacpp.py:399`);
  `ProgramAsWeightsBackend.infer` accepts no temperature or seed argument and passes only
  `max_tokens`, so paw-kit can neither set nor guarantee it. This applies to `paw-ft-bs48`
  exactly as it does to the fast compiler.
- **Arms A and B were restored from the service's compile cache, not recompiled.** The
  restore is verified 60/60 byte-identical against the recorded run, so this is a strong
  claim rather than an assumption — but it depends on the service returning cached
  programs for unchanged specs, which is service behaviour this project does not control
  and did not previously document.
- **The folding pool is the same 8 examples for B and C.** C's advantage over B on
  `critical` tickets is therefore specific to *this* skewed pool. Whether `paw-ft-bs48` is
  generally more robust to example skew, or happened to be on these eight, is not
  established by one pool.

## Finetune compiler on a rule the base model does not know (fiscal weeks): the first task it can do and the fast compiler cannot

The two sections above end on the same note. Phone extraction: 132 of 134 outputs
byte-identical between the two compilers. Ticket triage: 60.0% vs 53.3% against a 91.7%
ceiling, "on a task the finetune compiler fails, the fast compiler also fails, somewhat
worse ... a task the finetune compiler can do and the fast one cannot is the next thing to
look for." **This is that task.** `paw-ft-bs48` scores 49.0% exact against the fast
compiler's 11.3% and 10.3% — a 4.3x lift, on ground truth computed by code — and gets the
fiscal year right on 298 of 300 dates where the fast compiler gets it right on 256.

It is still not a usable adapter. A frontier model given the same spec text and no compile
at all scores 85.7%, and the exact answer is computable in three lines of Python. Read
this as "the finetune compiler learns a rule the fast compiler cannot represent", not as
"the finetune compiler solves this".

### The rule

Fiscal-week labelling. In, a calendar date; out, `FY<year>-W<nn>`.

- A fiscal year begins on the **first Monday of February** and is named for that calendar
  year. FY2026 begins 2026-02-02.
- Weeks are numbered from 1 starting that Monday. The seven days from the epoch are W01.
- A date in January, or in February before that year's first Monday, belongs to the
  **previous** fiscal year and continues its week count. 2026-01-20 is FY2025-W51.
- A fiscal year has 52 weeks, or 53. In this range only FY2027 has 53 (2027-02-01 →
  2028-02-06); its W53 falls in 2028 and so outside the evaluation window, which is stated
  rather than hidden — **no evaluation case is labelled W53** and the highest reachable
  week is W52.

The task was chosen for three properties, all of which it has and the two prior tasks did
not. The rule is **arbitrary**: no pretrained model knows this particular fiscal calendar,
and the thing it does know — ISO 8601 week numbers, anchored to January — gives a
systematically different answer. The ground truth is **computable**, so there is no judge,
no teacher, no label noise and no ceiling below 100%; every number below is exact and the
evaluation was free. And it needs **real multi-step arithmetic** (locate a weekday-anchored
epoch, subtract, divide, handle the wrap into the previous fiscal year), not
pattern-matching on the input.

Ground truth lives in `scripts/measure_finetune_fiscal.py:fiscal_label` and is unit-tested
in `tests/test_fiscal_week.py` (18 tests): the first Monday of February for 2023–2028
against a calendar, all three days around each of the four in-range boundaries, the whole
of W01, a hand-worked date in each fiscal year, both worked examples that appear inside the
spec text, the FY2027 53-week year including 2028-01-31 → `FY2027-W53` and 2028-02-07 →
`FY2028-W01`, and an exhaustive sweep asserting every date in range yields a well-formed
label with a week inside its year's length.

### The data

```
300 evaluation dates, seed 20260910, 2024-01-01 .. 2027-12-31
  81 of them are within +/-10 days of a fiscal-year start (the full window for all four
     in-range boundaries, oversampled deliberately); 219 drawn uniformly from the rest
  formatted round-robin, 75 each:
     ISO                2026-03-03
     long               March 3, 2026
     day-first          3 Mar 2026
     weekday-prefixed   Tuesday 3 March 2026
  no numeric-only forms: 03/04/2026 is ambiguous and an ambiguous input is not a fair test
  54 of the 300 are dates whose fiscal year differs from their calendar year -- the cases
     the February anchor exists to handle
8 folding examples, disjoint from the evaluation set by construction (their dates are
  removed from the pool before it is drawn), two per format, three of them boundary cases
  including the epoch itself and the Sunday before it
```

Committed as `measurements/finetune-fiscal-dates.json`. It is rebuilt deterministically and
costs nothing to regenerate — no model is involved in producing it.

**The weekday-prefixed format hands the model the weekday for free**, which is exactly the
fact the rule turns on. That is deliberate and the per-format scoring below is there to
show whether any arm exploits it. None does, to any useful degree.

### The commands

```bash
# Fixture is built on first run. Three compiles, one arm at a time, public=False.
uv run python scripts/measure_finetune_fiscal.py --label 3080 --no-run --arms A
uv run python scripts/measure_finetune_fiscal.py --label 3080 --no-run --arms B
uv run python scripts/measure_finetune_fiscal.py --label 3080 --no-run --arms C

# 300 dates x 3 adapters on the 3080 at temperature 0, plus 300 Haiku calls for arm D.
uv run python scripts/measure_finetune_fiscal.py --label 3080 --skip-compile --arms A,B,C,D

# paw-kit's own tools, read-only, on the same 300 cases.
uv run paw-test check measurements/finetune-fiscal-suite-A.yaml --backend real \
    --json measurements/finetune-fiscal-check-A.json          # and -B, -C

uv run paw-test compare \
    measurements/finetune_fiscal_A-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_fiscal_C-paw-ft-bs48.paw \
    measurements/finetune-fiscal-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-fiscal-compare-AC.json

uv run paw-test compare \
    measurements/finetune_fiscal_B-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_fiscal_C-paw-ft-bs48.paw \
    measurements/finetune-fiscal-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-fiscal-compare-BC.json
```

Run artifact: `measurements/finetune-fiscal-3080-20260910-190137.json` (every per-case row
— input, raw output, parsed label, latency — plus each arm's manifest fields mirrored in,
since the `.paw` files are gitignored).

**Compile calls made: three.** Arms A, B and C, one each, all `cache_hit: false`, all
`public=False`. The inference run reports `compiles_made_this_run: 0` because it ran under
`--skip-compile`; `paw-test check` and `paw-test compare` load existing manifests and
compile nothing. Arm D compiles nothing by construction.

### The arms

| Arm | Compiler | Snapshot | Examples | Program id | `full_spec_sha256` | Compile wall |
|---|---|---|---|---|---|---|
| A | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 0 | `a5b5abea4b2f342fb825` | `1b01c29438cbc16a…` | 4.06 s |
| B | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 8 folded | `f42233aa6a4538b5d9b9` | `f745103912be7a2c…` | 4.19 s |
| C | `paw-ft-bs48` | `paw-ft-bs48-20260530` | 8 folded | `547679d668122427359b` | `f745103912be7a2c…` | **133.07 s** |
| D | `claude-haiku-4-5-20251001` | n/a | 0 (spec text only) | n/a | n/a | none |

B and C share `full_spec_sha256` exactly, so **those two arms differ in the compiler and in
nothing else**. A differs only in having no folded examples (its `spec_sha256` and
`full_spec_sha256` are equal, as they must be). No `redis_unavailable` warning and no
refusal this time: health was `{"status":"healthy","queue_depth":0,"warnings":[]}` before
the run and all three compiles went through first attempt. C's 133.07 s is a real,
uncached finetune compile — 32x arm B's, and faster than the 180.8 s and 223.3 s the two
earlier sections recorded.

### Results

All 300 cases, all arms, temperature 0. Ground truth is exact, so 100% is the ceiling and
every column is a real accuracy, not an agreement rate.

| Arm | Exact | Fiscal year correct | Week correct | Week off by exactly 1 | Parses `FY\d{4}-W\d{2}` | Whole output is just the label | Latency/call (median) |
|---|---|---|---|---|---|---|---|
| A — fast, 0 examples | 11.3% (34/300) | 85.3% (256/300) | 11.7% (35/300) | 12.7% (38/300) | 100% (300/300) | 100% | 55.8 ms |
| B — fast, 8 examples | 10.3% (31/300) | 85.0% (255/300) | 10.7% (32/300) | 10.3% (31/300) | 100% (300/300) | 100% | 55.7 ms |
| C — finetune, 8 examples | **49.0% (147/300)** | **99.3% (298/300)** | **49.0% (147/300)** | 35.7% (107/300) | 100% (300/300) | 100% | 55.9 ms |
| D — Haiku 4.5, no compile | **85.7% (257/300)** | 86.7% (260/300) | 86.0% (258/300) | 1.0% (3/300) | 87.0% (261/300) | **0.3% (1/300)** | 3199 ms |

By input format (exact match), and boundary dates versus the rest:

| Arm | ISO | long | day-first | weekday-prefixed | Boundary (81) | Non-boundary (219) | FY ≠ calendar year (54) |
|---|---|---|---|---|---|---|---|
| A | 14.7% | 8.0% | 12.0% | 10.7% | 35.8% (29/81) | 2.3% (5/219) | 7.4% (4/54) |
| B | 8.0% | 8.0% | 12.0% | 13.3% | 30.9% (25/81) | 2.7% (6/219) | 3.7% (2/54) |
| C | 50.7% | 44.0% | 48.0% | 53.3% | 70.4% (57/81) | 41.1% (90/219) | **63.0% (34/54)** |
| D | 82.7% | 89.3% | 88.0% | 82.7% | 87.7% (71/81) | 84.9% (186/219) | 77.8% (42/54) |

**No arm fails on one input format.** The spread across the four formats is 6.7 points for
A, 5.3 for B, 9.3 for C and 6.6 for D — noise at this sample size, and the
weekday-prefixed form (which gives the weekday away) is not reliably the best for anyone.
The task's difficulty is arithmetic, not parsing.

**The boundary oversampling did its job in reverse for the fast compiler.** A and B look
three-and-a-half times better on boundary dates (35.8%, 30.9%) than off them (2.3%, 2.7%) —
but that is not competence. Near a boundary the correct answer is `W01` or `W52`, and both
arms emit `W01` constantly (84 times for A, 61 for B); they collect boundary hits by
standing still. Off the boundary, where the week number has to be computed, arm A is right
**5 times in 219** and arm B **6 times in 219**.

### Error patterns

**1. The fast compiler emits a small fixed vocabulary of week numbers and ignores the
date.** This is the whole story for arms A and B, and it is stark: across 300 distinct
dates, arm A produces **43 distinct labels** and arm B **31**, against arm C's 130 and
Haiku's 142. Arm A uses week 40 ninety-four times and week 1 eighty-four times; those two
values cover 59% of its output. Arm B is worse — week 43 alone, **129 times out of 300**:

| Input | Expected | B |
|---|---|---|
| `29 Jun 2024` | `FY2024-W21` | `FY2024-W43` |
| `October 4, 2024` | `FY2024-W35` | `FY2024-W43` |
| `2027-03-31` | `FY2027-W09` | `FY2027-W43` |
| `Friday 19 July 2024` | `FY2024-W24` | `FY2024-W43` |
| `February 21, 2024` | `FY2024-W03` | `FY2024-W43` |

**W43 and W42 are the folded examples' answers.** The folding pool contains
`December 1, 2024 → FY2024-W43` and `2025-11-20 → FY2025-W42`; arm B's two most common
outputs are W43 (129) and W42 (61), together 63% of its 300 answers. This is the
example-regurgitation failure the folded-examples section above documents for the fast
compiler, reproduced here in its purest form — and it is *why folding made arm B worse than
arm A* (10.3% vs 11.3%). Arm A, with no examples to copy, falls back on W40/W01 instead;
neither is a computation.

**2. The fast compiler ignores the February start; the finetune compiler does not.** Of
the 54 dates whose fiscal year differs from their calendar year, arm A gets the year right
on **10** and arm B on **11**. Arm C gets **53 of 54**. Put the other way: 246 of the 300
dates have fiscal year == calendar year, so a program that ignored the rule entirely and
echoed the calendar year would score 246/300 on the fiscal-year column. Arm A scores 256
and arm B 255 — **the fast compiler's 85% fiscal-year accuracy is almost entirely the
trivial baseline**, worth 10 and 9 dates more than doing nothing. Arm C scores 298. The failure is uniform and mechanical:

| Input | Expected | A | B | C | D |
|---|---|---|---|---|---|
| `Wednesday 29 January 2025` | `FY2024-W52` | `FY2025-W01` | `FY2025-W43` | **`FY2024-W52`** | **`FY2024-W52`** |
| `25 Jan 2027` | `FY2026-W52` | `FY2027-W01` | `FY2027-W01` | **`FY2026-W52`** | *(no label)* |
| `Sunday 1 February 2026` | `FY2025-W52` | `FY2026-W01` | `FY2026-W01` | **`FY2025-W52`** | *(no label)* |
| `29 Jan 2026` | `FY2025-W52` | `FY2026-W01` | `FY2026-W01` | **`FY2025-W52`** | **`FY2025-W52`** |

A's answer for a late-January date is always `FY<calendar year>-W01`: it has read "fiscal
year" as "calendar year" and "January" as "the start". That is the single fact the spec
spends a paragraph and a worked example on, and neither fast-compiler arm picked it up from
prose or from three boundary examples in the folding pool. **Arm C picked it up from both
and applies it to dates it has never seen.** Its `fy_delta` histogram is 298 zeros, one −1
and one +1.

**3. Arm C's residual error is arithmetic, and it is nearly all off-by-one.** C's week
delta distribution: 147 exact, 62 at +1, 45 at −1, 15 at +2, 14 at −2, and a long tail of
16 everything-else. **254 of 300 (84.7%) are within one week of correct, with the fiscal
year also right.** Compare arm A: 73/300 (24.3%) within one week.

| Input | Expected | C |
|---|---|---|
| `February 23, 2025` | `FY2025-W03` | `FY2025-W04` |
| `15 Dec 2025` | `FY2025-W46` | `FY2025-W47` |
| `2026-11-30` | `FY2026-W44` | `FY2026-W43` |
| `3 Nov 2025` | `FY2025-W40` | `FY2025-W39` |

The shape is a slight positive bias (+1 occurs 62 times, −1 45 times), consistent with an
inclusive/exclusive slip in the day count rather than a misunderstanding of the rule. C's
only two fiscal-year errors are the two hardest days in the whole set — the epoch and the
day the epoch is not:

| Input | Expected | C |
|---|---|---|
| `2025-02-04` (Tue, 1 day after the FY2025 epoch) | `FY2025-W01` | `FY2024-W52` |
| `February 3, 2024` (Sat, 2 days before the FY2024 epoch) | `FY2023-W52` | `FY2024-W01` |

**4. Nobody outputs the ISO week.** The hypothesis's named alternative failure — that an
arm would fall back on ISO 8601 week numbers — did not happen. Output week equals the ISO
week on 19/300 for arm A, 8/300 for B, **0/300 for C and 0/300 for D**. The fast compiler
is not computing the wrong week number; it is not computing one at all.

**5. Arm D is a different failure entirely: it is right when it answers and it often does
not answer.** Haiku 4.5 gets the arithmetic right — 258/300 week numbers correct, and of
the 261 cases where it emitted a label at all, **257 were exactly right (98.5%)**. But
**39 of 300 outputs contain no label**, because it reasons its way past `max_tokens=400`:

```
input '25 Jan 2027'  expect FY2026-W52
raw (tail): "...December 2026 = 31 days\n  - January 1-25, 2027 = 25 days\n
             - Total: 25+31+30+31+30+31+31+30+31+30+31+25 = 356 days\n\n
             Total days from Feb 3, 2025 to Jan "

input '2027-11-14'   expect FY2027-W41
raw (tail): "...Total: 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 14 = 287 days\n\n
             Wait, let me recalculate more carefully. From Feb 1 to Nov 14: ..."
```

Truncated outputs average 1052 characters against 748 for ones that finish. And **the spec
says "Output exactly one label and nothing else"; arm D obeys that on 1 case out of 300
(0.3%)**, against 300/300 for every compiled adapter. That is worth stating plainly because
it cuts the other way: on the one axis paw-kit's assertions actually measure — output
shape — the 55 ms adapters beat the 3.2 s frontier model 300–1. Arm D's four wrong-when-
answered cases are `2025-09-08 → FY2025-W31` (expected W32), `26 Jan 2026 → FY2025-W51`
(expected W52), `Thursday 30 January 2025 → FY2024-W51` (expected W52) and
`January 31, 2024 → FY2024-W52` (expected `FY2023-W52`, the only time it missed the
February anchor).

### Using paw-kit's own tools, and what got in the way

`paw-test compare` is the number that settles whether this is a different compile.
**On phone extraction the two compilers agreed byte-for-byte on 132 of 134 outputs. Here
they agree on 31 of 300.**

```
A vs C: 300 cases, 30 identical (byte-for-byte), 30 equivalent, errored 0/300
B vs C: 300 cases, 31 identical (byte-for-byte), 31 equivalent, errored 0/300
```

10.0% and 10.3% agreement, against 98.5% on the earlier task. Whatever `paw-ft-bs48` is doing, on this
spec it is not the fast compiler's program on a slower path.

Three things about the tools, recorded and not fixed:

1. **`paw-test check` reports 100% pass for every arm, including the one that is 10%
   correct.** All three adapters score `Pass rate: 100.0% (300/300)`, exit 0. The suite
   carries the exact ground-truth answer in each case's `expected` field — and the runner
   never reads it. `expected` on a `standard_case` is consumed only by
   `paw_kit/test/active.py:189` to build an active-learning dataset; `paw_kit/test/runner.py`
   evaluates assertions and nothing else (the `exact_match` *rule* at `runner.py:173`
   compares against `rule.value`, a single suite-wide literal, which is useless when every
   case has a different answer). So a suite can contain a complete, correct answer key and
   `paw-test check` will still pass an adapter that is wrong 267 times out of 300. The
   triage section above made the weaker version of this complaint — that the assertion
   vocabulary cannot express the contract. This is the stronger version: **the information
   was present in the suite file and the tool did not use it.**
2. **`paw-test check` has no way to point a suite at a different adapter.** The adapter is
   `adapter_path` inside the YAML and there is no `--adapter` flag, so checking three arms
   on one case set means writing three near-identical suite files
   (`finetune-fiscal-suite-A.yaml`, `-B`, `-C`) that differ in one line. `paw-test compare`
   takes both adapters as arguments and ignores the suite's `adapter_path` entirely, so the
   two subcommands disagree about where an adapter comes from.
3. **`paw-kit doctor`'s service check warned again and was again not predictive.** It
   reported `WARN — 200 OK but gpu_services is empty`, with the note that the fast compiler
   has been observed to compile anyway. All three compiles then succeeded, including the
   finetune one. That is now three sections in a row where `gpu_services: {}` did not
   predict anything; the check's own remediation text says as much.

A fourth, about this script rather than the tools: `--skip-compile` with a missing manifest
raises rather than compiling, copied from `scripts/measure_finetune_triage.py`'s guard for
the reason that section gives — `.paw` files are gitignored, and a run that reports itself
as `--skip-compile` while quietly compiling a different adapter is worse than one that
stops.

### What the hypothesis predicted, and what happened

The hypothesis: *the fast compiler is a single forward pass from spec text to adapter
weights and can only produce adapters of a kind its training covered, while the finetune
compiler generates examples from the spec with a teacher and trains on them, so it should
win on an explicit, arbitrary procedure stated in the spec that the base model does not
already know.*

**Confirmed, on the part that matters, and by a wide margin.** 49.0% vs 11.3%/10.3% exact;
99.3% vs 85.3%/85.0% on the fiscal year; 84.7% vs 24.3% within one week; 130 distinct
labels vs 43 and 31; 31/300 byte-identical outputs against the 132/134 that made the
phone-extraction comparison inconclusive. On a rule the base model does not know, the
finetune compiler learns it and the fast compiler does not represent it at all.

Two predictions inside the hypothesis were wrong. **No arm fell back on ISO week numbers**
(0/300 for C and D, 19/300 and 8/300 for A and B, at chance). And **folding examples into
the spec did not help the fast compiler even slightly** — arm B is 1.0 point *worse* than
arm A and collapses onto the folded examples' own answers 63% of the time. The eight
examples that taught arm C the February anchor taught arm B a constant.

### Conclusion

**On a task defined by a rule the base model does not know, the finetune compiler does
something the fast compiler cannot do at all — and the result is still not a usable
adapter.** `paw-ft-bs48` scores 49.0% exact (147/300) against the fast compiler's 11.3%
and 10.3%, a 4.3x lift on ground truth computed by code with no judge and no label noise.
The difference is not a few points traded between fields, as it was on ticket triage: it
is categorical. Arm C locates the fiscal year on 298 of 300 dates and on 53 of the 54 dates
where the February anchor actually bites, where arms A and B manage 10 and 11 of 54 and
answer `FY<calendar year>-W01` for every late-January date. Arm C produces 130 distinct
labels for 300 distinct dates; arm A produces 43 and arm B produces 31, of which W43 alone
— the answer to one of the eight folded examples — accounts for 129. Arm C's remaining
error is arithmetic and small: 84.7% of its answers are within one week, against 24.3% for
arm A. And unlike phone extraction, where 132 of 134 outputs were byte-identical between
the two compilers, here 31 of 300 are — this is demonstrably a different program, not a
slower path to the same one. Against that: 49.0% exact is a coin flip, `claude-haiku-4-5`
given the identical spec and no compile at all reaches 85.7% (98.5% of the times it
manages to answer within its token budget), and the true answer is three lines of Python.
Nobody should ship a 49%-accurate date labeller. The finding is not "use `paw-ft-bs48` for
this"; it is that **the finetune compiler's 133 s of compile buys a genuine capability
difference — learning an arbitrary stated procedure — that the fast compiler's 4 s does
not buy at any example count, and this is the first measurement in this document that
separates them.** Per-call latency is unchanged at 55.9 ms vs 55.7 ms, as in both earlier
sections: everything `paw-ft-bs48` costs, it costs once, at compile.

### Limitations

- **One task, one run, one seed, one machine** (RTX 3080), one compile per arm. The
  A-vs-C gap (38 points) is far too large to be a sampling artefact at n=300, but the
  precise figures are one run's.
- **The task was designed to separate the compilers**, after two tasks that did not. That
  is the honest framing: this is evidence that a separating task *exists* and what it looks
  like, not evidence about how often real work has this shape. A task whose whole content is
  an arbitrary stated procedure is the best case for a compiler that trains on
  spec-generated examples.
- **W53 is never tested.** FY2027 is a 53-week year but its W53 falls in 2028, outside the
  evaluation range, so no arm was asked for a `W53` label and the "52 or 53" clause of the
  rule is exercised only by the unit tests. Arm A emitted `W53` three times anyway, always
  wrongly — `December 23, 2027 -> FY2027-W53` where the answer is `FY2027-W47`, and twice in
  December 2026 where FY2026 has only 52 weeks.
- **Arm D is a reference, not a ceiling.** Ground truth is exact, so the ceiling is 100%.
  Arm D's 85.7% is depressed by a `max_tokens=400` budget it exceeds on 39 cases; a larger
  budget, or an explicit "answer only" instruction, would raise it. It is included to show
  the rule is hard, not to bound anything.
- **The comparison between arm D and the adapters is not like-for-like** in cost or in
  shape. D is a 3.2 s API call per input that emits a paragraph of arithmetic; A, B and C
  are 56 ms local calls that emit exactly the label. On output conformance the adapters win
  300–1.
- **Adapter temperature is 0 by inheritance, not by choice**, exactly as in the two
  sections above: `programasweights`' `PawFunction` defaults to `temperature=0.0` and
  `ProgramAsWeightsBackend.infer` accepts no temperature or seed argument.
- **The folding pool is the same 8 examples for B and C.** How much of C's advantage comes
  from those particular eight (three of which are boundary cases) rather than from the
  compiler is not separable by one pool — though C's 41.1% on the 219 non-boundary dates,
  against B's 2.7%, is hard to attribute to three boundary examples.

## Finetune compiler on an arbitrary lookup table (region codes): the gap is not about arithmetic

The fiscal-week section above found the first task the finetune compiler can do and the
fast one cannot — but on a rule whose whole content is multi-step arithmetic, which left
two readings open. Either the gap is about **computation** (the fast compiler's single
forward pass cannot produce an adapter that *computes*, and would be fine on a rule with
no arithmetic), or it is about **spec-defined mappings the base model lacks** (anything
stated in the spec and absent from pretraining, arithmetic or not).

This section separates them with a task that has no arithmetic anywhere: an arbitrary
region-code lookup table. **It is the second reading.** `paw-ft-bs48` scores 97.7%
(293/300) against the fast compiler's 33.0% and 29.0%, on a task where every answer is a
single dictionary lookup and a frontier model scores 100%. Arm A never emits two of the
six codes at all — 100 of the 300 cases are unwinnable by its own output vocabulary — and
arm B, given the eight folded examples, learns those eight countries (87.5%) and drops
*below chance* on the other twenty-two (7.7%, against 16.7% for guessing).

Unlike fiscal weeks, this one also produces a **usable** adapter: 97.7% exact at 32 ms a
call, against 100% at 750 ms for `claude-haiku-4-5`, with 293 of the 300 outputs
byte-identical to Haiku's.

### The rule

Thirty real countries are assigned to six made-up internal region codes, five countries
each, by a seeded shuffle. In, a short sales-system sentence naming exactly one of the
thirty; out, that country's code.

```
RG-K7  Brazil, Vietnam, Thailand, Egypt, Sweden
RG-M2  Kenya, Poland, Peru, Mexico, Malaysia
RG-Q9  Norway, Chile, Australia, Nigeria, Indonesia
RG-T4  Japan, Finland, Denmark, Argentina, Philippines
RG-V1  Ireland, Turkey, India, Colombia, Greece
RG-X6  Portugal, New Zealand, Canada, Morocco, Hungary
```

The grouping is deliberately not geographic, not alphabetical and not anything a
pretrained model could infer — Brazil sits with Vietnam and Sweden, Japan with Argentina.
The spec states the rule, prints the table as a plain list one country per line, and gives
two worked examples (Portugal and Kenya, which are therefore given away to every arm).

The task was chosen to keep everything the fiscal-week task had **except** the arithmetic.
The rule is arbitrary; the ground truth is computable, so there is no judge, no teacher and
no label noise; and the answer is a fixed six-character string. What it removes is
computation: `scripts/measure_finetune_lookup.py:region_code` is a `dict` subscript.

### The data

```
300 evaluation sentences, seed 20260910
  30 countries x 10 templates, so every country appears exactly 10 times and every
     template exactly 30 times -- both cuts are balanced by construction, no sampling noise
  10 templates, including:
     "Ship this order to Osaka, Japan."
     "Customer billing address is in Kenya."
     "Invoice for the Rosario office (Argentina) attached."
     "Our reseller in New Zealand needs the report."
     "Our Chiang Mai office handled the call, but the customer is in Denmark."   <- misleading city
     "Routed via Izmir, final delivery Indonesia."                               <- misleading city
     "please route the shipment to morocco."                                    <- lower case
  60 of the 300 name a city belonging to a *different* country than the one to look up
  30 of the 300 write the country name in lower case
  the country name always appears verbatim -- no demonyms, no abbreviations, so the
     difficulty is the table and not the parsing
8 folding examples, covering 8 different countries and all 6 codes, built from four
  phrasings that appear nowhere in the evaluation set, so the two sets are disjoint by
  construction. The eight folded *countries* do appear in the evaluation set, 10 times
  each -- which is what makes "does this arm only know the folded countries?" answerable,
  and it turned out to be the whole story for arm B.
```

Folded: Sweden (RG-K7), Poland (RG-M2), Australia (RG-Q9), Philippines and Denmark
(RG-T4), India and Turkey (RG-V1), Portugal (RG-X6). Committed as
`measurements/finetune-lookup-regions.json`; rebuilt deterministically, no model involved.

Guessing at random scores 16.7% (each code is the right answer exactly 50 times in 300).
That is the number to compare the fast compiler against, not zero.

### The commands

```bash
# Fixture is built on first run. Three compiles, one arm at a time, public=False.
uv run python scripts/measure_finetune_lookup.py --label 3080 --no-run --arms A
uv run python scripts/measure_finetune_lookup.py --label 3080 --no-run --arms B
uv run python scripts/measure_finetune_lookup.py --label 3080 --no-run --arms C

# 300 sentences x 3 adapters on the 3080 at temperature 0, plus 300 Haiku calls for arm D.
uv run python scripts/measure_finetune_lookup.py --label 3080 --skip-compile --arms A,B,C,D

# paw-kit's own tools, read-only, on the same 300 cases.
uv run paw-test check measurements/finetune-lookup-suite-A.yaml --backend real \
    --json measurements/finetune-lookup-check-A.json          # and -B, -C

uv run paw-test compare \
    measurements/finetune_lookup_A-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_lookup_C-paw-ft-bs48.paw \
    measurements/finetune-lookup-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-lookup-compare-AC.json

uv run paw-test compare \
    measurements/finetune_lookup_B-paw-4b-qwen3-0.6b.paw \
    measurements/finetune_lookup_C-paw-ft-bs48.paw \
    measurements/finetune-lookup-suite.yaml \
    --backend real --no-fuzz --json measurements/finetune-lookup-compare-BC.json
```

Run artifact: `measurements/finetune-lookup-3080-20260910-192159.json` (every per-case row
— input, raw output, parsed code, latency — plus each arm's manifest fields mirrored in,
since the `.paw` files are gitignored).

**Compile calls made: three.** Arms A, B and C, one each, all `cache_hit: false`, all
`public=False`. Health before the run was `{"status":"healthy","queue_depth":0,
"warnings":[]}` — no `redis_unavailable` — and all three compiles went through on the
first attempt, including the async finetune one. `paw-kit doctor` was clean apart from the
familiar `WARN — 200 OK but gpu_services is empty`, which for the fourth section running
predicted nothing. The inference run reports `compiles_made_this_run: 0` (it ran under
`--skip-compile`); `check` and `compare` compile nothing; arm D compiles nothing.

### The arms

| Arm | Compiler | Snapshot | Examples | Program id | `full_spec_sha256` | Compile wall |
|---|---|---|---|---|---|---|
| A | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 0 | `ab2f646b90aee687127d` | `8bb378d8f95378ed…` | 3.86 s |
| B | `paw-4b-qwen3-0.6b` | `paw-4b-qwen3-0.6b-20260407` | 8 folded | `a60100720070b1ffdee4` | `72a918df07342585…` | 4.04 s |
| C | `paw-ft-bs48` | `paw-ft-bs48-20260530` | 8 folded | `be5fc2427459c392556b` | `72a918df07342585…` | **96.08 s** |
| D | `claude-haiku-4-5-20251001` | n/a | 0 (spec text only) | n/a | n/a | none |

B and C share `full_spec_sha256` exactly, so **those two arms differ in the compiler and in
nothing else**. A differs only in having no folded examples (its `spec_sha256` and
`full_spec_sha256` are equal, as they must be). C's 96.08 s is a real, uncached finetune
compile — 24x arm B's, and the fastest finetune compile recorded in this document (against
133.07 s, 180.8 s and 223.3 s in the three earlier sections).

### Results

All 300 cases, all arms, temperature 0. Ground truth is exact, so 100% is the ceiling.
"Exact" is scored on the region code parsed out of the output, so an arm that answers
correctly but decorates the answer is not penalised twice — that lands in the next column
instead.

| Arm | Exact | Whole output is just the code | Distinct codes emitted | Top code's share | Countries whose modal answer is right | Latency/call (median) |
|---|---|---|---|---|---|---|
| A — fast, 0 examples | 33.0% (99/300) | **0% (0/300)** | **4 of 6** | 38.3% | 9/30 | 39.3 ms |
| B — fast, 8 examples | 29.0% (87/300) | **0% (0/300)** | 6 of 6 | 24.3% | 8/30 | 40.8 ms |
| C — finetune, 8 examples | **97.7% (293/300)** | 100% (300/300) | 6 of 6 | 17.7% | **30/30** | **32.2 ms** |
| D — Haiku 4.5, no compile | **100% (300/300)** | 100% (300/300) | 6 of 6 | 16.7% | 30/30 | 750.1 ms |

Random guessing scores 16.7%. Every arm parsed as a code on 300/300 and produced a code
from the six-code vocabulary on 300/300; there were no errors and no refusals in 1200
calls.

Folded countries against the rest, and the deliberately hard slices:

| Arm | Folded 8 countries (80) | Other 22 countries (220) | Misleading city (60) | Plain (240) | Lower case (30) |
|---|---|---|---|---|---|
| A | 31.2% (25/80) | 33.6% (74/220) | 36.7% (22/60) | 32.1% (77/240) | 33.3% (10/30) |
| B | **87.5% (70/80)** | **7.7% (17/220)** | 26.7% (16/60) | 29.6% (71/240) | 30.0% (9/30) |
| C | 96.2% (77/80) | 98.2% (216/220) | 98.3% (59/60) | 97.5% (234/240) | 93.3% (28/30) |
| D | 100% (80/80) | 100% (220/220) | 100% (60/60) | 100% (240/240) | 100% (30/30) |

By template (exact matches out of 30 each):

| Template | A | B | C | D |
|---|---|---|---|---|
| `Ship this order to {city}, {country}.` | 12 | 10 | 30 | 30 |
| `Customer billing address is in {country}.` | 9 | 8 | 29 | 30 |
| `Invoice for the {city} office ({country}) attached.` | 9 | 10 | 27 | 30 |
| `Our reseller in {country} needs the report.` | 9 | 9 | 30 | 30 |
| `Warehouse transfer: destination {country}.` | 10 | 8 | 30 | 30 |
| `Our {other city} office handled the call, but the customer is in {country}.` | 11 | 8 | 30 | 30 |
| `please route the shipment to {country in lower case}.` | 10 | 9 | 28 | 30 |
| `Support ticket opened by a customer in {country}; escalate…` | 9 | 8 | 30 | 30 |
| `Routed via {other city}, final delivery {country}.` | 11 | 8 | 29 | 30 |
| `We are opening a second depot in {country} next quarter.` | 9 | 9 | 30 | 30 |

**No arm is beaten by a phrasing.** The spread across ten templates is 3 cases for A, 2 for
B and 3 for C. The misleading city mostly does not mislead: A answers the *city's*
country's code on 7 of its 60 misleading cases and C on 11, against 10 expected under a
uniform guess and 8.6 and 10.0 under each arm's own output distribution. **Arm B is the one
exception** — 18 of 60, against 10.3 expected under its own marginal. That is 8 cases and
the only trace of geographic pull anywhere in the run, so it is reported rather than
leaned on. Lower case costs nothing for A and B and two cases for C. As on fiscal weeks,
the difficulty is the rule, not the input.

### Error patterns

**1. Arm A can only say four things.** Across 300 sentences naming 30 different countries,
arm A emits `RG-M2` 115 times, `RG-V1` 78, `RG-X6` 59 and `RG-K7` 48 — and **`RG-Q9` and
`RG-T4` zero times each**. Ten of the thirty countries are in those two regions, so 100 of
the 300 cases were unwinnable before the first token: arm A scored **0 of 100** on them.

| Input | Expected | A |
|---|---|---|
| `Ship this order to Osaka, Japan.` | `RG-T4` | `"RG-K7"` |
| `Customer billing address is in Norway.` | `RG-Q9` | `"RG-X6"` |
| `Routed via Tampere, final delivery Norway.` | `RG-Q9` | `"RG-K7"` |
| `We are opening a second depot in Brazil next quarter.` | `RG-K7` | `"RG-M2"` |
| `Support ticket opened by a customer in Japan; escalate to the regional desk.` | `RG-T4` | `"RG-V1"` |

This is the fiscal-week collapse in a new shape. There it was a fixed vocabulary of *week
numbers* (43 distinct labels for arm A, of which W40 and W01 covered 59% of outputs); here
it is a fixed vocabulary of *codes*, two of the six simply missing. Arm A is not reading
the table. It is not entirely ignoring the input either — 33.0% is twice the 16.7% chance
rate, and it gets eight countries right 10 times out of 10 (Greece, India, Kenya, Mexico,
New Zealand, Peru, Portugal, Thailand) — but its modal answer is correct for only **9 of
30** countries, and 14 countries it gets wrong every single time.

**2. Folding turns the fast compiler into a memoriser of exactly the folded rows — and
nothing else.** This is the cleanest demonstration of it anywhere in this document. Arm B
gets **70 of 80** on the eight folded countries and **17 of 220** on the other twenty-two.
17/220 is 7.7%: **below the 16.7% you get by guessing.** The eight examples did not teach
arm B the table; they taught it eight answers and actively degraded everything else, which
is why arm B is 4 points *worse overall* than arm A (29.0% vs 33.0%) despite being the arm
that was given the answers.

| Input | Expected | B | C |
|---|---|---|---|
| `Warehouse transfer: destination Sweden.` (Sweden is folded) | `RG-K7` | **`"RG-K7"`** | `RG-K7` |
| `Our reseller in Portugal needs the report.` (Portugal is folded) | `RG-X6` | **`"RG-X6"`** | `RG-X6` |
| `We are opening a second depot in Greece next quarter.` | `RG-V1` | `"RG-K7"` | `RG-V1` |
| `Invoice for the Rosario office (Argentina) attached.` | `RG-T4` | `"RG-K7"` | `RG-T4` |
| `please route the shipment to morocco.` | `RG-X6` | `"RG-K7"` | `RG-X6` |
| `Invoice for the Rio de Janeiro office (Brazil) attached.` | `RG-K7` | `"RG-X6"` | `RG-K7` |

**3. Denmark: the folded example arm B ignored, and arm A never had a chance at.** Denmark
is in the folding pool (`Our partner office in Denmark raised the request.` → `RG-T4`) and
is one of the ten countries whose code arm A never emits. All ten Denmark sentences, all
four arms:

| Input | A | B | C | D |
|---|---|---|---|---|
| `Ship this order to Aarhus, Denmark.` | `"RG-K7"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `Customer billing address is in Denmark.` | `"RG-M2"` | `"RG-K7"` | `RG-X6` | **`RG-T4`** |
| `Invoice for the Aarhus office (Denmark) attached.` | `"RG-X6"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `Our reseller in Denmark needs the report.` | `"RG-K7"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `Warehouse transfer: destination Denmark.` | `"RG-M2"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `Our Chiang Mai office handled the call, but the customer is in Denmark.` | `"RG-M2"` | `"RG-M2"` | **`RG-T4`** | **`RG-T4`** |
| `please route the shipment to denmark.` | `"RG-K7"` | `"RG-K7"` | `RG-X6` | **`RG-T4`** |
| `Support ticket opened by a customer in Denmark; escalate to the regional desk.` | `"RG-V1"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `Routed via Osaka, final delivery Denmark.` | `"RG-M2"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |
| `We are opening a second depot in Denmark next quarter.` | `"RG-X6"` | `"RG-K7"` | **`RG-T4`** | **`RG-T4`** |

The expected answer is `RG-T4` on every row. Arm B answers `"RG-K7"` on nine of ten —
a country whose answer was **handed to it in the spec text** — and arm A produces four
different wrong codes without once landing on the right one. Arm C gets 8 of 10.

**4. Arm C's seven errors, in full.** There is no pattern worth a paragraph, which is
itself the finding — 293 right, and the misses do not cluster by code, by country group or
by whether the country was folded (3 of the 7 are folded countries):

| Input | Expected | C |
|---|---|---|
| `Invoice for the Surabaya office (Indonesia) attached.` | `RG-Q9` | `RG-V1` |
| `Routed via Izmir, final delivery Indonesia.` | `RG-Q9` | `RG-V1` |
| `Invoice for the Arequipa office (Peru) attached.` | `RG-M2` | `RG-V1` |
| `Invoice for the Penang office (Malaysia) attached.` | `RG-M2` | `RG-X6` |
| `Customer billing address is in Denmark.` | `RG-T4` | `RG-X6` |
| `please route the shipment to denmark.` | `RG-T4` | `RG-X6` |
| `please route the shipment to turkey.` | `RG-V1` | `RG-K7` |

Four of the seven are two countries seen twice (Indonesia, Denmark) and three of the seven
sit in the `Invoice for the {city} office ({country}) attached.` template — the one that
puts the country in parentheses after a city. C's output distribution is near-uniform
(53/52/51/48/48/48 against a ground truth of exactly 50 each), which is the opposite of a
collapse.

**5. The fast compiler quotes its answer; the finetune compiler does not.** Every one of
arm A's 300 outputs and every one of arm B's arrives wrapped in double quotes — `"RG-X6"`,
not `RG-X6`. Every one of arm C's 300 and arm D's 300 is the bare code. The spec says "Output exactly one region code
and nothing else: … No explanation, no punctuation". **Arms A and B satisfy that on 0 of
300 cases; arms C and D on 300 of 300.** This exactly reverses the fiscal-week result,
where every compiled adapter was 300/300 on output shape and Haiku managed 1/300. It also
has teeth: it is the entire reason `paw-test check` scores arms A and B at zero (below),
and it means a caller doing `json.loads` and a caller doing `==` on this adapter's output
disagree about whether it works.

### Using paw-kit's own tools, and what got in the way

**A note on which code ran.** The measurement script imports `paw_kit` from this worktree
(`8b57b8d`). `paw-test` is a console script and resolves to the editable install in the
shared checkout, whose `main` moved to `952b4db` — *"fix(test): paw-test check now compares
output to expected"* — while this run was in flight. So the `check` numbers below are from
the **fixed** `check`, not the one the fiscal-week section complained about.

```
paw-test check, --backend real, 300 cases, at paw_kit 952b4db:
  arm A   passed 0/300     correct against expected   0/300
  arm B   passed 0/300     correct against expected   0/300
  arm C   passed 293/300   correct against expected 293/300
```

**The fix works, and it immediately found something my own scoring hides.** `check` now
reads each case's `expected` and compares it to the output under
`paw_kit/test/matching.py:values_equivalent`, so arm C's seven failures are exactly the
seven errors listed above — the tool and the script agree case for case. Arms A and B score
**0/300**, not 33.0% and 29.0%, because `values_equivalent('"RG-M2"', 'RG-M2')` is false:
`"RG-M2"` parses as JSON, `RG-M2` does not, so the function falls through to a whitespace
comparison of two different strings. Both numbers are defensible — the spec did say "no
punctuation" — but they are 33 points apart on the same 300 outputs, so **which one you
quote has to be stated, and this section quotes the parsed-code one as the headline and
the 0/300 here.**

`paw-test compare` inherits the same normalisation and the same blind spot, which makes its
headline number useless on this pair:

```
A vs C: 300 cases, 0 identical (byte-for-byte), 0 equivalent, A pass 300/300, C pass 300/300, errored 0/300
B vs C: 300 cases, 0 identical (byte-for-byte), 0 equivalent, B pass 300/300, C pass 300/300, errored 0/300
```

**0 of 300 equivalent is true and misleading.** Scoring the parsed code instead, A and C
give the same answer on **101 of 300 (33.7%)** and B and C on **88 of 300 (29.3%)** — real
disagreement, and comparable to the 31/300 the fiscal-week section recorded, but nothing
like zero. The `pass 300/300` on both sides is the older complaint restated: the suite's
assertions can only constrain the *shape* of the output, and `"RG-M2"` matches
`regex_match: RG-[A-Z]\d` just as well as `RG-M2` does.

The one comparison that needed no tooling at all is the most striking: **arm C and arm D
produce byte-identical output on 293 of 300 cases (97.7%)**. A 96-second compile and a
32 ms local call reproduce `claude-haiku-4-5` exactly, on every case but seven.

Three things recorded and not fixed:

1. **`values_equivalent` does not unwrap a JSON scalar against a bare string.** Two
   adapters that agree on every answer read as 0% equivalent if one of them quotes. The
   docstring says "both parse as JSON to equal values, or (when they don't both parse as
   JSON) equal after `normalize_whitespace`" — the missing case is *one* side parsing to a
   JSON string whose value equals the other side. This is now load-bearing in two places
   (`compare`'s equivalence and `check`'s `expected`), so it is worth a decision rather
   than a default.
2. **`paw-test check` still has no `--adapter` flag.** The adapter is `adapter_path` inside
   the YAML, so checking three arms on one case set still means three near-identical suite
   files (`finetune-lookup-suite-A.yaml`, `-B`, `-C`) differing in one line, while
   `paw-test compare` takes both adapters as arguments and ignores the suite's
   `adapter_path`. The fiscal-week section raised this; `952b4db` did not touch it.
3. **`paw-kit doctor`'s `gpu_services is empty` WARN was again not predictive.** Fourth
   section in a row: all three compiles succeeded, the finetune one included.

### What the fiscal-week section predicted, and what happened

The fiscal-week section's hypothesis was that the fast compiler is a single forward pass
from spec text to adapter weights and can only produce adapters of a kind its training
covered, while the finetune compiler trains on teacher-generated examples and so can learn
an arbitrary procedure stated in the spec. It confirmed that on a task made of arithmetic,
and left open whether **arithmetic** was the operative word.

**It was not.** Strip the arithmetic out entirely — leave a rule whose execution is one
dictionary lookup — and the fast compiler still cannot represent it, in the same two ways
it failed on fiscal weeks:

- **it collapses to a fixed output vocabulary** — there, 43 distinct labels with W40/W01
  covering 59%; here, four of six codes, with two never emitted at all across 300 calls;
- **folding examples in makes it worse, by regurgitation** — there, `FY2024-W43` (a folded
  example's own answer) 129 times out of 300 and a 1.0-point drop; here, 87.5% on the eight
  folded countries against 7.7% on the rest and a 4.0-point drop.

Two things came out differently from the fiscal-week run. **The finetune compiler does not
merely win here, it succeeds**: 97.7% against 49.0% there, because there is nothing left to
get arithmetically wrong once the mapping is learned. And **the output-shape result
inverts**: on fiscal weeks the adapters obeyed "output only the label" 300/300 and the
frontier model 1/300; here the fast compiler's adapters obey it 0/300 (they quote) and the
frontier model 300/300.

One prediction of mine was wrong. The misleading cities were built to catch an arm doing
geographic association rather than table lookup, and **almost nothing was caught**: arms A
and C answer the mentioned city's country's code no more often than their own output
distributions predict, and the misleading templates are arm A's two best. Arm B does it 18
times in 60 against 10.3 expected — the single hint that anything in this run is doing
geography, and small enough at n=60 to be worth no more than a sentence. Whatever the fast
compiler is doing, it is mostly not reasoning about geography either.

### Conclusion

**The finetune compiler's advantage over the fast compiler is not about arithmetic; it is
about whether a mapping stated in the spec makes it into the adapter at all.** On a task
with no computation in it whatsoever — thirty countries, six arbitrary codes, the table
printed in the spec, the answer a single lookup — `paw-ft-bs48` scores 97.7% (293/300)
while `paw-4b-qwen3-0.6b` scores 33.0% with no examples and 29.0% with eight, against a
16.7% chance baseline. The fast compiler fails the same two ways it failed on fiscal weeks:
arm A collapses to four of the six codes and never once emits the other two, making a third
of the evaluation set unwinnable by construction, and arm B — handed eight worked answers —
learns those eight countries (87.5%) and falls *below chance* on the remaining twenty-two
(7.7%), scoring worse overall than the arm that got no examples. That reproduces the
fiscal-week regurgitation result on a task that shares none of its content, which is what
makes it a generalisation rather than a second anecdote: the gap is spec-defined mappings
the base model lacks, arithmetic or not. What is new here is that the finetune compiler
does not just win, it *works*: 97.7% exact, all six codes emitted at near-uniform
frequency, every one of the thirty countries right more often than not, and 293 of 300
outputs byte-identical to `claude-haiku-4-5`'s — which scores 100% on the same spec and
takes 750 ms a call against the adapter's 32 ms. Ninety-six seconds of compile buys a 23x
faster program that agrees with a frontier model, character for character, on 293 of 300
inputs. (Whether `claude-haiku-4-5` is in fact `paw-ft-bs48`'s teacher is not something
this measurement can see; the agreement is an observation, not a lineage claim.) Against
that: the task is one a frontier model finds trivial, 97.7% is still not 100% and the seven
misses are silent, the fast compiler's adapters quote their output on 300 of 300 calls so
`paw-test check` scores them zero rather than a third, and this is one task, one seed, one
run. The honest headline is narrow and worth having: **the fast compiler cannot put an
arbitrary spec-stated table into an adapter, at any example count, and the finetune
compiler can.**

### Limitations

- **One task, one run, one seed, one machine** (RTX 3080), one compile per arm. The
  A-vs-C gap (65 points) is far too large to be sampling noise at n=300, but the precise
  figures are one run's.
- **The task was designed to isolate one variable** — an arbitrary stated mapping with the
  arithmetic removed — after the fiscal-week task confounded the two. That is the honest
  framing: this is evidence about what the separating property *is*, not evidence about
  how often real work has this shape.
- **Two of the thirty countries are given away in the spec's worked examples** (Portugal →
  `RG-X6`, Kenya → `RG-M2`), identically for all four arms. Both arms A and B get both of
  them right 10/10, which is 20 of their 99 and 87 correct answers — strip them and A falls
  to 28.2% and B to 23.9%.
- **The folding pool uses four phrasings the evaluation set never repeats.** That is what
  makes the two sets disjoint by construction, and it means arm B's 87.5% on folded
  countries is transfer of an *answer* across a phrasing change, not sentence memorisation
  — but it also means no arm was tested on a folded sentence verbatim.
- **`W53`-style unreachable cases have no analogue here** — every code is reachable and
  every country is tested 10 times — but the flip side is that the evaluation is perfectly
  balanced in a way real traffic would not be.
- **Arm D is a reference, not a ceiling.** It scored 100%, so on this task the reference
  and the ceiling coincide, which they did not on fiscal weeks. `max_tokens` was 30 rather
  than 400: the answer is six characters, so the truncation failure that cost arm D 39
  cases in the fiscal-week section is structurally impossible here.
- **The comparison between arm D and the adapters is not like-for-like** in cost. D is a
  750 ms API call per input; A, B and C are 32–41 ms local calls.
- **Adapter temperature is 0 by inheritance, not by choice**, exactly as in the three
  sections above: `programasweights`' `PawFunction` defaults to `temperature=0.0` and
  `ProgramAsWeightsBackend.infer` accepts no temperature or seed argument.
- **`check` and `compare` ran against a different commit of `paw_kit` than the measurement
  script did** — `952b4db` versus this worktree's `8b57b8d` — because `paw-test` resolves
  to the editable install. The adapters, the fixture and the 1200 inference calls are
  unaffected; only the tool-output numbers in the tools section come from `952b4db`.

## Constrained decoding against the real upstream adapter: the hook wasn't missing

Every claim in this project about grammar-constrained decoding has carried the same
caveat: it has never been applied to a real *compiled PAW adapter*.
`measure_schema_real_model.py` proved the FSM works, but against a separate HuggingFace
model. `ProgramAsWeightsBackend.infer()` warns it "cannot apply grammar_constraint at
decoding time: the upstream SDK exposes no grammar/logits hook." `RealPAWBackend` — the
placeholder where masking was supposed to eventually live — raises `NotImplementedError`.
The conclusion drawn from that, in this document and in `README.md`, was that paw-kit
would need its own in-process PyTorch runtime before `paw.schema` could reach a real
adapter.

**That conclusion was wrong, and it was wrong about the SDK's *public* surface only.**
The upstream runtime is a hand-rolled decode loop over `llama-cpp-python`
(`programasweights/runtime_llamacpp.py:490-500`). It holds a real `llama_cpp.Llama` at
`PawFunction._llm` and calls `_llm.sample(temp=...)` once per token.
`llama_cpp.Llama.sample()` accepts both `logits_processor=` and `grammar=`. The SDK just
never passes either. The hook is not missing — it is behind a private attribute.

`scripts/measure_constrained_decoding_upstream.py` injects one. It asks the
phone-extractor adapter (compiled to emit a bare string like `(555) 666-7777`) for a
Pydantic object it was never trained on, with the regex produced by paw-kit's own
`pydantic_to_regex`:

```python
class Contact(BaseModel):
    area_code: int
    number: str
    kind: Literal["mobile", "landline", "unknown"]
```

| | Valid `Contact` parses | Latency (5 runs) |
|---|---|---|
| Unconstrained (SDK as shipped) | **0/5** | mean 85 ms (range 13–160 ms) |
| `RegexLogitsProcessor` injected | **4/5** | median 3.2 s; 47 s on the first call, 334 ms on the two runs that revisited only cached FSM states |

The two columns are not a like-for-like latency comparison: unconstrained output here is a
~10-token bare string, constrained output a ~30-token JSON object. See the warm-up analysis
below for where the constrained time actually goes.

> **Corrected 2026-09-09, in Phase F review.** This table first reported "74 ms" and "334 ms
> warm" as *mean latency*. Neither is supportable from the artifact. The unconstrained
> samples are `[159.8, 86.8, 84.9, 80.9, 12.9]` — mean 85.0, median 84.9; **no aggregation
> of them yields 74**. The constrained samples are `[46816, 334, 3215, 334, 8819]`, and the
> measurement script's own printed warm statistic (`constrained[1:]`) is **3175 ms** — the
> "334 ms" reported was the best two of five, an order of magnitude below the script's own
> output, silently dropping the two runs (3.2 s and 8.8 s) that hit new FSM states. Same
> class of transcription error as the 93.3%→92.5% correction logged earlier in this file,
> and left visible here for the same reason.

```
'Office line: +1-555-666-7777'
   unconstrained -> '(555) 666-7777'
   constrained   -> '{ "area_code": 555, "number": "666-7777", "kind": "mobile" }'
```

A program compiled to emit a bare phone string was forced into a JSON schema it has no
training signal for, with the digits still correctly extracted from the input. Nothing
but token-level masking can do that, so this is not ambiguous: **paw-kit's constrained
decoding does work against the real upstream backend.** No in-process PyTorch runtime is
required for it, which was the single strongest argument for building one.

**The per-token masking cost is fine; the warm-up cost is not.** Instrumented over the
same run:

| | |
|---|---|
| numpy masking | 0.68 s over 1719 tokens = **0.40 ms/token** |
| `get_allowed_tokens` | **50.85 s** across 30 distinct FSM states |

Masking itself comes in comfortably under the "<2ms per-token overhead" budget in
`roadmap.md` — that claim survives contact with a 151k-token vocabulary. Essentially all
the cost is `get_allowed_tokens`, which walks the whole vocabulary the first time it sees
each FSM state: ~1.7 s per new state, 30 states for this schema, and the first call pays
most of it (46.8 s, vs 334 ms for a later call that revisits only cached states).

That is a **warm-up** cost, not a per-token one, and in principle it is fully
precomputable: the state → allowed-token-set map depends only on (regex, vocabulary),
both of which are fixed before any input arrives. Today it is recomputed from scratch in
every process, and `RegexLogitsProcessor`'s LRU caches are in-memory only. Persisting
that map per (schema, vocabulary) is the obvious fix and is not attempted here.

**Fragility, stated plainly.** This reaches into `PawFunction._llm` — a private attribute
of a third party's object — and monkeypatches `sample`. It is unsupported and can break
on any upstream release, with no deprecation contract to rely on. The script fails loudly
with a specific message if `_llm` ever stops being there, but that is detection, not
protection. **Nothing in this section is shipped in `paw_kit/`**, and it should not be
copied there as-is. The defensible near-term ask is upstream-facing: `PawFunction.__call__`
takes `(input_text, max_tokens, temperature)` and could take `logits_processor` and pass
it through to the `sample()` call it already makes. That is a small, additive upstream
change that would turn this from a private-attribute hack into a supported integration —
and it is a concrete thing to open an issue about rather than a reason to build a
competing runtime.

**What the 0/5 → 4/5 does and does not measure.** The unconstrained `0/5` is *definitional
rather than measured*: this adapter was compiled to emit a bare phone string, so it could
never have parsed as `Contact`, and the baseline was guaranteed before the run started. That
is the point of the probe — it shows masking can impose a schema the adapter was never
trained on — but the number does not travel with that context, so do not quote it as an
accuracy improvement. More pointedly: **all four constrained outputs answer `kind:
"mobile"`**, including for `"Office line: +1-555-666-7777"`, which is a landline by any
reading. That field is not being driven by the input at all. Masking made the output
*shaped*, and left one field unanchored — structural validity is the whole of what 4/5
claims.

**The 5th case is a real limitation, not a rounding error.** Given `"no phone number here
at all"`, the constrained adapter emits `{"area_code": 0, "number": ""` and stalls: the
model wants to stop, the FSM will not accept a terminator until the schema is satisfied,
and it exhausts the token budget mid-object. This is the same family as the forced-binary
`"neutral"` leakage logged elsewhere in this document — a schema with no representable
"not applicable" case, meeting input that has no valid answer. Constrained decoding makes
the output *shaped*; it cannot make it *answerable*.

**Scope limits**: one adapter, one schema, one machine, one run. The comparison is
structural validity only — no semantic judging was run on the constrained outputs, so
"4/5 parse" says nothing about whether `kind: "mobile"` is the right classification.

> **Note, 2026-09-15 (Track 15).** The hook is no longer private and this experiment is no
> longer how the mechanism is reached. `programasweights==0.4.6` (PR #6) made
> `logits_processor` a public kwarg on the callable, `ProgramAsWeightsBackend.infer()`
> passes it, and `scripts/measure_constrained_decoding_upstream.py` — which
> monkeypatched `PawFunction._llm.sample` — is **deleted**. The section's own "defensible
> near-term ask" ("a small, additive upstream change that would turn this from a
> private-attribute hack into a supported integration") is what shipped.
>
> Three of its numbers do not carry forward, because the engine changed:
>
> - **4/5 is now 5/5.** Re-run on the same phone-extractor adapter with the same five
>   inputs through the public hook and the byte-level engine, every case parses as
>   `Contact`, including `"no phone number here at all"` — which stalled mid-object here.
>   It now completes as `{"area_code": 0, "number": "", "kind": "unknown"}`. The
>   limitation the section describes was an artefact of that engine exhausting its token
>   budget, not of constrained decoding as such; the *semantic* point it was making (a
>   schema with no representable "not applicable" case) is unaffected, and `"unknown"` is
>   this schema's version of that case.
> - **The warm-up cost is gone.** `get_allowed_tokens`' 50.85 s across 30 FSM states has
>   no counterpart: the replacement engine is lazy and builds no state→allowed-token map
>   at all. Per-call matcher construction is 1.3 ms.
> - **The `kind: "mobile"` observation stands and is worth keeping.** Masking made the
>   output shaped and left one field unanchored. That is the whole of what any parse-rate
>   number here claims, then and now.

## Shadow mode, for real: the gate holds, the audit window is noisier than the design says

`docs/shadow-mode.md` makes four claims that had never been run on hardware: a
60%-agreement adapter never promotes at the shipped defaults and stalls after five
windows; `audit_rate=0.05` costs one teacher call per twenty served calls (~400 served
calls per audit window) and `audit_rate=0.0` costs none; "nothing in shadow mode ...
adds latency to [the caller]"; and shadow mode creates no new file.
`scripts/measure_shadow_mode.py` measures all four in one run
(`measurements/shadow-mode-3080-20260910-124735.json`, 160 s wall, RTX 3080 + CUDA).

> **Correction, 2026-09-11.** Everything below was run against the 20-ticket dataset
> whose "60% agreement" figure is corrected above to **46.7% held out** (60% conflated
> 5 tickets the adapter had memorised via its own spec with 15 genuinely scored ones).
> This section's window tables are an empirical replay of that exact recorded sequence —
> they are historical facts about that specific run, not recomputed, and still read 0.60
> on the cyclic draw because the sequence itself hasn't changed. What *is* recomputed
> below is the theoretical binomial arithmetic, which depends on the adapter's true
> agreement rate as an input: at the corrected p=0.467 (rather than the leaked 0.6), the
> chance of one lucky 16-of-20 window drops from **5.10%** to **0.25%**, and the chance of
> promoting within the five-window stall drops from **23.0%** to **1.2%** — the gate's
> real-world conclusion (this adapter should not and does not promote) gets *stronger*,
> not weaker, at the honest rate. A second, independent error in the same paragraph is
> also fixed below (B-8c): the "0.40 to 0.80 in 90% of draws" claim was never a 90%
> interval at p=0.6 — it's a 96.3% one. And **the audit-cost section's headline number is
> replaced**: "20 teacher calls over 621 served calls" was a stopping-time ratio (how long
> until 20 audits complete), not a measured rate, and reads as 3.2% against a configured
> 5% — a fixed-1,200-served-call re-run (§16 item 6, `shadow-mode-3080-b7fix-20260911-172502.json`)
> gives the honest rate: **68 teacher calls over 1,200 served calls, 5.67%**, against the
> configured 5%, with the old figure kept alongside as `calls_to_first_completed_window`.

**Setup, and what is real in it.** The adapter is the *same* compiled adapter that scored
60% in the section above (**46.7% held out** — see the correction) —
`measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw`,
run live through `ProgramAsWeightsBackend` on the GPU, ~113 ms per call. The teacher is a
**replay teacher**: a pure function returning the recorded live-Claude answer for each of
the 20 recorded tickets in `measurements/triage-semantic-agreement-3080-20260909-002033.json`.
No teacher API call is made and no money is spent, but the labels are real teacher labels.
Agreement uses the shipped recipe for that measurement's own scoring rule,
`field_tolerance_agreement({"urgency_score": 1})`. **Zero compiles**: the adapter already
exists, so each task is put into `shadow` by calling `TraceDB.set_shadow_started` with the
existing manifest and `threshold=10**9` makes the compile path unreachable;
`backend.compile` is wrapped in a counter for the whole run and the JSON records
`"compile_calls": 0`. As a check that replaying the teacher does not quietly change what is
being compared, the live adapter's answer was compared with the recorded one on all 20
tickets: **20/20 identical**, so the live agreement rate is exactly the recorded 12/20.

### 1. Does the 60% adapter promote at the shipped defaults? No, and it stalls on schedule

200 calls, `shadow_window=20`, `shadow_threshold=0.8`, `audit_rate=0.0`, one comparison
drained to completion per call so nothing is dropped.

| Window | Cyclic inputs | Random draw (seed 0) | State after window |
|---|---|---|---|
| 1 (calls 1-20) | 0.60 (12/20) | 0.60 | `shadow` |
| 2 (calls 21-40) | 0.60 | 0.65 | `shadow` |
| 3 (calls 41-60) | 0.60 | 0.65 | `shadow` |
| 4 (calls 61-80) | 0.60 | 0.55 | `shadow` |
| 5 (calls 81-100) | 0.60 | 0.60 | `shadow` |
| stall warning | call 101 | call 101 | `shadow (stalled)` |

Never promoted, in either draw. The stall WARNING fired on call 101 in both — exactly
`5 × shadow_window` comparisons — and `get_agreement()["stalled"]` and the `(stalled)`
marker in `paw-kit report` both flipped to true. Fail-opens: 0. Dropped: 0. This is the
design working as documented.

Two things the design's own text gets wrong, though, and neither is visible without
running it:

- **The residual is ~9x larger than the doc states.** `docs/shadow-mode.md:66` says a
  genuinely random 60% adapter "has roughly a 2.5% chance of producing one 16-of-20 window"
  within the first five windows. The exact binomial (in the JSON as `binomial_residual`) is
  **5.10% per window and 23.0% across five windows**. 2.5% is roughly what you get for a
  *coin-flip* adapter (p=0.5 gives 0.59% per window, 2.9% over five), which looks like the
  number that was actually computed. The mitigation the doc offers — raise `shadow_window`
  — is still the right one, but a reader budgeting risk off "2.5%" is off by an order of
  magnitude. This is arithmetic, not a measurement; anyone can check it.
- **A stalled task's reported agreement is not a window rate.** After the stall the runner
  keeps only one comparison in `shadow_window`; here 94 of the 100 post-stall comparisons
  were dropped by that subsample, and `get_agreement()["rate"]` then read **0.55** (and
  0.75 in the random-draw run) because its trailing 20 countable rows straddle the dense
  pre-stall and sparse post-stall samples. Every completed window was 0.60. The rate
  `paw-kit report` shows after `(stalled)` is a sparse estimate with a much wider error bar
  than the `(11/20)` next to it implies.

One structural caveat on this experiment specifically: with 20 recorded tickets cycled
against a `shadow_window` of 20, every window contains exactly the same inputs, so the
per-window rate is 0.60 by construction and the binomial residual above is *zero* — that
run cannot promote by luck. The random-draw variant is the one where luck is in play, and
it moved between 0.55 and 0.65 without coming near 16/20.

### 2. What the audit path costs, and what it can actually detect

This adapter cannot reach `ready` at the shipped `shadow_threshold=0.8` — that is section 1
— so to measure the served path at all, **`shadow_threshold` was lowered to 0.5** (and
`demote_threshold` to 0.4, which the validator requires to stay strictly below it). That is
a knob turned to make promotion reachable, not a recommendation. It promoted on the first
window at 0.60, after 20 calls.

| | `audit_rate=0.05`, stopping time | `audit_rate=0.05`, **fixed N (corrected)** | `audit_rate=0.0` |
|---|---|---|---|
| Served calls after promotion | 621 | **1,200 (fixed in advance)** | 200 |
| Teacher calls on those | **20** | **68** | **0** |
| Teacher calls per served call | 0.032 | **0.0567 (5.67%, vs. configured 5%)** | 0.000 |
| Served calls to complete one 20-sample audit window | 621 | *(not this experiment's question)* | never (0 audit rows) |
| `traces` rows written while `ready` | 0 | 0 | 0 |
| Caller p50 / p95 (ms) | 112.8 / 114.0 | *(not remeasured)* | 113.2 / 114.0 |

> **Correction, 2026-09-11.** The "0.032" / "3.2%" figure in the first column was always a
> **stopping-time ratio** — the served-call count needed to collect 20 audit samples — not
> a measured audit *rate*. Read as a rate against the configured 5%, it understates the true
> cost by more than a third, because 20 ÷ (a Bernoulli(0.05) stopping time) is a biased
> estimator of 0.05 at small sample counts. The fixed-N re-run (§16 item 6,
> `shadow-mode-3080-b7fix-20260911-172502.json`) fixes the served-call count in advance
> (1,200) and counts however many teacher calls land in it — 68, **5.67%**, matching the
> configured 5% far more closely than 3.2% did. The 621-served-calls figure is kept, renamed
> to what it actually measures: calls to the first *completed* window, a genuinely different
> and still useful quantity (it tells you how long you wait for the first audit signal, not
> what the audit costs on an ongoing basis).

`audit_rate=0.0` really is free: zero teacher invocations across 200 served calls, zero
`shadow_pairs` rows, and demotion is unreachable, exactly as documented. The `traces` row
count confirms the other storage claim — a promoted task persists no input text on the
served path.

The doc's "about 400 served calls per completed audit window" is the right expectation
(20 ÷ 0.05) but a wide one: the number of served calls needed to collect 20 Bernoulli(0.05)
samples has mean 400 and standard deviation ~87, and **this run needed 621** (two earlier
trial runs of the same script, not recorded here, needed 363 and 368). Budget the audit
window as "a few hundred to a thousand served calls", not 400.

**The finding that contradicts the design: at `audit_window=20`, the audit is not a drift
detector.** A 20-sample window drawn from an adapter whose true agreement is 0.60 reads
anywhere between 0.40 and 0.80 in 90% of draws<sup>‡</sup>. The audit windows actually observed on this
adapter, whose true rate is exactly 0.60: **0.75** (this run, experiment 2), **0.65** (this
run, experiment 2c), 0.55 and 0.48 in the two trial runs. Two consequences:

- The shipped `demote_threshold=0.6` fires on a strict `<`, so an adapter sitting at exactly
  0.60 is on the boundary and demotes only when the window happens to read low — P = 40.4%
  per completed audit window. Experiment 2c forced a promotion (leaving
  `shadow_threshold` at the shipped 0.8), set `demote_threshold=0.61` so the mechanism
  *should* fire, and ran `audit_rate=0.5`: 20 audit comparisons over 29 served calls, window
  read **0.65**, **no demotion**. The demotion path is reachable — an earlier trial run of
  the identical code demoted at 0.48 — but at this window size whether it fires on a given
  window is close to a coin flip.
- Combined with the several-hundred served calls a window costs, a real regression would be
  detected slowly and unreliably. `docs/shadow-mode.md` describes `audit_window` only as
  "comparisons per audit window after promotion" and offers no guidance on sizing it; on
  this evidence, a drift signal you would act on needs a window several times larger than
  the shipped 20, and the doc should say so.

<sup>‡</sup> **Correction, 2026-09-11.** "0.40 to 0.80 in 90% of draws" is not a 90% interval
at n=20, p=0.6 — it's a **96.3%** one (exact binomial `P(8 ≤ X ≤ 16) = 0.9630`). The
tightest range actually covering ~90% is `[9,15]`, i.e. **0.45 to 0.75** (89.25% coverage —
discreteness means no integer range hits exactly 90%). The point of the paragraph — that a
20-sample window is noisy relative to `demote_threshold=0.6` — is unaffected; the stated
band was simply wider than "90%" should mean. The four observed windows (0.75, 0.65, 0.55,
0.48) still straddle the corrected band about as they straddled the original one.

### 3. Caller-path latency: the claim holds, but the cost lands somewhere else

Per-call wall time of the wrapped function, 200 calls per state, caller looping flat out
with the shadow worker running (`audit_rate=0.0`, so `ready` is the bare served path):

| State | p50 (ms) | p95 (ms) | Comparisons queued | Dropped |
|---|---|---|---|---|
| `tracing` | 3.64 | 3.80 | n/a | n/a |
| `shadow` | 0.93 | 3.01 | 200 | **190** |
| `ready` | 113.07 | 114.02 | 0 | 0 |

`ready` is 113 ms because that is what the adapter's own GPU inference costs; it is not
shadow-mode overhead. The interesting row is `shadow` coming out **four times faster than
`tracing`**, which is impossible as a description of the code — a wrapper in `shadow` does
everything the `tracing` wrapper does plus one `put_nowait`. Experiment 3b is the control:
the same wrapper, in `tracing`, measured twice, with an artificial background thread running
adapter inferences during the second block and nothing else changed.

| Same wrapper, same `tracing` state | p50 (ms) | p95 (ms) |
|---|---|---|
| Idle machine | 3.68 | 3.82 |
| Background GPU load | 0.92 | 3.07 |

That reproduces the whole effect (0.92 vs the 0.93 measured in `shadow`). The machine's CPU
governor is `powersave`: the caller's ~3.7 ms is a SQLite trace write on a down-clocked
CPU, and any background work — the shadow worker included — clocks the CPU up and the same
write costs ~0.9 ms. **Read the result as: shadow mode's caller cost is a bounded
`put_nowait`, below this measurement's noise floor, and the wrapped call's cost in both
`tracing` and `shadow` is dominated by the trace write and by CPU clock state.** The claim
"adds no caller latency" survives; the specific numbers say more about `powersave` than
about `paw_kit`.

**What the no-latency design actually costs is samples.** The queue is bounded at
`shadow_queue_size=8` and drops the newest job when full. With a replay teacher (~0 ms) and
a 113 ms adapter, the caller outruns the worker immediately: the queue-full WARNING fired on
call **10**, and **190 of 200** comparisons were dropped — 10 recorded, 9 still pending at
the end of the phase. A dropped comparison is not a disagreement, it is simply not sampled,
so nothing is corrupted; but a window of 20 then needs ~400 calls rather than 20, and this
is invisible from `paw-kit report`, which has no dropped column (only the in-process
`get_agreement()["dropped"]` shows it). The real ratio is friendlier than this test — a live
Claude teacher is ~11x *slower* than this adapter (see the JIT section above), so the worker
keeps up easily — but any task whose teacher is faster than its adapter (a cache hit, a
local model, a cheap API) will silently sample only a few percent of its calls.

### 4. Sanity: `paw-kit report`, file modes, and files created

`paw-kit report --db <cache>/traces.db` on the stalled experiment-1 database, verbatim from
the JSON:

```
                                          paw-kit task report
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Task            ┃ State            ┃ Calls ┃ Agreement              ┃ Fail-open ┃ Promoted ┃ Demoted ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ 62a33b58c0e0... │ shadow (stalled) │   200 │ 0.55 (11/20) shadow/20 │         0 │ -        │ -       │
└─────────────────┴──────────────────┴───────┴────────────────────────┴───────────┴──────────┴─────────┘
```

(the promoted experiment-2 database renders `ready`, `0.75 (15/20) audit/20`, with a
`Promoted` timestamp — note its `Calls` column reads **20**, not the 641 calls that task
actually took: `Calls` is the traced-call count, and a promoted task records no traces, so
the column freezes at whatever it read on promotion.) Every one of the seven per-experiment cache directories came out
identical: mode `0700`, containing `traces.db`, `traces.db-wal` and `traces.db-shm`, **all
three mode `0600`**. The `-wal`/`-shm` pair is SQLite's WAL journal, present before shadow
mode existed; shadow mode itself added **no file**, as documented — its two tables live
inside `traces.db`. `paw-kit report` exited 0 on all seven and created nothing.

### Limitations

- **Replay teacher.** The teacher's answers are the recorded ones from the 2026-09-09 run,
  not fresh calls. Real teacher-side inconsistency — which the 60% figure itself warns
  about — is invisible here, and so is any drift over time.
- **Repeated inputs.** The same 20 tickets are cycled (experiments 1, 2, 3) or resampled
  with replacement (1b). The adapter is deterministic at temperature 0, so per-input
  agreement is fixed: what these numbers exercise is the **window arithmetic**, not fresh
  traffic. With cyclic inputs and `shadow_window=20` the windows are perfectly correlated,
  which is why experiment 1's five windows are identical to two decimal places.
- **The replay teacher is instantaneous and free**, which inverts the real latency ratio and
  is what drives the 190/200 drop rate in experiment 3.
- **Promotion had to be bought.** Experiment 2 lowered `shadow_threshold` to 0.5; experiment
  2c and experiment 3's `ready` phase called `TraceDB.try_promote` directly. All three are
  stated in the JSON (`shadow_threshold_used`, `promotion_forced`) and none of them is the
  shipped behaviour.
- One machine (RTX 3080, CUDA, `powersave` governor), one run, one adapter, one spec, one
  process. Every rate here is a single draw from a distribution with a standard deviation of
  about 0.11.

## Constrained decoding on the backend users run: the null is the result, and the cost was not where it looked

`scripts/measure_constrained_decoding.py` replaces both deleted constrained-decoding
scripts. It drives the shipped path — `ProgramAsWeightsBackend.infer(adapter, input,
grammar_constraint=pydantic_to_regex(model, anchors=False))`, which is the same call
`paw_kit.schema.load` makes — through the public `logits_processor` hook that arrived in
`programasweights==0.4.6`. Artifact:
`measurements/constrained-decoding-3080-20260915-185048.json`.

**Setup, and what is real in it.** RTX 3080, `offline=True`, `n_gpu_layers=-1`,
`programasweights` 0.4.6, `llguidance` 1.8.0, `llama-cpp-python` 0.3.19, numpy 2.5.2,
vocabulary 151,936 tokens, `INITIAL_LEXER_FUEL=10,000`. Two real compiled adapters,
`measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw` (whose recorded spec's
categories match the `Literal` `Triage` exactly) and
`measurements/phone_extractor-paw-4b-qwen3-0.6b.paw`. **Both are `.paw` manifests present
on this machine only** — `*.paw` is gitignored, so a fresh clone needs a paid compile for
any part of this, as for this document's whole existing evidence base. **Zero paid calls,
zero teacher calls, no network.** Inputs: the 15 `TICKETS` moved out of the deleted
`measure_schema_real_model.py`, and the committed 60-ticket double-labelled fixture
`measurements/finetune-triage-tickets.json`. Instrumentation is two wrappers that capture
and time the constraint object `infer()` builds; neither changes what `infer()` does.

### 1. Constrained versus unconstrained, on an adapter compiled for its own schema

| | 15 `TICKETS` | 60-ticket fixture |
|---|---|---|
| byte-identical pairs | **15/15** | **60/60** |
| discordant pairs | **0** | **0** |
| valid against the `Literal` `Triage` (both arms) | 15/15 | 60/60 |
| values inside the `Literal` sets (both arms) | 15/15 | 60/60 |
| emitted-token counts equal across arms | 15/15 | 60/60 |

**The null is the result, not a failed measurement.** On an adapter compiled for its own
schema there is nothing for masking to fix, and it fixes nothing. That is the number that
belongs beside "masking guarantees shape only": it is what the guarantee is worth when the
model was already going to produce the right shape.

Because the arms are byte-identical, **every per-arm statistic is identical by
construction**, so the artifact records agreement with the committed teacher labels
**once**: full exact (all three fields) against `teacher_label_1` is **31/60 (51.7%)**, and
against `teacher_label_2` **30/60**. That is a property of this adapter and carries no
information whatsoever about masking. It is not a per-arm comparison and must never be
quoted as one. (Note for anyone re-deriving it: "32/60" is a *different* statistic — the
urgency-within-1 figure — and the two have been confused once already.)

### 2. Evidence the constraint is applied at all

Section 1 cannot distinguish an applied constraint from an inert one, so the artifact
records two signals only a working constraint can produce, and a control where the
contrast still exists.

| | |
|---|---|
| constrained calls | 85 |
| calls where the processor ran on every generation step | **85/85** |
| calls with a vacuous masking record (processor never invoked) | **0** |
| masking steps | 1,878 |
| steps that masked **zero** logits | **0** |
| masked per step, triage arms | min 151,570, mean 151,809.0, max 151,935 of 151,936 |
| masked per step, `Contact` arms | min 5,016, mean 120,638 (phone) / 112,022 (triage) |

A note on the first row, because the obvious phrasing is wrong: the processor invocation
count is **one more** than the emitted-token count on an EOS-terminated call. The SDK's
decode loop calls `sample()` once per iteration and breaks *without emitting* when the
sampled token is EOS, so EOS is generated under the mask and then discarded. "Invocations
== generated tokens" would fail on every normal call; what is asserted is the relation.
The emitted count comes from the model's own KV-cache position, not from re-tokenizing the
returned string.

**Positive controls**, where the contrast survives:

| Control | Constrained | Unconstrained |
|---|---|---|
| `Contact` on the phone extractor, 5 inputs | **5/5** valid `Contact` | **0/5** (bare strings like `(555) 666-7777`, and `''` on the no-number input) |
| `Contact` forced onto the triage adapter, 5 tickets | **5/5** valid `Contact` | **0/5** (well-formed *Triage* JSON) |

The second control is the one to read carefully. Its constrained outputs are valid and
semantically poor — `{"area_code": 1, "number": "INV-9821", "kind": "unknown"}` for a
double-billing ticket — and that is the documented evidence for what masking does to
fail-open: a wrong-but-well-formed answer no longer fails validation, so it no longer
reaches the teacher. Whitespace as emitted varies between runs; these are illustrative.

### 3. The byte-level property, measured where it can be measured

At the matcher, on `Contact` (which has a free `str` field, unlike `Triage`), under
llama.cpp's own tokenizer and the vocabulary object the backend built for the loaded
model:

| State | Result |
|---|---|
| at a string-content state | **51/51** lone UTF-8 lead bytes admitted; **0/64** lone continuation bytes; closing `"` allowed (the string may end here) |
| after consuming one lone lead byte (`0xC3`) | allowed set collapses to **101** tokens: **64/64** continuation bytes, **0** lead bytes, closing `"` **forbidden** |
| after the character completes (`0xC3 0xA9` = `é`) | closing `"` allowed again |
| after a complete object | allowed set is exactly `{151645}` (EOS); `is_accepting` and `is_stopped` both true |

This is the direct evidence for the claim that the replacement engine closes the hole the
character-level FSM had: UTF-8 well-formedness inside a string field is structural here,
not bolted on.

End to end, a **`U+FFFD` scan** of every returned string in every arm: **0 of 170**
contain the replacement character.

**What that scan can and cannot prove**, stated because the stronger check is the one a
reader will assume was made. It can show that no returned string contains the character
the SDK substitutes for bytes it could not decode. It **cannot** show the generated bytes
were well-formed UTF-8: the SDK returns `output_bytes.decode("utf-8", errors="replace")`,
so paw-kit never sees raw token bytes, and a Python `str` is valid Unicode by
construction — a round-trip check on the returned value *cannot fail* and would be
recording nothing. It is not airtight in the other direction either: a grammar that
legitimately admitted the replacement character's own bytes would false-positive. **A
byte-exact end-to-end check is unavailable** without the SDK exposing its output tokens,
and none was made.

### 4. Per-token cost — and the reason the first run of this script failed its own budget

| | |
|---|---|
| processor time per invocation | **0.640 ms** (median 0.714, max 2.180, n=945) |
| end to end, constrained | **132.4 ms** per call |
| end to end, unconstrained | **115.7 ms** per call |
| emitted tokens per call | 20 |
| **overhead per generated token** | **0.834 ms** |
| `roadmap.md` Milestone 2a budget | `<2 ms` per token — **met** |
| per-call constraint build | 1.301 ms |
| `llguidance` tokenizer build | 258.93 ms, **once per model** |

**The first run of this script missed that budget by 6.5x, and the engine was not the
reason.** As first measured: 377.1 ms per constrained call against 114.8 ms unconstrained,
**13.118 ms per generated token**. The cause was `build_constraint` rebuilding the
`llguidance` `LLTokenizer` on every call — **249.13 ms of a 251.65 ms build**, against
1.35 ms for the matcher and its initial mask and 0.01 ms for `grammar_from_regex`. That
object walks the whole 151,936-token table and depends only on the vocabulary, never on
the pattern or on any parse state; the design always counted it as a per-model cost
(*"0.553 s per model: 0.293 s detokenize + 0.260 s tokenizer"*), and the code paid it per
call.

Why nobody caught it earlier, since this is the kind of thing that should have shown up in
review: the four Phase 0 rounds that measured cost built the tokenizer **once, outside
their own probe loops**, and drove their own processor rather than `build_constraint`.
They were measuring the engine, and the engine was fine. The shipped path's cost had never
been measured until this script ran. The fix caches the tokenizer on the `Vocabulary`,
which the backend already builds once per model and evicts with it; the **matcher** stays
strictly per call, because an `LLMatcher` dies on error. The as-found numbers are kept in
the artifact's `findings` section.

### What this section does and does not show

- It **does** show that the shipped backend applies the mask on every generation step of
  every call measured, that the byte-level property holds at the matcher, and that the
  cost is inside the project's published budget.
- It **does** show that on an adapter compiled for its own schema, masking changes
  nothing — and that where an adapter is asked for a schema it was not compiled for,
  masking produces valid output with wrong values.
- It **does not** show anything about semantic correctness, in either direction. No number
  here compares the arms on rightness, and the arms are byte-identical, so no such number
  could exist on this instrument.
- It **does not** prove UTF-8 well-formedness end to end. See section 3.
- **Scope limits**: two adapters, two schemas, one machine, one run, greedy decoding
  (`temperature=0.0` by SDK default). The 60-ticket fixture's own limitation — its fresh
  tickets were generated by the same teacher model that labels them — is recorded with the
  fixture and applies to the agreement figure in section 1.

> **Note, 2026-09-15 (Track 15, post-Phase-4 correction).** After this artifact was
> recorded at `INITIAL_LEXER_FUEL=10,000` (the "Setup" line above), Phase 4's own
> re-derivation of `grammar.py`'s budget comment found that value refusing an ordinary
> schema at construction, on every call: four `Field(ge=0, le=255)` int fields (an
> ordinary schema, not a pathological one) cost 10,334 fuel and were refused by the old
> 10,000 bound — a Track-D money-leak instance. `INITIAL_LEXER_FUEL` was raised to
> **100,000** the same day (`paw_kit/schema/constraint.py`'s module docstring carries
> the full re-derivation: bounded-int fields, `Literal` size, bare-alternation size, and
> construction time, all re-measured against the real 151,936-token vocabulary).
>
> **The per-token and per-call figures in sections 2 and 4 above are unaffected.** The
> fuel budget bounds *grammar construction size*, not per-token masking work — nothing
> in `_Constraint.__call__` (the per-step masking path measured in section 2's table and
> section 4's `processor time per invocation`) reads `initial_lexer_fuel` at all. And for
> an ordinary schema like `Contact` (this section's own free-string positive control),
> raising the ceiling costs nothing at construction either, because a schema's minimum
> fuel requirement depends on its own pattern, not on the ceiling above it: `Contact`
> needs only 763 fuel to construct, so it never approached either ceiling. Measured
> directly, matcher construction time (build + initial mask, real vocabulary, 5-run
> minimum) for `Contact` is **1.40 ms at `initial_lexer_fuel=100,000`** against 1.39 ms
> at the old 10,000 — statistically indistinguishable, and nowhere near the 1.301 ms
> `per-call constraint build` this section's own section-4 table already recorded at the
> old budget. This is an append, not an edit in place, per this file's own convention
> for dated corrections; the artifact JSON itself is unchanged and still records
> `initial_lexer_fuel: 10000`, which is what was actually run.

## Reproducing

```bash
uv sync --extra real                 # pulls the upstream SDK from PyPI
export PAW_API_KEY=paw_sk_...        # https://programasweights.com/settings; compile only

uv run python scripts/measure_real_backend.py examples/date_normalizer/suite.yaml \
    --compiler paw-4b-qwen3-0.6b --calls 50 --label your-machine-name

# semantic correctness of a suite compiled with a terse, docs-style spec (no folded examples)
uv run python scripts/measure_semantic_correctness.py measurements/spec-drafts/spec-1-json-repair.yaml \
    --compiler paw-4b-qwen3-0.6b --label your-machine-name

# re-score the JIT-speedup ticket-triage adapter against a fresh, independent teacher call
uv run python scripts/measure_triage_semantic_agreement.py --label your-machine-name

# constrained decoding against a real compiled adapter (needs no PAW_API_KEY --
# runs offline against an already-cached program)
uv run python scripts/measure_constrained_decoding_upstream.py --label your-machine-name

# shadow mode: promotion, stalling, audit cost and caller latency, against the recorded
# 60%-agreement triage adapter and a replay teacher (makes zero compile calls and zero
# teacher API calls; needs the program already in the SDK cache)
uv run python scripts/measure_shadow_mode.py --label your-machine-name
```

> **Note, 2026-09-15 (Track 15).** The constrained-decoding line in the block above is
> **dead**: `scripts/measure_constrained_decoding_upstream.py` is deleted, along with
> `scripts/measure_schema_real_model.py`. One script replaces both reproduction paths and
> drives the shipped backend rather than a private hook or a bare HuggingFace model:
>
> ```bash
> # grammar-constrained decoding through the public 0.4.6 logits_processor hook, on two
> # real compiled adapters (needs no PAW_API_KEY, makes no network or teacher call --
> # runs offline against already-cached programs)
> uv run python scripts/measure_constrained_decoding.py --label your-machine-name
> ```

**A note on cost, because we went looking and found nothing to report**: the upstream
`programasweights` SDK has no `/account`, `/usage`, or `/billing` endpoint anywhere in
`client.py`, and the readthedocs front page doesn't mention credits, pricing, or a free
tier at all. Over the course of this project's testing we ran 15+ real `compile()` calls
(3 hardware configs, JIT speedup, active learning, grammar decoding, fail-open, the three
terse-spec tests and the triage re-score above) against a real account with nothing paid
into it, and never once hit a credit-exhaustion or payment-required error. That's
consistent with an unmetered beta, not evidence of one — the only place a balance might
actually be visible is the authenticated `programasweights.com/settings` dashboard,
outside what any of these scripts (or an agent without a browser) can check. Worth a
manual look before relying on this being free indefinitely.

### If inference is unexpectedly slow (seconds, not milliseconds)

The installed `llama-cpp-python` is probably the CPU-only PyPI wheel. The fix (prebuilt
CUDA wheel, or source build with `-DGGML_CUDA=on`) is documented for users under "GPU
support" in [`docs/install.md`](../docs/install.md#gpu-support); this section keeps only
the notes specific to reproducing the measurements.

**If you've installed `paw-kit[measure]` (named `[torch]` at the time) in the same environment**: prefer the source
build over the prebuilt wheel. Hit this twice on the same machine: a working
source-built `llama-cpp-python` was later replaced with the prebuilt cu121 wheel (as a
shortcut), and `import llama_cpp` started crashing with `SIGILL` on GPU init — `torch`
had pulled in its own newer CUDA 13.x `nvidia-*` packages alongside the system's CUDA
12.1, and the prebuilt wheel's `dlopen`-found libraries stopped matching what it was
built against. Rebuilding from source (which links against the system CUDA install
directly, not whatever `nvidia-*` pip packages happen to be lying around) fixed it both
times. If both `torch` and `llama-cpp-python` need to coexist in one venv, source-build
the latter.

On a shared cluster: interactive GPU jobs may be restricted to smaller GPUs (check your
cluster's policy) — the actual measurement run needs to go through the batch scheduler
(`sbatch`), not an interactive shell. Also watch your home-directory quota if one exists;
`pip`/`uv` caches and the downloaded base model (~600MB, one-time, in
`~/.cache/programasweights/`) add up faster than you'd expect on a small quota.

> **Note, 2026-09-15 (Track 15).** The `paw-kit[measure]` extra **no longer exists**. It
> carried torch, transformers, peft, safetensors and accelerate for one consumer,
> `scripts/measure_schema_real_model.py`, which is deleted; nothing else in the project
> imported any of them. The torch-versus-`llama-cpp-python` CUDA conflict described just
> above is therefore no longer reachable through a paw-kit extra, and the advice in that
> paragraph now applies only if you install torch yourself into the same environment for
> some other reason. What the constrained-decoding measurement needs instead is
> `paw-kit[paw]`, which pulls `llguidance` alongside the SDK — no torch, no GPU compiler
> toolchain, and prebuilt wheels on Python 3.11-3.13 for Linux (glibc >= 2.31), macOS and
> Windows.

---

## Note, 2026-09-09 (Track 13)

`paw_kit/backend/real.py` and the `RealPAWBackend` class were **deleted**. Passages above
that describe its behaviour — the corrected docstring, the `runtime_executor` pass-through,
the "only place the logits processor could be applied" framing — are left exactly as
written: they are the dated record of what was measured and when, and rewriting them would
destroy the thing this document is for. Read any reference to `real.py` above as historical.

Two consequences for anything in this file that is still load-bearing:

- **No backend shipped in `paw_kit` applies grammar-constrained decoding.** That was already
  true when these measurements were taken; deleting the stub only removes a class that never
  applied it either. `paw.load` validates after generation and falls back on failure.
- **The constrained-decoding results stand unchanged.** They were produced by
  `scripts/measure_schema_real_model.py` and
  `scripts/measure_constrained_decoding_upstream.py`, neither of which routed through
  `RealPAWBackend`. Reproducing the first now needs `uv sync --extra measure` (renamed from
  `--extra torch`).

See `conductor/decisions.md` §3 for why an in-process runtime is out of scope by design.

> **Note, 2026-09-15 (Track 15).** The first bullet above — "**No backend shipped in
> `paw_kit` applies grammar-constrained decoding**" — was true when it was written and is
> no longer true. `ProgramAsWeightsBackend` applies it, by default, through the public
> `logits_processor` hook that shipped in `programasweights==0.4.6` on 2026-09-13. The
> reasoning in that bullet was sound for its date: the premise it rested on was upstream's
> public surface, and upstream changed it. `MockPAWBackend` still ignores
> `grammar_constraint`, and `paw.load` still validates after generation and falls back on
> failure, both of which the bullet also says and both of which remain true. The second
> bullet's reproduction instruction (`uv sync --extra measure`) is dead — see the note in
> "Reproducing" above. `conductor/decisions.md` §2 is amended accordingly; §3 is unchanged
> and still says why an in-process runtime is out of scope.

---

## Corrections logged 2026-09-11

Two corrections were first recorded on `docs/results.md`. That page now carries only
current numbers, so the record moves here, alongside the other dated corrections in this
file.

**Fiscal-weeks frontier baseline: 85.7% → 98.0%.** Arm D (Claude Haiku zero-shot, no
compile) in the fiscal-weeks section was silently capped at 400 output tokens, which
truncated 39 of 300 answers into failures. Re-run uncapped at 2,000 tokens: 300/300
answered, 0 truncated, 98.0% exact. The frontier baseline was understated, so the finetune
compiler's 49.0% is further from it than first published, not closer. The 85.7% figures in
the fiscal-weeks tables above are left as the dated record; read them against this note.

**Programs compiled before the private-by-default fix are still public.** Compiles have
defaulted to `public=False` since 2026-09-10 (`b47ddea`), but every program compiled
before that is still live and public. Verified 2026-09-11: all six `program_id`s this
project has ever compiled report `public: True` from the server and download anonymously,
no key required, from the `hf_url` it returns. Every one predates `b47ddea` by at least a
day. Nothing in this project's data was sensitive (the folded examples are synthetic demo
tickets), but the mechanism is general: a manifest's `public` field records what was
*requested*, not what the server confirmed, upstream caches by spec text and ignores
`public` on a cache hit, and re-running any of this project's historical measurement
scripts unchanged returns the same old public program regardless of what `public=` the
caller passes now. Publishing this repository publishes every `program_id` it commits, and
each one resolves, permanently. A later code fix does not reach backward.

> **Corrected 2026-09-12, Track D (`A-2`).** The sentence above — "a manifest's `public`
> field records what was *requested*, not what the server confirmed" — was true when
> written and is no longer the shipped behaviour. The manifest now keeps `public_requested`
> and a three-state `public_confirmed` (`true` / `false` / `null`, with
> `public_confirmed_reason` saying why on `null`) apart; passing `verify_visibility=True`
> asks the server directly after a compile and records what it says, at the cost of one
> extra authenticated GET. This does not reach backward: the six historical programs above
> were compiled with no verification call at all, so nothing retroactively fills in their
> `public_confirmed`, and they remain public and anonymously downloadable regardless. What
> changes is only what a *future* compile's manifest can tell you about itself, and only
> when `verify_visibility=True` is passed.
