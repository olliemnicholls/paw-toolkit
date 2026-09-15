# Results: what has been measured, and what it showed

Every number here comes from one machine and one run, recorded with the exact commands
in [`measurements/README.md`](../measurements/README.md). That file is the lab notebook
and keeps the raw output and the provenance of every figure. This page carries the
numbers. Nothing below was produced on the mock backend.

**The short version.** PAW compiles a spec into a tiny local model that answers in about
65 ms on a consumer GPU, and paw-kit can swap it in for a frontier API call at 11x lower
latency. Whether you should is a different question: the first model swapped in this way
agreed with Claude on 46.7% of held-out tickets. Shadow mode is the gate that keeps such
an adapter from reaching users by default. The same test tools located the difference
between upstream's two compilers: the fast one cannot put a mapping stated in the spec
into an adapter, and the finetune one can.

## Speed

| Question | Result |
|---|---|
| Steady-state latency of a compiled function | 65 ms on an RTX 3080; 89 ms on a shared A100; 5.9 s on the CPU-only PyPI wheel |
| Compile time | 1 to 5 s with the default compiler; 181 s with the finetune compiler |
| First call | 2 s to ~110 s, depending on whether the 600 MB base model is cached |
| Real hot-swap, Claude Haiku teacher to local adapter | Teacher ~987 ms per call; local 84 to 96 ms; about 11x |

The A100 being no faster than the 3080 is real: at 0.6B parameters and batch size 1 the
work is latency-bound, so a consumer GPU is the realistic target. GPU versus CPU is the
90x that matters.

## Correctness

| Question | Result |
|---|---|
| Does the swapped-in triage model agree with a fresh Claude call? | 7 of 15 held-out tickets, 46.7% |
| Semantic correctness of three one-sentence specs, judged by Claude | 70% to 90% depending on the task |
| Structural pass rate of the same specs | 0% to 100% depending on whether the spec pinned the output format |
| Date normaliser test suite (82 cases) | 71 of 82; the 11 failures are whitespace-only inputs the suite gave no legal answer for |
| Fabrications found by the fuzzer | `1-800-FLOWERS` became invented digits |
| Judge noise | Re-judging identical input/output pairs at the API's default temperature flipped 4.5% of verdicts (6 of 134). The shipped judge pins temperature to zero; `paw-test judge --diff` shows whether that held |

## What helps and what does not

| Question | Result |
|---|---|
| Folding traced examples into the spec | Fixed format failures outright on one task (0% to 92.5% structural). Did nothing for unicode-handling failures. Introduced verbatim memorisation of an example on out-of-distribution input |
| Finetune compiler vs fast, easy task (phone extraction) | 132 of 134 outputs byte-identical. The two differences: the finetune adapter stopped copying an example verbatim, and it fabricated a phone number where the fast one declined |
| Finetune compiler vs fast, hard task (ticket triage, 60 tickets, teacher ceiling 91.7%) | Fast with no examples 38.3%; fast with 8 examples 53.3%; finetune with the same 8 examples 60.0%, at 223 s to compile against 5 s. It recovered the critical tickets that folding had cost and is the only arm willing to say "low", but lost 6.7 points on department. A task the finetune compiler fails, the fast compiler also fails |
| Finetune compiler vs fast, on a rule the base model does not know (fiscal weeks, 300 dates) | Fast 11.3% with no examples and 10.3% with 8; finetune 49.0% with the same 8; Claude Haiku zero-shot 98.0%. The fast compiler emitted a fixed vocabulary of week labels regardless of the date. The finetune compiler located the fiscal year on 298 of 300 and was within one week on 85%: a real capability difference, and still not a usable adapter |
| Finetune compiler vs fast, on an arbitrary lookup table (30 countries to 6 made-up codes, 300 sentences) | Fast 33.0% with no examples and 29.0% with 8; finetune 97.7% with the same 8; Claude Haiku zero-shot 100%. With examples the fast compiler memorised the eight folded countries (87.5%) and fell below chance on the other 22 (7.7%). The finetune adapter matched Haiku byte for byte on 293 of 300 at 32 ms per call against 750 ms |
| The active-learning repair loop on the 11 date failures | 0 repaired, correctly: every teacher label failed the suite's own rules, so the loop refused to train on them. The gap was in the suite, which is what `abstain_value` is for |
| Grammar-constrained decoding on a real adapter | On an adapter compiled for its own schema it changes nothing: all 75 constrained/unconstrained pairs came back byte-identical, and every output was valid either way. Forced onto a schema the adapter was *not* compiled for, 5 of 5 parse against 0 of 5 unconstrained — well-formed, not right. Costs 0.8 ms per generated token |

Together, the fiscal-week and lookup-table rows say: the fast compiler cannot put an
arbitrary spec-stated mapping into an adapter at any example count, and the finetune
compiler can.

## Safety

| Question | Result |
|---|---|
| Adapter file deleted after a real compile | Calls fell back to the teacher; no crash |
| A served adapter call that hangs indefinitely | Falls back to the teacher after `adapter_timeout_s` (10 s by default) instead of blocking the caller forever |
| Shadow mode on a recorded 20-ticket replay, shipped defaults | 12 of 20 on every window (cyclic) or 11 to 13 of 20 (random draw); parked after five windows, never promoted. Zero compiles, zero paid calls |
| Chance of a weak adapter being promoted by luck | An adapter that agrees on a random 60% of inputs passes one 20-sample window with probability 5.1%, about 23% across the five windows before it is parked. Raise `shadow_window` if that matters; at 50 the per-window chance is below 0.3% |
| Post-promotion audit at 5% | 68 teacher calls over 1,200 served calls (5.7%); zero at the default rate. A 20-sample audit window catches a drift to 60% agreement on only about 40% of windows, so size `audit_window` for the drift you care about |
| Latency added to the caller by shadow mode | None measurable |
| Compiles private by default | Yes. Note that upstream caches compiles by spec text and ignores `public` on a cache hit, so re-running a spec that was once compiled public returns the same public program; see [the real-backend notes](./real-backend.md#compiles-are-private-by-default) |

## Limitations

One machine, one run each. The finetune-compiler comparison covers a handful of tasks;
the hard one used 48 teacher-generated tickets alongside 20 real ones, and its adapter
scores sit within a few points of the teacher's own run-to-run variation. The shadow-mode
run used a replay of recorded Claude answers, so it exercised the gate arithmetic, not
fresh traffic. Judge-scored numbers carry the judge's own ±2-point noise.
