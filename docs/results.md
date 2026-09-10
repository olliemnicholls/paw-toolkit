# Results: what has been measured, and what it showed

Every number here comes from one machine and one run, recorded with the exact commands
in [`measurements/README.md`](../measurements/README.md). Several first-pass figures were
wrong and are corrected there in place, with the correction visible. Nothing below was
produced on the mock backend.

**The short version.** PAW compiles a spec into a tiny local model that answers in about
65 ms on a consumer GPU, and paw-kit can swap it in for a frontier API call at 11x lower
latency. Whether you should is a different question: the first model swapped in this way
agreed with Claude on 60% of tickets. paw-kit's value is in that gap. The test tools
found it, and shadow mode now stops it from reaching users. The same tools then located
the difference between upstream's two compilers: the fast one cannot put a mapping stated
in the spec into an adapter, and the finetune one can.

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
| Does the swapped-in triage model agree with a fresh Claude call? | 12 of 20 tickets, 60% |
| Semantic correctness of three one-sentence specs, judged by Claude | 60% to 90% depending on the task |
| Structural pass rate of the same specs | 0% to 100% depending on whether the spec pinned the output format |
| Date normaliser test suite | 71 of 82; the 11 failures are whitespace-only inputs the suite gave no legal answer for |
| Fabrications found by the fuzzer | `1-800-FLOWERS` became invented digits |

The test tools produced every number in this table. The judge itself flips 4.5% of
verdicts on identical input at default temperature; the shipped judge pins temperature to
zero and `paw-test judge --diff` shows whether that held.

## What helps and what does not

| Question | Result |
|---|---|
| Folding traced examples into the spec | Fixed format failures outright on one task, 0% to 92.5% structural. Did nothing for unicode-handling failures. Introduced verbatim memorisation of an example on out-of-distribution input |
| The finetune compiler versus the fast one, easy task (phone extraction) | 132 of 134 outputs byte-identical. The two differences: it stopped copying an example verbatim, and it fabricated a phone number where the fast compiler declined |
| The finetune compiler versus the fast one, hard task (ticket triage, 60 tickets, teacher ceiling 91.7%) | Fast with no examples 38.3%; fast with 8 examples 53.3%; finetune with the same 8 examples 60.0%, at 223 s to compile against 5 s. It closed about a sixth of the gap: it recovered the three critical tickets that folding had cost, and it is the only arm willing to say "low", but it lost 6.7 points on department. On a task the finetune compiler fails, the fast compiler also fails; neither rescues it |
| The finetune compiler versus the fast one, on a rule the base model does not know (fiscal weeks, 300 dates, exact ground truth) | Fast 11.3% with no examples and 10.3% with 8; finetune 49.0% with the same 8; Claude Haiku zero-shot 85.7%. The fast compiler emitted a fixed vocabulary of week labels regardless of the date, and folding examples made it worse. The finetune compiler located the fiscal year on 298 of 300 and was within one week on 85%. First measurement that separates the two compilers: a real capability difference, and still not a usable adapter |
| The finetune compiler versus the fast one, on an arbitrary lookup table (30 countries to 6 made-up codes, 300 sentences, exact ground truth) | Fast 33.0% with no examples and 29.0% with 8; finetune 97.7% with the same 8; Claude Haiku zero-shot 100%. The fast compiler never emitted two of the six codes, and with examples it memorised the eight folded countries (87.5%) and fell below chance on the other 22 (7.7%). The finetune adapter matched Haiku byte for byte on 293 of 300 at 32 ms per call against 750 ms. Together with fiscal weeks: the fast compiler cannot put an arbitrary spec-stated mapping into an adapter at any example count, and the finetune compiler can |
| The active-learning repair loop on the 11 date failures | 0 repaired, correctly: every teacher label failed the suite's own rules, so the loop refused to train on them. The gap was in the suite, which now has `abstain_value` |
| Grammar-constrained decoding on a real model | 15 of 15 valid outputs versus 0 of 15 unconstrained, no latency cost once warm. The official runtime cannot apply it yet |

## Safety

| Question | Result |
|---|---|
| Adapter file deleted after a real compile | Calls fell back to the teacher; no crash |
| Shadow mode on the 60% adapter, shipped defaults | 12 of 20 on every window, parked after five windows, never promoted. Zero compiles, zero paid calls |
| Cost of the optional post-promotion audit at 5% | 20 teacher calls over 621 served calls; zero at the default rate |
| Latency added to the caller by shadow mode | None measurable |
| Compiles private by default | Upstream publishes the spec and folded examples otherwise; verified live |

Two corrections came out of the shadow-mode run. The chance of a 60% adapter passing
one window by luck is 5.1%, about 23% over five windows, not the 2.5% first written.
And a 20-sample audit window catches drift to 60% on only about 40% of windows, so
the audit window needs sizing for the drift you care about.

## Limitations

One machine, one run each. The finetune-compiler comparison covers two tasks; the hard
one used 48 teacher-generated tickets alongside 20 real ones, and its adapter scores sit
within a few points of the teacher's own run-to-run variation. The shadow-mode run used a
replay of recorded Claude answers, so it exercised the gate arithmetic, not fresh
traffic. Judge-scored numbers carry the judge's own ±2-point noise.
