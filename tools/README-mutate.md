# `tools/mutate.py` — mutation testing for `paw_kit`

`paw_kit` has ~94% line coverage. That number says every line ran during the suite. It
does **not** say that any assertion would have failed had the line been wrong. Mutation
testing asks the second question directly.

The tool takes a target module, makes one small but definitely-semantic change to it
(flip `<` to `<=`, swap `and` for `or`, negate a `True`, bump an integer literal), runs
the tests, and records the verdict:

| verdict | meaning |
| --- | --- |
| **KILLED** | a test failed. The suite pins that line's behaviour. |
| **SURVIVED** | the suite passed with the broken code. |

## What a surviving mutant means

A survivor is a place where `paw_kit` could be silently wrong and CI would stay green.
It is not automatically a bug — it is an *unasserted behaviour*. Three things it can be:

1. **A missing assertion.** The usual case, and the useful one. Some code path has a
   boundary, a flag, or a limit that no test constrains. Write the test.
2. **Genuinely arbitrary.** A tuning constant, a log-message threshold, a retry count
   that could be 3 or 4 without anyone caring. Nothing to fix.
3. **Equivalent.** The mutation happens not to change observable behaviour at all
   (a `<=` where the equal case is impossible). Nothing to fix.

The tool cannot tell these apart; a human reads the survivor and decides. The recorded
baseline therefore is not a list of bugs, it is a **ratchet**: whatever the current level
of unasserted behaviour is, a change must not make it worse on the modules it touches.

At the baseline commit the answer is **81 of 260 mutations survive** — roughly one in
three semantic changes to the most safety-relevant modules goes unnoticed by 546 passing
tests. That gap is the reason this tool is checked in.

(The throwaway harness this tool was ported from reported 75. The 81 are a strict
superset: the six extra are mutants that that run scored as killed by a test
failure the mutant did not cause. See *Attributable kills* below.)

## What the control gate protects against

This is the most important part of the tool, and it exists because the throwaway version
of this harness lied.

Every mutant is built by `ast.parse` → change one node → `ast.unparse`, which reformats
the *whole* file as a side effect. So before any mutant runs, `mutate.py` writes an
unparse round-trip of each target file with **no semantic change at all** and requires
that file's tests to pass. If the suite cannot survive a no-op, the run is aborted.

The first version of this harness reported 548 mutations with a **100% kill rate in 20
seconds**. It was passing `--timeout=60` to a pytest that does not have `pytest-timeout`
installed, so every single run exited `rc=4` (usage error) — and the harness was scoring
"anything that isn't rc=0" as a killed mutant. A harness that cannot run the tests at all
reports perfect test quality. The control runs are what caught it.

Consequences, all deliberate, none to be relaxed:

- **Control failure is a hard abort (exit 2), not a warning.** Nothing is reported, since
  nothing reported would mean anything.
- **Full stdout and stderr of a failing control are printed.** The rc=4 diagnosis was
  only visible in stderr.
- **Exit codes are classified, not thresholded.** `rc=0` is SURVIVED, `rc=1` is KILLED,
  and *anything else* (2 interrupted, 3 internal error, 4 usage, 5 nothing collected) is
  an **ERROR**, counted and printed separately, never folded into "killed".
- **Provenance is asserted first.** The shared `.venv` has an editable-install `.pth`
  pointing at the main checkout, so a worktree can silently import the wrong `paw_kit`.
  The tool refuses to run unless `paw_kit` resolves inside the mutant copy.

A suspiciously high kill rate is the known failure mode of this kind of tool. If the
number gets nicer, assume the harness broke until proven otherwise.

Two smaller hazards handled in the same spirit: mutants are applied to throwaway copies
of the repo, never the working tree, and byte-compilation is disabled (`int` mutations of
the form `n → n+1` change neither the file's size nor, usually, its mtime-to-the-second,
so a cached `.pyc` can run the *unmutated* code and manufacture a false survivor).
`PYTHONHASHSEED` is pinned so set/dict iteration order cannot move a verdict between runs.

## Attributable kills — why a kill is verified, not assumed

A phase-2 run executes the **whole** suite with one mutated module, so any failure
anywhere scores as a kill — including failures the mutant cannot possibly have caused.
Six suites run in parallel, and the suite has at least one order- and timing-sensitive
test, so this is not theoretical. Measured, on a tree with no changes at all:

- the first baseline run scored a `paw_kit/test/compare.py` counter-initialisation mutant
  as killed by `tests/test_jit_shadow.py::test_demotion_on_audit_agreement_below_demote_threshold`
  — a test in a subsystem `compare.py` does not touch (this is campaign finding **M-4**,
  the known ~1-in-12 shadow-queue flake);
- a second run of the same module then reported 18 survivors against a baseline of 16,
  i.e. **the gate failed on a branch nobody had touched**.

A baseline built from unverified kills is a false-alarm generator. So every phase-2 kill
is now *attributed* with two cheap single-test runs:

1. re-run only the failing test(s) with the mutant still applied — if they pass, the
   full-suite failure was not caused by the mutant → **SURVIVED**;
2. re-run them with the mutant removed — if they fail there too, the test is flaky or
   broken independently of the mutant → **SURVIVED**, and reported loudly by name.

Only *fails with the mutant, passes without it* counts as a kill. The bias toward
SURVIVED is deliberate: over-reporting survivors makes the ratchet conservative, while
over-reporting kills makes the gate fire on innocent branches. With this in place the
targeted gate returned the identical survivor set on two consecutive runs.

Phase-1 kills are not confirmed this way. Phase 1 proved bit-for-bit identical across
three full runs, its selection is small and purpose-built, and confirming a kill by
re-running one test in isolation would misjudge a genuinely order-dependent test as a
non-kill. If phase 1 ever stops being reproducible, that assumption needs revisiting.

## How a run works

**Phase 1** runs each mutant against its module's fast test selection (the `TARGETS` map
at the top of `mutate.py`). **Phase 2** re-runs every phase-1 survivor against the *whole*
suite, because a test outside the selection may still catch it — at the baseline exactly
one of 82 phase-1 survivors was genuinely killed this way (`runner.py`'s `len(output) >=
min_len`, caught by `tests/test_examples.py::test_date_normalizer_cli_invocation`). Only
full-suite survivors are counted and recorded.
`--no-recheck` skips phase 2: faster, but the counts are inflated and not comparable to
the baseline (the tool refuses to write a baseline from such a run).

## Usage

Always through `campaign.sh` when working in a worktree, so the provenance and
`uv run --no-sync` rules are enforced:

```bash
# everything in the default module set (~12 min: 1 min phase 1, ~10 min phase 2)
tools/campaign.sh wt-run <track> uv run --no-sync python tools/mutate.py

# one module, compared against the baseline  <- the campaign's Gate 3 (~2.5 min)
tools/campaign.sh wt-run <track> uv run --no-sync python tools/mutate.py \
    --modules paw_kit/test/runner.py --baseline tools/mutation-baseline.json

# what would run, without running it
... python tools/mutate.py --list
```

Useful flags: `--modules` (bare filenames are matched by suffix), `--workers N`,
`--kinds cmp bool const int`, `--json PATH` for the full per-mutant record,
`--keep-workdir` to inspect a mutant copy, `--strict-identity` to fail on a survivor the
baseline does not know about even when the count did not rise.

### Exit codes

| code | meaning |
| --- | --- |
| 0 | ran cleanly; with `--baseline`, no targeted module regressed |
| 1 | **regression** — a targeted module has more survivors than the baseline records |
| 2 | control gate or provenance failed — nothing ran, results would be fiction |
| 3 | harness errors (unexpected pytest exit codes) — results are untrustworthy |

## The baseline

`tools/mutation-baseline.json` records, per module, how many mutations were generated,
how many survived, and for each survivor its line, column, operator, the expression
before and after, and the source line. The identity is what lets a later run print
*which* mutants newly survive rather than only how many — comparison matches on the exact
id first and falls back to a line-independent fingerprint, so unrelated edits that shift
line numbers do not produce phantom "new" survivors.

Regenerate it (only from a clean `main`, and only when the change is understood):

```bash
tools/campaign.sh wt-run <track> uv run --no-sync python tools/mutate.py \
    --write-baseline tools/mutation-baseline.json
```

The tool refuses to write a baseline from a `--no-recheck` run or from a run that had
harness errors.

## Module coverage

The default set is the six modules where a wrong answer is silent rather than loud:
`test/matching.py`, `test/runner.py`, `test/compare.py`, `test/active.py`,
`schema/grammar.py`, `schema/logits_processor.py`.

`jit/agreement.py`, `jit/shadow.py`, `jit/decorator.py` and `jit/db.py` are configured but
**not** in the default set or the baseline — they are slow (20–40 s per test run), and
`jit/decorator.py`'s selection includes `tests/test_jit_shadow.py`, which has a known
~1-in-12 flake (campaign finding M-4). A flaky test in the selection makes the control
gate fail intermittently, which is the gate working correctly: a suite that does not
reliably pass without a mutation cannot be used to judge one. Fix M-4 before adding those
modules to the baseline.

`schema/grammar.py` skips `int` mutations: its integer literals are almost entirely regex
quantifier and cap tuning constants, and mutating them produced 48 mutants that say
nothing about test quality.
