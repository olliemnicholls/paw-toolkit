# Golden CLI snapshots

One file per case, holding the exact output the paw-kit CLI produces today for a fixed
fixture. `tests/test_golden_cli.py` re-runs every case and compares byte for byte.

* `<case>.txt` — the invocation, its exit code, and its stdout (and stderr, when there
  is any).
* `<case>.json` — the machine-readable artifact, for the commands that write one
  (`check --json`, `compare --json`, `judge --out`, `report --json`).

## These snapshots record current behaviour *including its known defects*

This is deliberate and it is the whole point. The suite exists so that the
`bug-hunt-remediation` campaign's eight sub-tracks cannot change CLI output by accident:
several of them touch `paw_kit/test/runner.py`, `judge.py`, `compare.py` and `cli.py`,
whose output a reviewer cannot check by eye. A summary line that loses a column, or a
`--json` artifact that loses a field, is invisible in a code diff and loud here.

Some of the text pinned in these files is text a later track is *expected* to change —
several of the hunt's findings are about misleading CLI output. **A snapshot is not an
endorsement of the behaviour it records.** When a track intentionally changes an output,
it regenerates that snapshot in its own diff, where the reviewer sees the before and the
after next to the code change that caused it. That is the mechanism; regenerating a
snapshot on its own, to turn a red test green, defeats it entirely.

Cases whose captured behaviour looks wrong carry a `LOOKS WRONG, NOT FIXED HERE:` note
on their `Case(...)` entry in `tests/test_golden_cli.py`. Fixing them belongs to the
track that owns the finding, not here.

## Regenerating

```
PAW_GOLDEN_UPDATE=1 tools/campaign.sh test <track> tests/test_golden_cli.py
```

Every comparison becomes a write, so the run is **not** a check — a tripwire test
(`test_snapshots_are_not_being_updated_in_ci`) fails on purpose to make that impossible
to miss. Always `git diff tests/golden/` afterwards and read what moved.

Adding a case: add a `Case(...)` to `_cases()` and regenerate. Deleting one: delete the
`Case` *and* its files — `test_no_orphan_snapshots` fails on a snapshot with no live
case, so the directory cannot silently become a museum.

## What is normalised, and what deliberately is not

Every scrubber lives in `tests/test_golden_cli.py` with a docstring saying what it hides.
The list is short on purpose: the value of the suite is entirely in what it still
notices.

| Scrubber | Hides | Still caught |
| --- | --- | --- |
| `strip_ansi` | ANSI colour/style codes | all text; only a pure colour change is invisible |
| `rstrip_lines` | trailing space padding Rich adds to help panels | box-drawing borders are not trimmed, so real width changes still show |
| `scrub_workdir` | the absolute fixture directory, replaced by a **same-length** token | every path component below it (`report.json`, `docker/`, …) |
| `scrub_short_timestamps` | `YYYY-MM-DD HH:MM:SS` in the `report` table, replaced by a same-length placeholder | the timestamp's presence, position and column |
| `scrub_json` | the values of `latency_ms`/`latency_a_ms`/`latency_b_ms`/`compile_wall_s` and of six timestamp keys, **by key name only** | the keys themselves; a renamed or dropped field fails |

There is deliberately **no** scrubber for numbers in general, for pass rates, counts,
case ordering, exit codes, or whitespace inside a line.

Two tripwires stop the scrubbers from silently under-covering:
`assert_no_leaked_paths` fails if a fragment of the fixture directory survives (which
happens when Rich wraps a path across two lines — widen that case's `COLUMNS` rather
than broadening a scrubber), and `assert_no_unscrubbed_timestamps` fails if a JSON
artifact still contains an ISO datetime (add the key to `_TIMESTAMP_KEYS`; this is how
`shadow_started_at` was found).

Determinism that is arranged rather than scrubbed:

* **Console width** is pinned at 120 columns, in both places it is read from — see
  `_COLUMNS` in the test module. Rich resolves `COLUMNS` at `Console.__init__`, and
  `paw_kit.cli.console` is built at import time, so the environment override alone is
  not enough; the `pinned_console` fixture replaces it.
* **Working directory length** is padded to a fixed 64 characters (`_WORKDIR_PATH_LEN`),
  because Rich sizes table columns from cell *length* — an unpinned length moves table
  borders even after the path text is scrubbed.
* **No backend, network or key.** Everything runs against `MockPAWBackend` or
  hand-written fixture files. `paw-test judge`'s two judging cases replace
  `paw_kit.cli.anthropic_judge` with a deterministic stub, so the whole of
  `judge_outputs`, `parse_verdict` and `cli.judge_cmd`'s rendering is exercised for real
  while nothing leaves the machine.

The suite passes unchanged under a hostile ambient environment
(`COLUMNS=40 LINES=10 FORCE_COLOR=1 ANTHROPIC_API_KEY=… PAW_API_KEY=… TZ=Asia/Tokyo`).

## Known couplings

* `check_unknown_rule.txt` contains a `https://errors.pydantic.dev/2.13/…` link from a
  pydantic validation error. A pydantic minor upgrade will change it. That is a real
  change to user-visible output, so it is left in rather than scrubbed.
* Help snapshots pin option names, defaults, ordering and help strings. Editing a help
  string is supposed to show up here.

## What is not snapshotted, and why

* `paw-kit demo` (both scenarios) — prints per-call latencies and depends on a
  background compile winning a race against the next call, and sleeps 300ms. `--help`
  only.
* `paw-kit serve` — binds a socket. `--help` only.
* `paw-kit doctor` — its whole output is a description of the local machine (SDK
  install, GPU visibility, `PAW_API_KEY`, upstream service health). `--help` only.
* `--backend real` on any command — needs the upstream SDK; out of scope by
  construction.
* The judge *prompt* (`paw_kit.test.judge.JUDGE_PROMPT`). The stub judge reads the
  prompt but the prompt text is not a CLI output, so it is not pinned here; a change
  to its `<model_output>` delimiters does fail loudly (see `_StubJudge`), a reworded
  instruction does not.
* `export dataset`'s written `traces.jsonl` — the exported record count is pinned via
  stdout, but the file itself is not snapshotted (it is JSONL, not JSON, and `export` is
  not a module this campaign touches).
