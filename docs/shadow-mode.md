# Shadow mode: how the adapter earns production traffic

A compiled adapter that finishes compiling is not yet trusted. The project's own
measurement showed why: a real ticket-triage adapter replaced a live Claude teacher at
~11x lower latency and agreed with a fresh teacher call on under half of held-out tickets
(see [results](./results.md)). Shadow mode is the gate in front of that swap. It is on by
default.

## States

A task moves through `tracing` → `compiling` → `shadow` → `ready`, with `failed` as a
terminal state for a compile that raised.

- **`tracing`**: every call goes to your function and the input/output pair is recorded in
  the SQLite trace database.
- **`compiling`**: the call-count threshold was reached and a compile is running in a
  background thread. Your function keeps serving.
- **`shadow`**: the compile finished. Your function still serves every call. Off the
  request path, the adapter runs on the same input and its answer is compared with yours.
  Each comparison is one sample in a *tumbling* window of `shadow_window` samples. When a
  window completes, the task is promoted if the agreement rate over that window is at
  least `shadow_threshold`; otherwise the window is discarded and a new one starts. A
  window is evaluated exactly once, so a weak adapter does not get a fresh draw on every
  call.
- **`ready`**: the adapter serves. Any exception or schema violation falls back to your
  function (fail-open), as before. If `audit_rate > 0`, a sampled fraction of calls also
  runs your function in the background and the comparison feeds an audit window; if the
  agreement rate over a completed audit window drops below `demote_threshold`, the task is
  demoted to `shadow` and your function serves again.

Nothing in shadow mode raises into the caller or adds latency to it. The comparison work
runs on one daemon thread per task with a bounded queue (`shadow_queue_size`); when the
queue is full the sample is dropped and counted, not waited for. At process exit the
worker drains for at most two seconds.

## Parameters and defaults

| Parameter | Default | What it does |
|---|---|---|
| `shadow_window` | `20` | Comparisons per window. Matches the sample size of the repo's own semantic measurement. `0` disables shadow mode: the adapter is promoted as soon as the compile finishes, exactly the pre-shadow behaviour, and a task already sitting in `shadow` is promoted on the next call. |
| `shadow_threshold` | `0.8` | Minimum agreement over a completed window to promote. Deliberately well above what the repo's measured adapter scored. |
| `audit_window` | `20` | Comparisons per audit window after promotion. **Size this for the drift you want to catch**: a 20-sample window from an adapter that has drifted to 60% agreement reads anywhere from 0.40 to 0.80 in nine draws out of ten, so at the shipped `demote_threshold` it demotes on only about 40% of windows (measured; see [results](./results.md)). A window of 100 makes the same drift demote reliably, at five times the teacher spend per verdict. |
| `audit_rate` | `0.0` | Fraction of served calls that also run your function for comparison. **Off by default** because it spends real teacher calls after promotion and re-invokes your function on a background thread, which requires it to be thread-safe. `0.05` is the recommended value if you turn it on: one call in twenty, so about 400 served calls per completed audit window. Hard-capped at `0.5`. With `0.0` there is no post-promotion drift signal and demotion is unreachable; that is the accepted trade. |
| `demote_threshold` | `0.6` | Audit agreement below this demotes. Must be strictly below `shadow_threshold` so a task cannot flap on window noise. |
| `agreement_fn` | `None` | `(teacher_answer, adapter_answer) -> bool`. The default is conservative: strings are compared after Unicode normalisation and whitespace stripping; Pydantic models and dicts field by field; a string against a structured value by serialising the structure; anything else by equality. `field_tolerance_agreement` is shipped for the "urgency within 1" style of comparison. An `agreement_fn` that raises counts as a disagreement. |
| `shadow_queue_size` | `8` | Bounded work in flight per task. |
| `shadow_max_pairs` | `500` | Retention cap on stored comparisons, oldest pruned first. Must be at least `max(shadow_window, audit_window)`. |
| `redact_trace` | `False` | Also governs the three text columns of each stored comparison. Comparison happens on the raw text; redaction is applied when the row is written. |

The four window parameters are persisted with the task. Changing any of them between
runs starts a fresh window rather than re-slicing old comparisons under new arithmetic.

## When a task is stalled

A task that completes five windows (`5 × shadow_window` comparisons) without promoting is
treated as not converging. From then on it keeps comparing, at a reduced random sample of
one in `shadow_window`, so `paw-kit report` still shows a live agreement rate, but a
completed window past that point is **not evaluated for promotion**. The task stays in
`shadow` and your function keeps serving. A one-time warning says so. The two ways out are
changing any persisted shadow parameter (which starts a fresh epoch) or passing
`shadow_window=0`. `get_agreement()["stalled"]` and the `(stalled)` marker in
`paw-kit report` show this state.

This is what makes "a weak adapter never serves" true rather than merely likely, with one
residual: a window of 20 draws from an adapter that agrees on a random 60% of inputs
reaches 16 of 20 with probability 5.1%, so over the five windows before the stall point
the chance of one lucky promotion is about 23%. That is inherent to any sampling gate.
Raise `shadow_window` if that residual matters for your task; at 50 the per-window chance
falls below 0.3%. In the measured run the adapter's disagreements were fixed per input
rather than random, and it scored exactly 12 of 20 on every window.

## What is stored, and where

Everything lands in the existing `traces.db` (mode `0600`, in a `0700` directory); shadow
mode creates no new file. Two tables are added: `shadow_pairs` (input, teacher output,
adapter output, agreement, per task and epoch, capped at `shadow_max_pairs`) and
`state_transitions` (capped at 200). The `traces` table is untouched: shadow comparisons
never enter the compile corpus. In `ready`, the served path persists no input text at all;
only a fail-open counter on the task row is incremented, and that happens on the worker
thread. Because your function keeps serving while a task is in `shadow`, the `traces`
table keeps growing for as long as it stays there.

Opening an older trace database migrates it in place to the current schema. A task that
was already `ready` keeps serving after the upgrade. A decorated function that is never
called still appears in `paw-kit report`, as a `tracing` task with no calls, because
decoration persists its shadow configuration.

## Watching it

```python
wrapper.get_agreement()          # state, agreement rate, samples, window, stalled,
                                 # dropped, teacher errors, last disagreements
wrapper.get_fail_open_count()    # fallbacks to your function while in ready
```

```bash
paw-kit report --db .paw/traces.db            # every task: state, agreement, fail-opens
paw-kit report --task <id> -n 10 --json       # one task, ten disagreements, JSON
```

`paw-kit report` opens the database with the library's own code, so it migrates an older
file in place; it writes. Two things it shows loosely: for a stalled task the agreement
rate is estimated from the sparse post-stall sample, not from the full windows that were
scored; and the call count freezes at promotion. Dropped comparisons are only visible
in-process, through `get_agreement()["dropped"]`. When the teacher is faster than the
adapter (a replay teacher, or a cached one), most comparisons are dropped because the
bounded queue fills; the window still completes, it just takes more calls.

## Demos

The mock quickstart, `paw-kit demo` and the three examples all pass `shadow_window=0`: a
five-call demo has nowhere near enough calls to fill a window. Shadow mode itself has been
measured against a real adapter, including what `audit_rate=0.05` costs in teacher calls;
see the safety table on the [results page](./results.md).
