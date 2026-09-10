# Roadmap

In order, and nothing gets announced until the first item is done:

1. ~~Run `ProgramAsWeightsBackend` end to end against the real service on an RTX 3080
   (11GB) and an A100. Publish actual compile time, per-call latency, and `paw-test`
   pass rates on the three example tasks, with the exact commands used.~~ **Done**; see
   [`measurements/`](../measurements/README.md), plus real JIT hot-swap,
   grammar-constrained decoding, fail-open, semantic-correctness, and finetune-compiler
   tests that went beyond the original scope of this item.
2. ~~Decide from those numbers whether folding traced examples into the spec helps at
   all.~~ **Done, and the honest answer is "it depends on the failure mode."** A real
   A/B test (`measurements/README.md#does-folding-examples-into-the-spec-text-actually-help-a-real-answer-on-the-second-try`)
   found folding examples in fixed format-ambiguity failures dramatically (one task
   went from 0% to 92.5% structural pass) and did nothing for failures unrelated to
   format (unicode-handling edge cases), and introduced a new failure mode of its own
   (verbatim memorization of an example for out-of-distribution input). Not a flat
   yes/no; read the section before deciding whether to fold examples in for your task.
3. ~~Wire `--backend real` in the CLI to `ProgramAsWeightsBackend`.~~ **Done**; it
   resolves to the upstream SDK, announces any fallback to the mock, and refuses to
   recompile (a paid, destructive operation) unless asked explicitly. See `paw_kit/cli.py`.
4. ~~Decide `RealPAWBackend`'s fate.~~ **Done, deleted.** It was the placeholder for an
   in-process PEFT path, justified mainly as "the only place the logits processor could
   ever be applied", which [`measurements/`](../measurements/README.md) showed to be
   false: constrained decoding reaches the real upstream adapter through llama.cpp's own
   sampling loop. paw-kit wraps upstream PAW rather than reimplementing it, so the
   placeholder was removed instead of built.
5. Ask upstream for a supported `grammar` / `logits_processor` passthrough on
   `PawFunction.__call__`. `llama_cpp.Llama.sample()` already accepts both; the SDK's
   decode loop already calls it. Until there is an answer, **nothing in this package
   applies constrained decoding**: `paw.load` validates after generation and falls back
   on failure. Whether `RegexLogitsProcessor` stays here at all depends on that answer.
6. ~~Put a gate in front of the hot-swap.~~ **Done**: [shadow mode](./shadow-mode.md),
   on by default. Measuring it against a real adapter, including what `audit_rate=0.05`
   costs in teacher calls, is the next measurement.
