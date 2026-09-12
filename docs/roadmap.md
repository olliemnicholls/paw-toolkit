# Roadmap

What is planned, in order. What has already been done and measured is on the
[results page](./results.md).

1. **Grammar-constrained decoding through the official runtime.** The upstream SDK's
   callable exposes no grammar or logits hook, so nothing in this package constrains
   generation today: `paw.load` validates after generation and falls back on failure.
   `llama_cpp.Llama.sample()` already accepts both `grammar` and `logits_processor`, and
   the SDK's decode loop already calls it, so the request to upstream is a passthrough
   rather than new machinery. Until it lands, `paw_kit.schema`'s logits processor is not
   applied by any shipped backend.
2. **Shadow mode against live traffic.** The shipped shadow-mode measurement replays
   recorded teacher answers, so it exercised the gate arithmetic rather than fresh
   traffic. A run against a live teacher, with `audit_rate` on, is the next measurement.
3. **A PyPI release.** paw-kit installs from source only until the items above and the
   current round of fixes have settled.
