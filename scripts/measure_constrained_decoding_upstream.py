"""Can `RegexLogitsProcessor` reach the real upstream-compiled adapter?

Context. `paw.schema`'s constrained decoding is one of paw-kit's three headline
features and the "0.0% Pydantic syntax failures" invariant in `conductor/decisions.md`
rests on it. Until now it had no path to a real model inside the library:
`ProgramAsWeightsBackend.infer()` warns that it "cannot apply grammar_constraint at
decoding time: the upstream SDK exposes no grammar/logits hook", and `RealPAWBackend`
-- the placeholder where masking was supposed to live -- raised `NotImplementedError`.
(That class was deleted in Track 13; an in-process runtime is out of scope by design.)
The only real-model evidence (`measure_schema_real_model.py`) drives a *separate*
HuggingFace model, not a compiled PAW adapter.

That framing turns out to be true only of the SDK's *public* surface. Its runtime is a
hand-rolled decode loop over `llama-cpp-python` (`programasweights/runtime_llamacpp.py`),
holding a real `llama_cpp.Llama` at `PawFunction._llm`, and calling `_llm.sample(temp=...)`
per token. `Llama.sample()` accepts `logits_processor=` and `grammar=`; the SDK simply
never passes either. So the hook is not missing -- it is behind a private attribute.

This script measures what happens when we inject one, and it is deliberately an
experiment rather than a shipped code path: it reaches into `_llm` (private, unsupported,
may break on any SDK release) and monkeypatches `sample`. Do not copy this into
`paw_kit/` without reading the "Fragility" note in the write-up.

Two conditions, same adapter, same machine, same inputs:
  unconstrained  the SDK exactly as shipped
  constrained    the same call with paw-kit's own `pydantic_to_regex` ->
                 `RegexLogitsProcessor` masking every token that would leave the
                 schema's language

Needs no PAW_API_KEY: it runs `offline=True` against an already-cached program.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel

from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.schema.grammar import pydantic_to_regex
from paw_kit.schema.logits_processor import RegexLogitsProcessor

DEFAULT_ADAPTER = "measurements/phone_extractor-paw-4b-qwen3-0.6b.paw"

INPUTS = [
    "Office line: +1-555-666-7777",
    "Call me on 555-222-3333 ext. 204",
    "Reach support at (555) 987 6543 anytime",
    "Text 555-111-2222 for mobile support",
    "no phone number here at all",
]


class Contact(BaseModel):
    """Deliberately NOT the shape the adapter was compiled for.

    The phone-extractor adapter emits a bare string like `(555) 666-7777`. Asking it for
    this JSON object is the point: if masking works, a program compiled to emit a bare
    string is forced into a schema it was never trained on. Nothing but token-level
    masking can do that.
    """

    area_code: int
    number: str
    kind: Literal["mobile", "landline", "unknown"]


class _Injector:
    """llama-cpp `LogitsProcessor` protocol: (input_ids, scores) -> scores.

    Tracks FSM state across decoding steps. `RegexLogitsProcessor` is deliberately
    framework-agnostic (`filter_logits`/`get_next_state` take plain ints and sequences,
    with a `Dict[int, str]` vocabulary), so nothing here is HuggingFace-specific -- the
    vocabulary is built from llama.cpp's own detokenizer below.
    """

    def __init__(self, processor: RegexLogitsProcessor, eos_token_id: int) -> None:
        self.p = processor
        self.eos = eos_token_id
        self.stats: Dict[str, Any] = {"calls": 0, "allowed_s": 0.0, "mask_s": 0.0, "states": set()}
        self.reset()

    def reset(self) -> None:
        self.state = self.p.initial_state
        self._seen: Optional[int] = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self.stats["calls"] += 1
        if self._seen is None:
            # First call of this generation: everything present is prompt, not output.
            self._seen = len(input_ids)
        while self._seen < len(input_ids):
            nxt = self.p.get_next_state(self.state, int(input_ids[self._seen]))
            if nxt is None:
                break
            self.state = nxt
            self._seen += 1

        self.stats["states"].add(self.state)
        t0 = time.time()
        allowed = self.p.get_allowed_tokens(self.state)
        self.stats["allowed_s"] += time.time() - t0

        t0 = time.time()
        mask = np.full(scores.shape, -np.inf, dtype=scores.dtype)
        idx = np.fromiter((i for i in allowed if i < scores.shape[-1]), dtype=np.int64)
        if idx.size:
            mask[idx] = scores[idx]
        if self.p.is_final_state(self.state) and self.eos < scores.shape[-1]:
            # Let the model stop once the schema is satisfied.
            mask[self.eos] = scores[self.eos]
        self.stats["mask_s"] += time.time() - t0
        return mask


def _build_vocabulary(llm: Any) -> Dict[int, str]:
    """token_id -> decoded string, straight from llama.cpp's own detokenizer."""
    vocab: Dict[int, str] = {}
    for tid in range(llm.n_vocab()):
        try:
            s = llm.detokenize([tid]).decode("utf-8", errors="replace")
        except Exception:
            continue
        if s:
            vocab[tid] = s
    return vocab


def _parses(raw: str) -> bool:
    try:
        Contact.model_validate_json(raw)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    pattern = pydantic_to_regex(Contact)
    backend = ProgramAsWeightsBackend(
        offline=True, n_gpu_layers=args.n_gpu_layers, max_tokens=args.max_tokens
    )
    fn = backend._get_function(args.adapter)

    llm = getattr(fn, "_llm", None)
    if llm is None or not hasattr(llm, "sample"):
        print(
            "FAIL: this SDK build does not expose a llama_cpp.Llama at PawFunction._llm, "
            "so there is no sampling loop to inject into. The write-up's fragility note "
            "has come true -- re-read it before trusting any of this."
        )
        return 2

    unconstrained: List[Dict[str, Any]] = []
    for text in INPUTS:
        t0 = time.time()
        out = fn(text)
        unconstrained.append(
            {"input": text, "output": out, "parses": _parses(out), "ms": (time.time() - t0) * 1000}
        )

    vocab = _build_vocabulary(llm)
    eos = llm.token_eos()
    processor = RegexLogitsProcessor(pattern, vocab, eos_token_id=eos)
    injector = _Injector(processor, eos)

    import llama_cpp

    lp_list = llama_cpp.LogitsProcessorList([injector])
    original_sample = llm.sample
    llm.sample = lambda *a, **kw: original_sample(*a, **{**kw, "logits_processor": lp_list})

    constrained: List[Dict[str, Any]] = []
    try:
        for text in INPUTS:
            injector.reset()
            t0 = time.time()
            out = fn(text)
            constrained.append(
                {
                    "input": text,
                    "output": out,
                    "parses": _parses(out),
                    "ms": (time.time() - t0) * 1000,
                }
            )
    finally:
        llm.sample = original_sample

    un_ok = sum(r["parses"] for r in unconstrained)
    c_ok = sum(r["parses"] for r in constrained)
    warm = [r["ms"] for r in constrained[1:]]
    stats = injector.stats

    print(f"\n{'input':<40} {'unconstrained':<26} {'constrained'}")
    for u, c in zip(unconstrained, constrained):
        print(f"  {u['input'][:36]!r:<40}")
        print(f"      unconstrained -> {u['output'][:56]!r}  parses={u['parses']}")
        print(f"      constrained   -> {c['output'][:56]!r}  parses={c['parses']}  ({c['ms']:.0f}ms)")

    print(f"\nvalid `Contact` parses: unconstrained {un_ok}/{len(INPUTS)} -> constrained {c_ok}/{len(INPUTS)}")
    print(f"distinct FSM states visited: {len(stats['states'])}")
    print(f"time in get_allowed_tokens:  {stats['allowed_s']:.2f}s  (one full vocab walk per NEW state)")
    print(f"time in numpy masking:       {stats['mask_s']:.2f}s over {stats['calls']} tokens "
          f"= {stats['mask_s'] / max(stats['calls'], 1) * 1000:.2f}ms/token")
    if warm:
        print(f"constrained mean after first call: {sum(warm) / len(warm):.0f}ms")

    results = {
        "label": args.label,
        "adapter": args.adapter,
        "schema_regex": pattern,
        "vocabulary_size": len(vocab),
        "max_tokens": args.max_tokens,
        "unconstrained": unconstrained,
        "constrained": constrained,
        "unconstrained_valid": un_ok,
        "constrained_valid": c_ok,
        "cases": len(INPUTS),
        "fsm_states_visited": len(stats["states"]),
        "get_allowed_tokens_s": round(stats["allowed_s"], 3),
        "mask_s": round(stats["mask_s"], 3),
        "logits_processor_calls": stats["calls"],
        "ms_per_token_masking": round(stats["mask_s"] / max(stats["calls"], 1) * 1000, 3),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"constrained-upstream-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
