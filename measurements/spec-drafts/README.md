# Spec drafts: testing the docs' terse-spec style

`programasweights.compile()`'s own docstring says the `spec` argument should
"include examples in the text" — but the docs' front-page example itself is a
single terse sentence with no embedded input/output examples at all
(`paw.compile("Fix malformed JSON: repair missing quotes and trailing
commas")`). Every example this toolkit ships (`date_normalizer`,
`triage_ticket`, `pii_scrubber`) follows the docstring's advice, not the
front page's. That leaves an untested gap: what actually happens when a real
user copies the pattern the front page shows them, rather than the pattern
the docstring recommends? The three specs below are written in that terse,
no-examples style on purpose, in three domains not already covered by this
toolkit's examples, chosen specifically because a one-line spec leaves real
ambiguity that a longer, carefully-scoped spec would have closed off.

Note on assertions: only `regex_match`, `max_length`, `min_length`,
`exact_match`, and `not_contains` are implemented in
`paw_kit/test/runner.py::evaluate_assertion` (grep-confirmed). `contains`,
`is_valid_json`, and `one_of` fall through to the `"Unknown assertion rule"`
branch and would always fail, so none of the three drafts below use them.

## spec-1-json-repair.yaml

Spec: `"Fix malformed JSON: repair missing quotes and trailing commas"` (the
docs' own front-page example, verbatim).

The spec names two specific defects — missing quotes, trailing commas — but
never says anything about: quote-style mixing (single vs. double quotes in
the same object), comments (JSON5-style `//` or `/* */`), non-JSON literals
(`NaN`, `undefined`), idempotence on input that's already valid, or
formatting of the output (compact vs. pretty-printed, key ordering). Because
the spec's two named defects are so narrow, I predict the fuzzer will surface
failures concentrated at the *edges of the named categories* rather than
within them: the adapter should handle bare "missing quotes + trailing
comma" cases fine (that's literally what it was told to fix), but will
either (a) leave single-quoted keys/values untouched when they're mixed with
already-double-quoted ones in the same object (`{'a': 'b', "c": 'd',}`),
because the spec's wording ("missing quotes") doesn't obviously cover
"wrong quote character" as a case, or (b) mangle or pass through inputs with
JS-only constructs (`// comment`, `NaN`, `undefined`) that were never
mentioned as things to fix, producing output that fails the
`^[\{\[].*[\}\]]$` shape check entirely rather than a clean repair.

## spec-2-phone-extractor.yaml

Spec: `"Pull out the phone number from this text and format it
consistently."`

"Format it consistently" never says *what* the target format is — E.164
(`+15551234567`), parenthesized US style (`(555) 123-4567`), plain dashes
(`555-123-4567`), or something else — and the spec is silent on what to do
when a message contains more than one phone number, contains an extension,
uses a non-US country code, or uses a vanity number spelled with letters
(`1-800-FLOWERS`). Because there is no anchor format anywhere in the spec
text, I predict the fuzzer will surface *format drift*: outputs that pass
the `standard_cases` (single, unambiguous US numbers) will converge on
whatever format dominated whatever pretraining/compilation data the adapter
saw, but the two-number probe
(`"Call 555-123-4567 or 555-987-6543, whichever works."`) will produce an
arbitrary, non-reproducible choice of which number "wins" — and the
extension probe (`"...555-222-3333 ext. 204."`) will either silently drop
the extension or leak the literal string `"ext. 204"` into the output,
failing the `not_contains: "ext"` assertion, precisely because the spec
never told the compiler extensions were even a category of thing to decide
about.

## spec-3-review-sentiment.yaml

Spec: `"Decide if this product review is positive or negative."`

The spec hard-codes a forced binary with no third option, and never
specifies the output token's format (case, whether it should be a bare word
at all vs. a sentence). It also gives the compiler no instruction for
reviews that are genuinely mixed (praise plus a complaint in the same
sentence), sarcastic (positive vocabulary, negative intent), or contain no
evaluative content at all (empty string, gibberish, a foreign-language
review). I predict the fuzzer will surface two distinct, falsifiable
failures: first, the sarcasm probe (`"Oh great, ANOTHER product that breaks
in a week. Love it."`) will be classified `"positive"` because the spec
gives the compiled adapter no signal that literal-polarity words can carry
inverted intent; second, the mixed-praise-and-complaint probe (`"Good
packaging but the product itself is garbage."`) and the truly neutral probe
(`"It's fine I guess, does the job."`) will each get a confident binary
label anyway (rather than erroring or hedging), and which label they get
will be unstable across recompiles — because the terse spec never
acknowledged a middle category exists for the compiler to route those
inputs into consistently.
