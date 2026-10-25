# Verified test inputs

These are intentionally short, atomic claims. They are useful smoke tests because
they have a clear subject, a checkable fact, and enough context for web retrieval.
Run them from the app with the indicated language and keep a result only when the
returned status, reason, and sources agree.

## Accepted inputs

| Input | Language | Expected good result |
| --- | --- | --- |
| `The Taj Mahal is in Agra, India.` | English | Supported or high-confidence evidence-backed result; sources should identify Agra as the location. |
| `The Pacific Ocean is the largest ocean on Earth.` | English | Supported result with multiple independent reference sources. |
| `India's capital is New Delhi.` | English | Supported result with reference evidence identifying New Delhi. |
| `The Sun is a star.` | English | Supported result with stable reference evidence. |

## What counts as a good result

- The claim is split as one atomic claim.
- At least two retrieved sources or one strong primary/reference source support the
  same wording.
- The result status matches the evidence (`verified`, `contradicted`, or
  `unknown`); sources alone must not force `verified`.
- The reason explains why the sources support, conflict with, or fail to establish
  the claim.
- Confidence is consistent with the result: an `unknown` claim must not display
  high confidence merely because search results exist.

## OCR relevance-filter test

For an uploaded social-media screenshot, the input should retain only the factual
headline or claim. It should remove profile names, timestamps, reaction counts,
`LIVE`, `BREAKING NEWS`, comments, shares, watermarks, and isolated OCR fragments.
The browser first performs local cleanup, then sends the cleaned OCR text to
`POST /api/clean-text`. The API uses the configured Sarvam model when available
and safely falls back to the local cleaner if it is unavailable.

Example expected cleaned text from the supplied screenshot:

`गृहमंत्री अमित शाह कोरोना की चपेट में`

The cleaned text is what is placed in the input box and sent to `/api/analyze`.
