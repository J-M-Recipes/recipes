# Public reproducibility package — secondary structured-output V2 receipts

Status: **experimental release-candidate evidence only**. This package does **not** make MTP a quality-approved default and does **not** change the original 2026-09-06 MTP campaign verdict, which remains **INCONCLUSIVE / promotion blocked**.

## Scope

These receipts rerun the original V1 supplied-data secondary fixtures with a frozen V2 request contract. The only request-level change is adding OpenAI-compatible `response_format: {"type":"json_schema", ...}`. The structural schemas constrain output shape only; they do not encode expected IDs, expected totals, expected order, or fixture oracle answers.

Lanes:

| lane | file | rows | schema OK | correct | note |
|---|---|---:|---:|---:|---|
| V1 preserved baseline | `v1-secondary-structured-v2.jsonl` | 40 | 40 | 37 | V1 restored state was explicitly cancelled by the user; all keepers were last stopped. This receipt preserves the prior V1 evidence only. |
| `sc13g` no MTP | `sc13g-nomtp-v2.jsonl` | 40 | 40 | 38 | matched slot-cache control |
| `sc13g` + MTP(1) | `sc13g-mtp-v2.jsonl` | 40 | 40 | 37 | experimental release candidate only; not default quality-approved |

Matched no-MTP vs MTP unique-task comparison from `paired-summary.json`: 20 unique tasks, 40 paired requests, 18 ties, 1 MTP win, 1 MTP loss.

## Verification

Run from this directory:

```bash
python3 receipt-audit.py
```

Expected: exit code `0`. The script reads only package files and verifies:

- 40 rows per lane, 20 fixtures × 2 repeats.
- 40/40 protocol, parse, and schema pass per lane.
- correctness counts: V1 37/40, no-MTP 38/40, MTP 37/40.
- request pairing and fixture SHA identity across V1, no-MTP, and MTP.
- `response_format` schemas contain no fixture/oracle-like values.
- package receipts and docs contain no private home paths or credential-shaped markers.

The saved audit output is `receipt-audit.json`.

## Files

- `VALIDATION-CONTRACT-V2-FROZEN.md` — public-safe copy of the frozen V2 contract; only a private local source path was redacted.
- `secondary_gates_v2.py` — frozen runner copy, SHA-256 `c9f5f378c4e30d05fe2e2afe5e99b428fe4fcf5c0d42539a27c2f3d173d65188`.
- `v1-secondary-structured-v2.jsonl` — preserved V1 raw receipt.
- `sc13g-nomtp-v2.jsonl` — matched no-MTP raw receipt.
- `sc13g-mtp-v2.jsonl` — matched MTP(1) raw receipt.
- `paired-summary.json` — matched comparison summary.
- `public-evidence-manifest.json` — relative source names plus source/published SHA-256 hashes.
- `public-privacy-scan.json` — public-safety scan summary.

## Practical API example

Example request shape for grounded supplied-data answers:

```json
{
  "model": "glm-5.3-big",
  "messages": [
    {
      "role": "user",
      "content": "From ONLY these supplied records, select status=open AND region=east. Return one JSON object with count, total (sum of amount), and ids sorted lexicographically. No extra keys or prose. Data: [...]"
    }
  ],
  "temperature": 0,
  "top_p": 1,
  "seed": 20260906,
  "max_tokens": 4096,
  "stream": false,
  "chat_template_kwargs": {"reasoning_effort": "low"},
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "grounded_answer",
      "schema": {
        "type": "object",
        "properties": {
          "count": {"type": "integer"},
          "total": {"type": "integer"},
          "ids": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["count", "total", "ids"],
        "additionalProperties": false
      }
    }
  }
}
```

The schema is structural. It contains field names and JSON types only, not hardcoded answers.

## Speed note

The historical +31.9% MTP warm short-C1 speed gain came from the original unconstrained-quality MTP campaign. It was not remeasured under this V2 structured-output schema run and must not be presented as a schema-mode speed measurement.
