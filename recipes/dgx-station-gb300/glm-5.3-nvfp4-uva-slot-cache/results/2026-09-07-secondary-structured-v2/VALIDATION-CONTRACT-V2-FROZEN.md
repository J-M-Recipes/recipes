# Secondary structured-output validation contract V2 frozen

Frozen before any scored V2 live validation output.

## Scope

This contract applies only to a new V2 validation of the existing V1 supplied-data secondary fixtures. It does not mutate, reinterpret, or promote the frozen V1 campaign corpus, frozen files, or result denominators.

## Frozen runner

- Runner: `diagnosis/secondary_gates_v2.py`
- Runner SHA-256: `c9f5f378c4e30d05fe2e2afe5e99b428fe4fcf5c0d42539a27c2f3d173d65188`
- Local tests: `diagnosis/test_secondary_gates_v2.py`
- Local tests SHA-256: `b170ae86f0e004719c066194290a3bc17828dc1b3ff312ee1f5691ec9c5eceb8`
- Original V1 runner source read: private local source path redacted for public package; fixture logic is preserved in `secondary_gates_v2.py`.

## Request contract

For every supplied-data `grounded` or `structured` secondary fixture:

- Use the exact original V1 fixture generator: 10 grounded fixtures + 10 structured array fixtures, repeated twice = 40 scored requests.
- Use the original V1 model/settings:
  - `model: glm-5.3-big`
  - `temperature: 0`
  - `top_p: 1`
  - `seed: 20260906`
  - `max_tokens: 4096`
  - `stream: false`
  - `chat_template_kwargs.reasoning_effort: low`
- Sole request change: add OpenAI-compatible `response_format` with `type: json_schema`.
- The schema must not encode fixture oracle answers, expected IDs, expected totals, or expected order.

### Grounded response_format

```json
{
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
```

### Structured response_format

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "enabled_names",
    "schema": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

## Scoring contract

Each row records protocol, parse/schema, and correctness independently:

1. `transport_fail`: request exception, timeout, non-JSON HTTP body, or missing top-level response shape.
2. `protocol_fail`: missing `choices[0].message`, `finish_reason != "stop"`, `message.content` not a string, tool calls present, or visible `<think>`, `</think>`, `<tool_call>`, or `</tool_call>` tags in content.
3. `format_fail`: `json.loads(message.content)` fails. It must consume the full string; prefix salvage is forbidden.
4. `schema_fail`: parsed value violates the fixture category schema.
5. `oracle_fail`: protocol/format/schema-valid answer differs from the unchanged fixture `want`.
6. `pass`: protocol/format/schema-valid answer exactly equals the unchanged fixture `want`.

Required row fields include: sanitized exact `request`, full raw API `response`, `protocol_ok`, `parse_ok`, `schema_ok`, `correctness_ok`, `parsed_answer`, errors, `failure_class`, `fixture_sha256`, and `passed`.

## Forbidden rescues

- No trimming after the first JSON value.
- No regex extraction.
- No retry inside this denominator.
- No oracle values in schema.
- No changes to original V1 fixtures, original V1 runner, frozen corpus, freeze artifacts, or historical results.
- Historical V1 tool-chain episodes may remain historical evidence but are not counted as new V2 passes.
