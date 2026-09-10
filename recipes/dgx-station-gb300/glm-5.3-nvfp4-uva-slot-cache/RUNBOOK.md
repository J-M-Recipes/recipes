# Slot-cache live-window runbook

Measurement-only. Nothing in this recipe authorizes promotion.
`promotion_authorized=false` on every receipt.

## T0 gate — required before any live release build or Station candidate launch

Three of four C2 live attempts halted on mismatches between fake systemd fixtures and real systemd. A live release is blocked until this gate is green.

1. Fixture tests against real Station `ExecStart` shapes pass:
   `python3.11 -m pytest tests/test_systemd_readback_real_fixtures.py tests/test_systemd_readback_contract.py -q`
2. On the Station, run the dummy-timer contract through the production `read_systemd_timer` path. It must arm a system-scope transient timer whose service is `/bin/true --restore-only`, read it back, cancel it, and leave zero units. It must never name the incumbent or the C2 restore runner.

```bash
# Copy systemd_readback_contract.py, window_c2_continuation.py, and window_e1_v2.py
# into one directory on the Station, then:
python3 systemd_readback_contract.py \
  --dry-run \
  --sudo \
  --unit-prefix glm53-contract-test \
  --out /tmp/glm53-contract-test-out
```

Expected stdout includes `CONTRACT_OK`. Then prove:

```bash
systemctl list-units --all --plain --no-legend 'glm53-contract-test-*'
# empty
```

Record `systemd --version` from the receipt. Operator preflight must **not** create `/tmp/glm53-c2-continuation-host-operation.lock`.

Do not build or stage a live release until T0 is `CONTRACT_OK` on the Station.

## Inherited live rules

- Exact incumbent `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907` is the rollback baseline.
- Independent system-scope restore timer, read back before incumbent stop.
- Finite per-window deadlines.
- Never patch on the Station. Fix offline, TDD, PR, CI, rebuild, re-review, then retry.
- Credentials never appear in receipts, logs, commits, or summaries.
- Do not touch `:30003` or any non-GLM container.
- Independent hash-scoped live-release review before copy/launch; independent results review before any speed/quality claim.
