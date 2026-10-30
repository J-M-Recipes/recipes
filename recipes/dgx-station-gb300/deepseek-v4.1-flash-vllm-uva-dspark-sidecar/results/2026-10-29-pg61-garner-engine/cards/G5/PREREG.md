# PREREG — Card G5: per-class C1 A/B at temperature 1.0, Pg (pre-#61161) vs Pg61 vs h2-REF (written 2026-10-29 before boot)

Question: does the #61161 port lower T > 0 speculative acceptance / C1 speed (G4 T1 filler hinted: C1 accept
0.357/0.313/0.331 → 0.303/0.305/0.306, C4 mixed, GSM8K T=1 no change)? Also: the C1 headline (mean of 3) for Pg61.

Arms, one boot each, same window, order Pg61 → Pg → A:
- Pg61: `launch-pg61.sh up` (as G4).
- Pg: pol tree + ovl-58132g (#60945 guard only), stock fastpath, PEER3_SPINS default — exactly G1f/G2's Pg.
- A: h2-REF (serving lane; carries #61161 + guard since AP2/L3).
Instrument per arm: `perclass_decode_t.py` (sha1 recorded), private perclass prompts (sha1 recorded), 5 classes,
`--temperature 1.0 --top-p 1.0`: one natural-length warm pass (`--natural --fl 384`, seed 42, not scored), then three
forced C1 passes (FL 256 per class) with seeds 42, 43, 44. Reported per class: mean of 3 tok/s and tokens/step.

Reading (no promotion bar; this measures):
- Port effect = Pg61 / Pg per class. "Port costs C1 at T=1" if the mean ratio < 0.97 AND tokens/step drops on ≥ 4/5
  classes; "no effect" if the mean ratio is within ±3 %.
- Headline = Pg61 / A per class (mean of 3, both measured tonight).
