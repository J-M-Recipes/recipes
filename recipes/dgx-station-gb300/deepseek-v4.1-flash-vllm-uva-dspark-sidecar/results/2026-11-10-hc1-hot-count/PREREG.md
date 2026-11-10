# Card HC1 pre-registration (written 2026-11-10 05:19:00 CST, before the arm boots)
One variable: hot experts per layer 255 -> 275 on the Pg61 sidecar lane (rowmap only; 109 cold per layer on the RTX).
Arms: control A (live Pg61), arm (H=275), control A2 (restored Pg61). Same cells, same salts, loopback.
This card reports; it does not promote. Bars for calling H=275 a candidate worth a promotion card:
1. C1 mixed (mean of 4 passes) >= 0.98 x the control mean.
2. A gain outside the A-vs-A2 spread on at least one of: C32 aggregate (median of 3), C8 aggregate (median of 3), 64K first token.
3. No cell worse than the worse control by more than the A-vs-A2 spread.
4. Tool gauntlet 12/12; server alive; p3 watchdog trip null.
5. KV at boot >= 4,400,013 tokens (the declared pool of the October sidecar launch).
Expectation written down beforehand: C1 flat (the GB300 waits ~0.5 us per layer call at 1-2 rows); gain, if any, at C8/C32 and long prompts.
