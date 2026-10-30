# Card G5 — per-class C1 A/B at temperature 1.0: Pg (pre-#61161) vs Pg61 vs h2-REF (2026-10-29, 9:03–9:54pm CDT)

**Verdict: the #61161 port has no C1 speed cost (PREREG: "no effect").** Pg61/Pg = 1.012 mean (0.989–1.047 per
class; tokens/step lower on only 3/5 classes). **Headline: Pg61 = 1.31× h2-REF at C1, temperature 1.0, mean of 3
seeds** (1.26–1.36× per class). The G4 T1-filler C1 drop (and its acceptance story) did not reproduce on the
per-class instrument — treat it as filler/single-run noise. 0 errors, 0 foreign tokens, 0 p3 trips. Lane back
9:43pm (smoke OK, watchdog armed), isolate off 9:54pm.

| class | Pg61 tok/s (3 seeds) | Pg tok/s | h2-REF tok/s | Pg61/Pg | **Pg61/h2-REF** | tokens/step Pg61 / Pg / h2-REF |
|---|---|---|---|---|---|---|
| prose | 367.2 (356–377) | 367.2 (366–368) | 289.7 (283–296) | 1.000 | **1.27×** | 2.02 / 2.03 / 1.96 |
| structured | 396.9 (393–404) | 401.2 (395–404) | 314.0 (311–316) | 0.989 | **1.26×** | 2.27 / 2.33 / 2.21 |
| code | 494.9 (480–504) | 482.8 (467–506) | 368.8 (357–386) | 1.025 | **1.34×** | 3.31 / 3.20 / 2.82 |
| shell_ops | 530.4 (525–536) | 532.1 (524–537) | 388.9 (386–393) | 0.997 | **1.36×** | 3.65 / 3.68 / 3.03 |
| tool_json | 463.8 (456–471) | 442.8 (441–445) | 356.5 (349–369) | 1.047 | **1.30×** | 2.97 / 2.79 / 2.68 |
| **mean** | | | | **1.012** | **1.31×** | |

Instrument: `perclass_decode_t.py`, private perclass prompts (40/class, FL 256 forced), `--temperature 1.0 --top-p 1.0`,
seed_base 42/43/44 (seed = base + row), one natural warm pass per arm (not scored). Arms one boot each: Pg61 via
`launch-pg61.sh up`; Pg = pol tree + ovl-58132g, stock fastpath (G1f/G2 config); h2-REF restored lane.
`acc/step` here is the instrument's accepted-tokens-per-draft-step column (bonus token not included).
