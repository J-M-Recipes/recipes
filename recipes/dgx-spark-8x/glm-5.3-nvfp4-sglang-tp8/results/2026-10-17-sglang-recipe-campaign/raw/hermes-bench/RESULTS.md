# Hermes bench: vLLM vs SGLang, GLM-5.3 NVFP4 TP8 on 8x Spark (2026-10-17)

hermes chat -Q, fresh session per fixture, -t terminal, reasoning low, isolated HERMES_HOME (no memory/honcho/rules). One run per fixture unless ×2. tok/s = output tokens / agent duration (includes ~7K-token system-prompt prefill).

| fixture | SGLang wall s | vLLM wall s | SGLang tok/s | vLLM tok/s | pass S/V |
|---|---:|---:|---:|---:|---|
| oneword-1 | 3.4 | 4.0 | 10.0 | 1.2 | ✅/✅ |
| oneword-2 | 2.6 | 2.7 | 2.1 | 2.3 | ✅/✅ |
| count300-1 | 39.8 | 40.5 | 36.5 | 36.0 | ✅/✅ |
| count300-2 | 32.1 | 31.5 | 33.4 | 34.2 | ✅/✅ |
| code | 8.0 | 7.8 | 27.6 | 26.6 | ✅/✅ |
| artifact | 35.5 | 37.5 | 19.2 | 20.2 | ✅/✅ |
| gauntlet20 | 41.7 | 52.2 | 22.5 | 18.3 | ✅/✅ |

Totals: SGLang 163.1 s, vLLM 176.2 s.
