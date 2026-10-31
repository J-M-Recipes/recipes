# Fault injection: a worker dies mid-decode (2026-10-30)

Both runs: a streamed request in flight, then `docker kill k3-worker` on Spark 5 (plan target spark4 = proxy conn 4).

| run | proxy failure action | what happened |
|---|---|---|
| 1 (23:10, full model) | `abort()` | proxy tripped after ~2 s (`k3proxy FATAL: collect layer 68: timeout on conn 4`), but the engine core then sat writing a core dump of ~480 GB resident memory through apport (state `I`, 2.6 GB of `/var/crash` file after 2 min) while the API server kept answering `/health` **200** and every new request hung; the in-flight stream ended after 151/600 tokens at 103 s. `fault1-abort.server.log`, `fault1-client.txt`. **Not acceptable: looks healthy, serves nothing.** |
| 2 (23:19, 12-layer model, same proxy code) | `kill(getpid(), SIGKILL)` | proxy tripped at layer 10 (conn 4) ~2 s after the kill; vLLM `AsyncLLM output_handler failed` within a second; the engine container **exited on its own within ~5 s**; `/health` unreachable; no core dump. vLLM 0.31.0 emits a streaming error chunk before `[DONE]` on engine death (`entrypoints/openai/completion/serving.py:497`). `fault2-sigkill.server.log`, `fault2-client.txt`. |

The published `scripts/worker/k3hop.c` is the SIGKILL version. In no run did the pod emit a token computed without
a layer's remote experts: the proxy never returns a partial collect.
