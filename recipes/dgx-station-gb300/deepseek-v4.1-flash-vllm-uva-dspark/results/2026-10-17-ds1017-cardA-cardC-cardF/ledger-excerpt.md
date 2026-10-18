## 2026-10-17 08:25 CDT — Card A: DeepSeek-V4.1-Flash-1017 (1bae0e1088993b9b82de08f59de5be5a34e42170) on v21 config, CTL1→CAND→CTL2
worker: Milo (hermes milo) · James: 'you have the station' 2026-10-17 ~07:15
Card A — DeepSeek-V4.1-Flash-1017 (CAND) vs df42c109 (CTL1/CTL2), identical v21 launch, same window.
knee r2 agg tok/s
arm    C1     C2     C4     C8     C12    C16   | gauntlet | replay n=4 r1/r2
CTL1   182.7  286.3  433.1  685.6  841.2  980.9  |   12/12 | [316.7, 472.9]
CAND   192.6  288.9  423.9  672.6  850.9  946.1  |   12/12 | [257.1, 444.4]
CTL2   182.6  286.3  434.2  686.0  852.9  984.9  |   12/12 | [313.1, 467.1]
CAND vs mean(CTL): C1 +5.4%  C2 +0.9%  C4 -2.2%  C8 -1.9%  C12 +0.5%  C16 -3.7%

agent fixture: accept% (tok/s, acc/step)
CTL1  prose 34.9% (203, 1.75)  structured 52.5% (246, 2.63)  code 66.1% (244, 3.30)  shell_ops 90.8% (277, 4.54)  tool_json 87.8% (220, 4.39)
CAND  prose 25.2% (174, 1.26)  structured 48.9% (232, 2.44)  code 69.6% (251, 3.48)  shell_ops 80.0% (261, 4.00)  tool_json 77.3% (200, 3.87)
CTL2  prose 34.9% (204, 1.75)  structured 52.5% (248, 2.63)  code 66.1% (246, 3.30)  shell_ops 90.8% (279, 4.54)  tool_json 87.8% (221, 4.39)

Parity: CAND vs CTL = different weights (expected to differ). CTL1 vs CTL2 = instrument baseline.
parity CTL1 vs CTL2: non-tool identical 16/16 · all 18/18
parity CAND vs CTL1: non-tool identical 0/16 · all 0/18
receipts /home/milo/dsv41/1017/results · dsv41-vllm-1017A-{CTL1,CAND-KEPT,CTL2} stopped-and-kept · v21 stopped-and-kept · 1017 serving as dsv41-vllm-1017A-LIVE on :30006 (NOT promoted)

## 2026-10-17 08:40 CDT — Card D: 1017 quality gate on dsv41-vllm-1017A-LIVE (v21 config) vs v21/df42c109 baseline (2026-10-05-lib-t1-v21)
worker: Milo (hermes milo) · lil-bench v0.7.7 from the Mac
GSM8K-200: 96.5% (193/200) vs baseline 97.5% (195/200) — Δ −1.0 pp, McNemar p=0.5 NOT significant; flips 2 baseline-only / 0 cand-only; completion tokens −13.6% (214 vs 248 avg)
needle-checksum 500/500 EXACT both; completion tokens 161 vs 211 avg (−24%)
receipts (Mac) ~/hermes/jm-recipes/.../results/2026-10-17-ds1017-cardD/ · 1017 still LIVE on :30006, NOT promoted

## 2026-10-17 09:20 CDT — Card C: k-schedule recheck on 1017 (CTL=[[1,4,5],[5,24,1]] on LIVE; CAND1=[[1,2,5],[3,24,1]]; CAND2=[[1,4,3],[5,24,1]])
worker: Milo (hermes milo) · James: '1) go' 2026-10-17 08:45
2026-10-17 08:43:40 CDT ARM CTL1: autotune HASH HIT
2026-10-17 08:45:45 CDT ARM CTL1 knee: C1=192.0 C2=288.2 C4=431.6 C8=671.4 C16=945.8 
2026-10-17 08:45:45 CDT ARM CTL1 fixture: prose 181.5(25.2%) structured 232.5(48.9%) code 252.2(69.6%) shell_ops 262.8(80.0%) tool_json 213.1(77.3%) 
2026-10-17 08:45:45 CDT ARM CTL1 replay: agg 325.9 acc 57.0% toolturns 17;agg 410.8 acc 59.5% toolturns 15;
2026-10-17 08:54:03 CDT boot->ready CAND1 486s
2026-10-17 09:02:23 CDT ARM CAND2: autotune HASH HIT
2026-10-17 09:04:52 CDT ARM CAND2 knee: C1=110.8 C2=204.6 C4=396.6 C8=567.5 C16=929.6 
2026-10-17 09:04:52 CDT ARM CAND2 fixture: prose 107.9(40.0%) structured 134.3(61.0%) code 166.3(77.7%) shell_ops 167.1(91.7%) tool_json 126.6(87.8%) 
2026-10-17 09:04:52 CDT ARM CAND2 replay: agg 284.0 acc 70.8% toolturns 14;agg 418.7 acc 73.8% toolturns 13;
2026-10-17 09:10:19 CDT ARM CAND1: autotune HASH HIT
2026-10-17 09:12:33 CDT ARM CAND1 knee: C1=192.2 C2=288.1 C4=475.1 C8=697.1 C16=950.7 
2026-10-17 09:12:33 CDT ARM CAND1 fixture: prose 173.7(25.2%) structured 232.6(48.9%) code 252.3(69.6%) shell_ops 262.8(80.0%) tool_json 200.9(77.3%) 
2026-10-17 09:12:33 CDT ARM CAND1 replay: agg 251.4 acc 86.0% toolturns 16;agg 363.5 acc 86.4% toolturns 15;
2026-10-17 09:18:00 CDT ARM CTL2: autotune HASH HIT
2026-10-17 09:20:11 CDT ARM CTL2 knee: C1=191.5 C2=286.7 C4=419.8 C8=668.8 C16=936.8 
2026-10-17 09:20:11 CDT ARM CTL2 fixture: prose 180.6(25.2%) structured 230.7(48.9%) code 247.9(69.6%) shell_ops 258.7(80.0%) tool_json 220.9(77.3%) 
2026-10-17 09:20:11 CDT ARM CTL2 replay: agg 272.7 acc 55.3% toolturns 14;agg 397.1 acc 58.1% toolturns 15;
receipts /home/milo/dsv41/1017/results-cardC · dsv41-vllm-1017C-* stopped-and-kept · LIVE restarted (schedule unchanged, no promotion)

## 2026-10-17 09:42 CDT — Card F: v0.31.0 + deep_gemm_mega_moe + MEGA_ATTN on 1017, hook off, UVA off60 — F1 DIED
worker: Milo (hermes milo) · James: go
2026-10-17 09:40:36 CDT launched dsv41-vllm-1017F-v0310-megamoe-hookoff-EXP image=vllm/vllm-openai:v0.31.0
2026-10-17 09:40:36 CDT waiting 0 min: 
2026-10-17 09:42:06 CDT F1 RESULT: DIED — NotImplementedError: DeepSeek V4 MegaMoE currently requires expert parallel. Enable it with --enable-expert-parallel, or pick a different moe backend.|    raise RuntimeError(|RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}|
receipts /home/milo/dsv41/1017/results-cardF · dsv41-vllm-1017F-v0310-megamoe-hookoff-EXP-BOOT-FAIL kept · LIVE restarted

## 2026-10-17 09:44 CDT — Card F: v0.31.0 + deep_gemm_mega_moe + MEGA_ATTN on 1017, hook off, UVA off60 — F1 DIED
worker: Milo (hermes milo) · James: go
2026-10-17 09:40:36 CDT launched dsv41-vllm-1017F-v0310-megamoe-hookoff-EXP image=vllm/vllm-openai:v0.31.0
2026-10-17 09:40:36 CDT waiting 0 min: 
2026-10-17 09:42:06 CDT F1 RESULT: DIED — NotImplementedError: DeepSeek V4 MegaMoE currently requires expert parallel. Enable it with --enable-expert-parallel, or pick a different moe backend.|    raise RuntimeError(|RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}|
2026-10-17 09:43:22 CDT launched dsv41-vllm-1017F-v0310-megamoe-ep1-hookoff-EXP image=vllm/vllm-openai:v0.31.0
2026-10-17 09:43:22 CDT waiting 0 min: 
2026-10-17 09:44:52 CDT F1 RESULT: DIED — torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 270.00 MiB. GPU 0 has a total capacity of 249.81 GiB of which 158.00 MiB is free. Including non-PyTorch memory, this process has 249.62 GiB memory in use. Of the allocated memory 248.76 GiB is allocated by PyTorch, and 36.24 MiB is reserv|    raise RuntimeError(|RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}|
receipts /home/milo/dsv41/1017/results-cardF · dsv41-vllm-1017F-v0310-megamoe-ep1-hookoff-EXP-BOOT-FAIL kept · LIVE restarted

## 2026-10-17 09:50 CDT — Card C VERDICT: keep [[1,4,5],[5,24,1]] on 1017
CTL1 192.0/288.2/431.6/671.4/945.8 · CTL2 191.5/286.7/419.8/668.8/936.8 (C1/C2/C4/C8/C16) · replay CTL 326,411 / 273,397
CAND1 [[1,2,5],[3,24,1]]: C4 475.1 (+12% vs bracket), C8 697 (+4%), C1/C2/C16 flat — but replay 251/364, below both controls on both reps (v13 mechanism: agent text accepts 3.5–4/step at k=5; k=1 caps step at ~1.86). FAIL for the agent lane.
CAND2 [[1,4,3],[5,24,1]]: C1 110.8 (−42%), C8 567 (−15%) — k=3 inside num_speculative_tokens=5 pays the 5-token verify shape for 3 tokens. FAIL; do not re-spend.
Runner v1 false-flagged CAND1 (docker logs | grep -q under pipefail → SIGPIPE); v2 reads logs to file first. CAND1 result is from its restart (valid same-window control on 0909).

## 2026-10-17 09:50 CDT — Card F CLOSED (hard no): vLLM v0.31.0 deep_gemm_mega_moe is incompatible with UVA expert offload
Attempt 1 (no EP): NotImplementedError "DeepSeek V4 MegaMoE currently requires expert parallel". Attempt 2 (+--enable-expert-parallel, EP=1): OOM at construction, 248.76 GiB allocated — experts not offloaded.
Root cause (vllm v0.31.0 vllm/models/deepseek_v4/nvidia/model.py ~L547–575): DeepseekV4MegaMoEExperts.transform at first forward calls deep_gemm.transform_weights_for_mega_moe(w13/w2 …) → full interleaved GPU copy of all experts, then drops loader params. Offloading the originals cannot help; 269 GiB transformed experts > 250 GiB HBM. Reopen trigger: upstream transform-into-pinned-host (+ DeepGEMM host-read unverified).
Containers dsv41-vllm-1017F-*-BOOT-FAIL kept · LIVE restarted · fresh cache dir /home/milo/dsv41/vllm-cache-v0310 (empty, no autotune happened)

## 2026-10-17 09:55 CDT — Card E: PROMOTED DeepSeek-V4.1-Flash-1017 (1bae0e10) on the v21 config — James: GO
dsv41-vllm-1017A-LIVE → dsv41-vllm-1017-v21cfg-BOUND-REF (:30006, served name dsv41-flash-uva unchanged) · dsv41-vllm-v21-fused-split-BOUND-REF → -RETIRED-REF stopped-and-kept (rollback = docker stop <1017 REF> && docker start <v21 RETIRED>, ~8 min)
Config unchanged from v21: 0909 image, hook v19 dir, rowmap-static-v1, off60/util0.97/1M/seqs24/lpt6144, ksched [[1,4,5],[5,24,1]], cgsizes 1..128. Model mount /models/DeepSeek-V4.1-Flash-1017-1bae0e10…
Gates: Card A speed (C1 +5.4%, C8 −1.9%, C16 −3.7% vs CTL bracket; gauntlet 12/12) · Card C schedule (kept) · Card D quality (GSM8K 96.5 vs 97.5 n.s.; needle 500/500)
WAIVER: promotion rule "DSpark acceptance within 2 pts" NOT met (shell 91→80, tool_json 88→77, prose 35→25). Waived by James: acceptance shift is a property of the retrained 1017 drafter, not an engine/lane defect (same image, same kernels, controls bit-identical 18/18). Rule stands for engine changes on fixed weights.
