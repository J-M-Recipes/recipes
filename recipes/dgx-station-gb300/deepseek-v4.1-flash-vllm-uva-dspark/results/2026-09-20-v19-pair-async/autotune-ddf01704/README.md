# FlashInfer autotune set — config hash ddf01704 (v19 release)

`autotune_configs.json` is the exact set the v19 reference loads ("Loaded 189 configs"). Config hash `ddf01704bfd5…` = nightly
`dee37d89`, FlashInfer 0.6.18.post1, SM103, v15 hook on, `--cpu-offload-gb 54`, `--kv-cache-dtype fp8_ds_mla`, seqs 24, the
v18 cudagraph sizes. To reproduce v19 numbers, place it at
`<vllm-cache>/flashinfer_autotune_cache/0.6.18.post1/103a/ddf01704bfd54f378b73e381bec6601ad21c7ce1b07d359ef605429001378c0f/autotune_configs.json`
before the first boot. Without it the first boot live-tunes (~75 min) and produces a *different* set — faster at C1 by ~5%
(193 vs 183 tok/s, T3b6 live vs loaded) but not the pinned release, and not reproducible boot-to-boot. See FlashInfer RFC #3920.
Identical to `../../2026-09-19-overnight-ksched-nightly-adaptive/T3b6/receipts/autotune_configs-fp8kv-ddf01704-from-T3b6.json`.
