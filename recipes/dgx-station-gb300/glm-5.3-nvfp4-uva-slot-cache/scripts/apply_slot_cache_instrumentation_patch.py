#!/usr/bin/env python3
"""Apply exact-hash guarded slot-cache instrumentation to gpu_model_runner.py.

This writes a patched copy only. It refuses unexpected source hashes, output
path escapes, and overwrites. It does not touch the live runtime.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

EXPECTED_SOURCE_SHA256 = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"patch anchor count {count}, expected 1 for: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_source(text: str) -> str:
    text = replace_once(
        text,
        "from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast\n",
        "from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast\n"
        "import atexit as _slot_cache_atexit\n"
        "import os as _slot_cache_os\n"
        "try:\n"
        "    from slot_cache_window_instrumentation import (\n"
        "        SlotCacheWindowController,\n"
        "        finalize_slot_cache_snapshot,\n"
        "        finalize_slot_cache_windows,\n"
        "    )\n"
        "    _SLOT_CACHE_INSTRUMENTATION_IMPORT_ERROR = None\n"
        "except Exception as _slot_cache_import_error:\n"
        "    SlotCacheWindowController = None\n"
        "    _SLOT_CACHE_INSTRUMENTATION_IMPORT_ERROR = _slot_cache_import_error\n"
        "    def finalize_slot_cache_snapshot(*_args: Any, **_kwargs: Any) -> None:\n"
        "        return None\n"
        "    def finalize_slot_cache_windows(*_args: Any, **_kwargs: Any) -> None:\n"
        "        return None\n",
    )
    text = replace_once(
        text,
        "        num_nans: torch.Tensor | None = None,\n    ):\n",
        "        num_nans: torch.Tensor | None = None,\n"
        "        slot_cache_snapshot: Any | None = None,\n"
        "        slot_cache_window_controller: SlotCacheWindowController | None = None,\n"
        "    ):\n",
    )
    text = replace_once(
        text,
        "        self._has_fault: torch.Tensor | None = None\n\n        # Initiate the copy on a separate stream, but do not synchronize it.\n",
        "        self._has_fault: torch.Tensor | None = None\n"
        "        self._slot_cache_snapshot = slot_cache_snapshot\n"
        "        self._slot_cache_window_controller = slot_cache_window_controller\n"
        "        self._slot_cache_snapshot_cpu = None\n\n"
        "        # Initiate the copy on a separate stream, but do not synchronize it.\n",
    )
    text = replace_once(
        text,
        "            if check_ep_fault:\n                has_fault = get_ep_all2all_manager().query_fault()\n                self._has_fault = has_fault.to(\"cpu\", non_blocking=True)\n            self.async_copy_ready_event.record()\n",
        "            if self._slot_cache_snapshot is not None:\n"
        "                self._slot_cache_snapshot_cpu = (\n"
        "                    self._slot_cache_snapshot.counters_device.to(\n"
        "                        \"cpu\", non_blocking=True\n"
        "                    )\n"
        "                )\n"
        "            if check_ep_fault:\n                has_fault = get_ep_all2all_manager().query_fault()\n                self._has_fault = has_fault.to(\"cpu\", non_blocking=True)\n            self.async_copy_ready_event.record()\n",
    )
    text = replace_once(
        text,
        "        max_gen_len = self.sampled_token_ids_cpu.shape[-1]\n"
        "        self.async_copy_ready_event.synchronize()\n\n"
        "        # Release the device tensors once the copy has completed.\n",
        "        max_gen_len = self.sampled_token_ids_cpu.shape[-1]\n"
        "        self.async_copy_ready_event.synchronize()\n"
        "        if self._slot_cache_snapshot is not None:\n"
        "            finalize_slot_cache_snapshot(\n"
        "                self._slot_cache_window_controller,\n"
        "                self._slot_cache_snapshot,\n"
        "                self._slot_cache_snapshot_cpu,\n"
        "            )\n\n"
        "        # Release the device tensors once the copy has completed.\n",
    )
    text = replace_once(
        text,
        "        self.device = device\n        self.dtype = self.model_config.dtype\n\n        self.check_ep_fault = False\n",
        "        self.device = device\n        self.dtype = self.model_config.dtype\n"
        "        if SlotCacheWindowController is None:\n"
        "            if _slot_cache_os.environ.get(\"SLOT_CACHE_QUIESCENT_SNAPSHOTS\", \"0\") == \"1\":\n"
        "                raise RuntimeError(\n"
        "                    \"slot-cache quiescent snapshots are enabled but instrumentation import failed\"\n"
        "                ) from _SLOT_CACHE_INSTRUMENTATION_IMPORT_ERROR\n"
        "            self.slot_cache_window_controller = None\n"
        "        else:\n"
        "            slot_cache_window_controller = SlotCacheWindowController.from_env()\n"
        "            if slot_cache_window_controller.enabled:\n"
        "                slot_cache_unsupported_topology = []\n"
        "                for slot_cache_attr in (\n"
        "                    \"pipeline_parallel_size\",\n"
        "                    \"tensor_parallel_size\",\n"
        "                    \"data_parallel_size\",\n"
        "                    \"decode_context_parallel_size\",\n"
        "                ):\n"
        "                    slot_cache_value = getattr(parallel_config, slot_cache_attr, 1)\n"
        "                    if slot_cache_value != 1:\n"
        "                        slot_cache_unsupported_topology.append(\n"
        "                            f\"{slot_cache_attr}={slot_cache_value}\"\n"
        "                        )\n"
        "                if getattr(parallel_config, \"use_ubatching\", False):\n"
        "                    slot_cache_unsupported_topology.append(\"use_ubatching=True\")\n"
        "                if slot_cache_unsupported_topology:\n"
        "                    raise RuntimeError(\n"
        "                        \"slot-cache quiescent snapshots only supports single-GPU \"\n"
        "                        \"topology; unsupported: \"\n"
        "                        + \", \".join(slot_cache_unsupported_topology)\n"
        "                    )\n"
        "                _slot_cache_atexit.register(\n"
        "                    finalize_slot_cache_windows,\n"
        "                    slot_cache_window_controller,\n"
        "                    reason=\"process_shutdown\",\n"
        "                )\n"
        "                self.slot_cache_window_controller = slot_cache_window_controller\n"
        "            else:\n"
        "                self.slot_cache_window_controller = None\n\n"
        "        self.check_ep_fault = False\n",
    )
    text = replace_once(
        text,
        "            model_output = self._model_forward(\n                input_ids=input_ids,\n                positions=positions,\n                intermediate_tensors=intermediate_tensors,\n                inputs_embeds=inputs_embeds,\n                **model_kwargs,\n            )\n\n        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n",
        "            try:\n"
        "                slot_cache_target_range = (\n"
        "                    self.slot_cache_window_controller.nvtx_range_for_boundary(\n"
        "                        scheduler_output=scheduler_output,\n"
        "                        boundary=\"target_forward\",\n"
        "                        cudagraph_stats=cudagraph_stats,\n"
        "                    )\n"
        "                    if self.slot_cache_window_controller is not None\n"
        "                    else nullcontext()\n"
        "                )\n"
        "                with slot_cache_target_range:\n"
        "                    model_output = self._model_forward(\n                    input_ids=input_ids,\n                    positions=positions,\n                    intermediate_tensors=intermediate_tensors,\n                    inputs_embeds=inputs_embeds,\n                    **model_kwargs,\n                )\n"
        "            except Exception:\n"
        "                if self.slot_cache_window_controller is not None:\n"
        "                    self.slot_cache_window_controller.mark_failed_boundary(\n"
        "                        scheduler_output=scheduler_output,\n"
        "                        boundary=\"target_forward_exception\",\n"
        "                        reason=\"target_forward_exception\",\n"
        "                        cudagraph_stats=cudagraph_stats,\n"
        "                    )\n"
        "                raise\n"
        "            if self.slot_cache_window_controller is not None:\n"
        "                self.slot_cache_window_controller.mark_target_forward_boundary(\n"
        "                    scheduler_output=scheduler_output,\n"
        "                    cudagraph_stats=cudagraph_stats,\n"
        "                )\n\n"
        "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n",
    )
    text = replace_once(
        text,
        "            with record_function_or_nullcontext(\"gpu_model_runner: draft\"):\n                self._draft_token_ids = self.propose_draft_token_ids(\n",
        "            with record_function_or_nullcontext(\"gpu_model_runner: draft\"):\n"
        "                slot_cache_draft_range = (\n"
        "                    self.slot_cache_window_controller.nvtx_range_for_boundary(\n"
        "                        scheduler_output=scheduler_output,\n"
        "                        boundary=\"draft_forward\",\n"
        "                        cudagraph_stats=cudagraph_stats,\n"
        "                        draft_metadata={\"has_drafter_config\": True},\n"
        "                    )\n"
        "                    if self.slot_cache_window_controller is not None\n"
        "                    else nullcontext()\n"
        "                )\n"
        "                with slot_cache_draft_range:\n"
        "                    self._draft_token_ids = self.propose_draft_token_ids(\n",
    )
    text = replace_once(
        text,
        "                    slot_mappings,\n                )\n                self._copy_draft_token_ids_to_cpu(scheduler_output)\n",
        "                        slot_mappings,\n                    )\n                self._copy_draft_token_ids_to_cpu(scheduler_output)\n",
    )
    text = replace_once(
        text,
        "        with record_function_or_nullcontext(\"gpu_model_runner: eplb\"):\n            self.eplb_step()\n\n        # self.kv_connector_output may be modified during drafting\n",
        "        with record_function_or_nullcontext(\"gpu_model_runner: eplb\"):\n            self.eplb_step()\n\n"
        "        slot_snapshot = None\n"
        "        if self.slot_cache_window_controller is not None:\n"
        "            if getattr(self, \"use_v2_model_runner\", False):\n"
        "                raise RuntimeError(\"slot-cache quiescent snapshots do not support V2 model runner\")\n"
        "            slot_cache_metadata = self.slot_cache_window_controller.make_metadata(\n"
        "                scheduler_output=scheduler_output,\n"
        "                boundary=\"step_complete\",\n"
        "                cudagraph_stats=cudagraph_stats,\n"
        "                draft_metadata={\n"
        "                    \"has_drafter_config\": spec_config is not None,\n"
        "                    \"drafter_runs_model_forward\": (\n"
        "                        drafter_runs_model_forward\n"
        "                        if spec_config is not None\n"
        "                        else False\n"
        "                    ),\n"
        "                    \"drafter_after_bookkeeping\": draft_after_bookkeeping,\n"
        "                    \"abort_seen_after_execute\": False,\n"
        "                },\n"
        "            )\n"
        "            slot_snapshot = self.slot_cache_window_controller.maybe_snapshot_device(\n"
        "                metadata=slot_cache_metadata\n"
        "            )\n"
        "            if slot_snapshot is not None and not self.use_async_scheduling:\n"
        "                self.slot_cache_window_controller.finalize_slot_cache_snapshot(\n"
        "                    slot_snapshot\n"
        "                )\n\n"
        "        # self.kv_connector_output may be modified during drafting\n",
    )
    text = replace_once(
        text,
        "        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n            sampler_output = self._sample(logits, spec_decode_metadata)\n",
        "        with record_function_or_nullcontext(\"gpu_model_runner: sample\"):\n"
        "            try:\n"
        "                sampler_output = self._sample(logits, spec_decode_metadata)\n"
        "            except Exception:\n"
        "                if self.slot_cache_window_controller is not None:\n"
        "                    self.slot_cache_window_controller.mark_failed_boundary(\n"
        "                        scheduler_output=scheduler_output,\n"
        "                        boundary=\"sample_exception\",\n"
        "                        reason=\"sample_exception\",\n"
        "                        cudagraph_stats=cudagraph_stats,\n"
        "                    )\n"
        "                raise\n",
    )
    text = replace_once(
        text,
        "                num_nans=num_nans_device,\n            )\n",
        "                num_nans=num_nans_device,\n"
        "                slot_cache_snapshot=slot_snapshot,\n"
        "                slot_cache_window_controller=self.slot_cache_window_controller,\n"
        "            )\n",
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    actual = sha256(source)
    if actual != args.expected_sha256:
        print(f"sha256 mismatch for {source}: got {actual}, expected {args.expected_sha256}", file=sys.stderr)
        return 2
    if output.exists():
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 3
    output.parent.mkdir(parents=True, exist_ok=True)
    patched = patch_source(source.read_text())
    output.write_text(patched)
    print(f"source_sha256={actual}")
    print(f"output={output}")
    print(f"output_sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
