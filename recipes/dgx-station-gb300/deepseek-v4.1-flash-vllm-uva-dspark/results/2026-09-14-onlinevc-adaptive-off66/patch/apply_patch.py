#!/usr/bin/env python3
"""Apply the online-verify-curve patch to the 0909 image's adaptive_verification.py
and model_runner.py (copies taken from the image as *.orig). Idempotent."""
import sys, pathlib

here = pathlib.Path(__file__).parent


def sub(s, old, new, name):
    if old not in s:
        sys.exit(f"anchor not found in {name}: {old[:60]!r}")
    return s.replace(old, new, 1)


# ---------------- adaptive_verification.py ----------------
av = (here / "adaptive_verification.py.orig").read_text()

av = sub(av, '''logger = init_logger(__name__)
_PROFILE_REPLAYS = 5
''', '''logger = init_logger(__name__)
_PROFILE_REPLAYS = 5

# --- online verify-cost curve (jmeadlock experiment, vllm#38256) ---------------
# The boot-time verify curve is profiled on dummy batches whose tokens all route
# to the same few experts, so on an expert-offloaded MoE it never sees that extra
# verify tokens pull extra experts over the host link. This keeps a per-padded-size
# EMA of *real* verify-step forward times and rebuilds the verify table from it.
import os as _os

_ONLINE_ENABLED = _os.getenv("VLLM_DSPARK_ONLINE_VERIFY_CURVE", "0") == "1"
_ONLINE_WINDOW = int(_os.getenv("VLLM_DSPARK_ONLINE_VERIFY_WINDOW", "32"))
_ONLINE_MIN_SAMPLES = int(_os.getenv("VLLM_DSPARK_ONLINE_VERIFY_MIN_SAMPLES", "3"))
_ONLINE_REBUILD_EVERY = int(_os.getenv("VLLM_DSPARK_ONLINE_VERIFY_REBUILD_EVERY", "32"))
_ONLINE_LOG_EVERY = int(_os.getenv("VLLM_DSPARK_ONLINE_VERIFY_LOG_EVERY", "400"))
_ONLINE_EVENT_POOL = 16


class OnlineVerifyCurve:
    """Median of recent observed forward_ms keyed by executed (padded) token count.

    Events are recorded on the current stream around the real forward and
    resolved later with a non-blocking ``query()``, so the observation never
    forces a host sync.
    """

    def __init__(self):
        self.window: dict[int, list[float]] = {}
        self.count: dict[int, int] = {}
        self.total = 0
        self._pool = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(_ONLINE_EVENT_POOL)
        ]
        self._free = list(range(_ONLINE_EVENT_POOL))
        self._pending: list[tuple[int, int]] = []  # (pool_idx, padded_tokens)
        self._open: tuple[int, int] | None = None

    def drain(self) -> int:
        """Resolve finished observations. Returns how many landed."""
        landed = 0
        while self._pending:
            idx, n = self._pending[0]
            start, end = self._pool[idx]
            if not end.query():
                break
            ms = start.elapsed_time(end)
            self._pending.pop(0)
            self._free.append(idx)
            if ms > 0:
                w = self.window.setdefault(n, [])
                w.append(ms)
                if len(w) > _ONLINE_WINDOW:
                    del w[0]
                self.count[n] = self.count.get(n, 0) + 1
                self.total += 1
                landed += 1
        return landed

    def start(self, padded_tokens: int) -> None:
        if self._open is not None or not self._free:
            return
        idx = self._free.pop()
        self._pool[idx][0].record()
        self._open = (idx, padded_tokens)

    def end(self) -> None:
        if self._open is None:
            return
        idx, n = self._open
        self._open = None
        self._pool[idx][1].record()
        self._pending.append((idx, n))

    def settled_curve(self) -> list[tuple[int, float]]:
        return sorted(
            (n, float(np.median(w)))
            for n, w in self.window.items()
            if self.count[n] >= _ONLINE_MIN_SAMPLES
        )
''', "av/header")

av = sub(av, '''        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
''', '''        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        # Online verify-curve state (see OnlineVerifyCurve).
        self._boot_curves: (
            tuple[list[tuple[int, float]], list[tuple[int, float]]] | None
        ) = None
        self._online: OnlineVerifyCurve | None = None
        self._online_last_rebuild = 0
        self._online_last_log = 0
        if _ONLINE_ENABLED:
            if get_tp_group().world_size != 1:
                logger.warning(
                    "VLLM_DSPARK_ONLINE_VERIFY_CURVE ignored: TP>1 ranks would "
                    "learn different tables."
                )
            else:
                self._online = OnlineVerifyCurve()
                logger.info(
                    "DSpark online verify curve ENABLED (window=%d min_samples=%d "
                    "rebuild_every=%d)",
                    _ONLINE_WINDOW,
                    _ONLINE_MIN_SAMPLES,
                    _ONLINE_REBUILD_EVERY,
                )
''', "av/init")

av = sub(av, '''        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)
''', '''        self._boot_curves = (draft_curve, verify_curve)
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)
        logger.info(
            "DSpark boot verify curve (tokens, ms): %s",
            [(n, round(ms, 3)) for n, ms in verify_curve],
        )
        logger.info(
            "DSpark boot draft curve (reqs, ms): %s",
            [(n, round(ms, 3)) for n, ms in draft_curve],
        )

    # ---- online verify curve hooks (called from the model runner) ----------
    def online_forward_start(self, input_batch: "InputBatch") -> None:
        online = self._online
        if online is None:
            return
        # Only pure verify steps: a prefill in the batch prices attention, not
        # the draft budget, and steps without drafts never consult the table.
        if input_batch.num_draft_tokens <= 0 or input_batch.has_prefill:
            return
        if online.drain():
            self._maybe_rebuild_online()
        online.start(input_batch.num_tokens_after_padding)

    def online_forward_end(self) -> None:
        if self._online is not None:
            self._online.end()

    def _maybe_rebuild_online(self) -> None:
        online = self._online
        assert online is not None and self._boot_curves is not None
        if online.total - self._online_last_rebuild < _ONLINE_REBUILD_EVERY:
            return
        self._online_last_rebuild = online.total
        settled = online.settled_curve()
        if not settled:
            return
        draft_curve, boot_verify = self._boot_curves
        merged = dict(boot_verify)
        merged.update(settled)
        verify_curve = sorted(merged.items())
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        if online.total - self._online_last_log >= _ONLINE_LOG_EVERY:
            self._online_last_log = online.total
            boot = dict(boot_verify)
            rows = [
                (n, round(boot.get(n, float("nan")), 2), round(ms, 2), online.count[n])
                for n, ms in settled
            ]
            logger.info(
                "DSpark online verify curve after %d obs (tokens, boot_ms, "
                "online_ms, n): %s",
                online.total,
                rows,
            )
''', "av/curves")

(here / "adaptive_verification.py").write_text(av)

# ---------------- model_runner.py ----------------
mr = (here / "model_runner.py.orig").read_text()

mr = sub(mr, '''        self.step_timing.record_batch(
            input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL
        )
        self.step_timing.forward_start()
''', '''        self.step_timing.record_batch(
            input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL
        )
        self.step_timing.forward_start()
        if self.adaptive_verification is not None and not dummy_run:
            self.adaptive_verification.online_forward_start(input_batch)
''', "mr/start")

mr = sub(mr, '''        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
''', '''        if self.adaptive_verification is not None and not dummy_run:
            self.adaptive_verification.online_forward_end()

        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
''', "mr/end")

(here / "model_runner.py").write_text(mr)

import ast
for f in ("adaptive_verification.py", "model_runner.py"):
    ast.parse((here / f).read_text())
print("patched + syntax ok")
