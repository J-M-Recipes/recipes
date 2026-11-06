#!/bin/bash
# run_matrix.sh: offline DSML tool-parser matrix in a lane's own image (CPU only, --network none, no GPU, no server).
# Runs stock, patched and kill-switch arms, then gates them with compare.py.
#   ENGINE=vllm|sglang  IMAGE=<serving image>  MODEL_DIR=<checkpoint dir (tokenizer only is read)>
#   OVERLAY=<path to the overlay file>  [OUT=./out]
# vLLM:   PARSER_PATH defaults to /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v41.py (stock vllm-openai);
#         the spark-ds41f fork image needs PARSER_PATH=/opt/spark3/candidate/vllm/vllm/parser/deepseek_v41.py
# SGLang: PARSER_PATH defaults to /sgl-workspace/sglang/python/sglang/srt/function_call/deepseekv41_detector.py
set -euo pipefail
: "${ENGINE:?vllm or sglang}" "${IMAGE:?}" "${MODEL_DIR:?}" "${OVERLAY:?}"
HERE=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$PWD/out}; mkdir -p "$OUT"
if [ "$ENGINE" = vllm ]; then
  P=${PARSER_PATH:-/usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v41.py}; S=matrix.py; K=VLLM_DSV41_STRUCTURAL_DSML
else
  P=${PARSER_PATH:-/sgl-workspace/sglang/python/sglang/srt/function_call/deepseekv41_detector.py}; S=matrix_sglang.py; K=SGLANG_DSV41_STRUCTURAL_DSML
fi
base=(docker run --rm --network none --entrypoint python3 -e CUDA_VISIBLE_DEVICES= -v "$MODEL_DIR:/model:ro" -v "$HERE:/p:ro")
ov=(-v "$(cd "$(dirname "$OVERLAY")" && pwd)/$(basename "$OVERLAY"):$P:ro")
"${base[@]}" "$IMAGE" /p/$S > "$OUT/stock.jsonl" 2> "$OUT/stock.err"
"${base[@]}" "${ov[@]}" -e $K=1 "$IMAGE" /p/$S > "$OUT/patched.jsonl" 2> "$OUT/patched.err"
"${base[@]}" "${ov[@]}" -e $K=0 "$IMAGE" /p/$S > "$OUT/killswitch.jsonl" 2> "$OUT/killswitch.err"
python3 "$HERE/compare.py" "$OUT/stock.jsonl" "$OUT/patched.jsonl" "$OUT/killswitch.jsonl" | tee "$OUT/VERDICT.txt"
