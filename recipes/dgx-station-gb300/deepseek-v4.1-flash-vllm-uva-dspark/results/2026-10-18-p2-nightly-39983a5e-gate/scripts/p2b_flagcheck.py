import vllm.platforms as P
from vllm.platforms.cpu import CpuPlatform
P._current_platform = CpuPlatform()
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
p = FlexibleArgumentParser(); p = AsyncEngineArgs.add_cli_args(p)
for a in (["--model", "x"], ["--model", "x", "--no-swa-bounded-replay"], ["--model", "x", "--swa-bounded-replay"]):
    ns = p.parse_args(a); print("FLAGCHECK", a[2:], "swa_bounded_replay=", ns.swa_bounded_replay)
