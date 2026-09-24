#!/usr/bin/env python3
"""W3a.2 grammar unit test. Run INSIDE lmsysorg/sglang:v0.5.20-cu130 with the patched files bind-mounted and
the model dir mounted at /model (tokenizer only; no weights are loaded). Needs a few MB of GPU.

T5  With the real GLM glm47 full_assistant_ebnf grammar that serving_chat attaches to every chat request,
    compiled by xgrammar on the real tokenizer, the grammar bitmask built over a K-wide linear chain
    (GrammarTree.from_linear_chain(v_tokens), what W3a.2 does) equals the first K rows of the bitmask built over
    the full 7-wide chain, for K=5,4,3,2, many grammar states, and chains with/without disallowed tokens.
    Also checks that the grammar FSM is unchanged after mask generation (DFS rolls back), and that the test is
    meaningful: some rows are actually constrained and some corrupted drafts are rejected.
T5r The same check with the grammar wrapped in ReasonerGrammarObject (what --reasoning-parser glm45 produces).
"""
import json, random, sys, types
import torch

dev = "cuda"
ok = True
def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    ok = ok and bool(cond)

import sglang.srt.speculative.dflash_worker_v2 as W
src = open(W.__file__).read()
check("T1 dflash_worker_v2 is W3a.2", "PATCHED (jmeadlock W3a.2" in src
      and "GrammarTree.from_linear_chain(v_tokens)" in src and "not batch.has_grammar\n" not in src)

from transformers import AutoTokenizer
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.function_call.glm4_moe_detector import GlmSpecialTokenConfig, generate_glm_grammar
from sglang.srt.speculative.spec_utils import GrammarTree, generate_token_bitmask

tok = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
cfg = json.load(open("/model/config.json"))
vocab = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
gcfg = json.load(open("/model/generation_config.json"))
eos = gcfg.get("eos_token_id") or cfg.get("eos_token_id") or cfg.get("text_config", {}).get("eos_token_id")
eos = [e for e in (eos if isinstance(eos, list) else [eos]) if e is not None]
assert vocab and eos, (vocab, eos)
be = XGrammarGrammarBackend(tok, vocab_size=vocab, model_eos_token_ids=eos)
ebnf = generate_glm_grammar(enable_thinking=True, functions=None, special_tokens=GlmSpecialTokenConfig(),
                            chat_template_version="glm47", accommodate_chat_template=True,
                            allow_multiple_assistant_turns=False, required=False, parallel_tool_calls=True)
base = be.dispatch_ebnf(ebnf)
print("vocab", vocab, "grammar", type(base).__name__, "ebnf_lines", ebnf.count("\n") + 1, flush=True)
think_end = tok.convert_tokens_to_ids("</think>")

_SHIFT = torch.arange(32, dtype=torch.int64)
def allowed(mask_row):
    bits = (mask_row.to(torch.int64) & 0xFFFFFFFF)[:, None] >> _SHIFT & 1
    idx = torch.nonzero(bits.flatten()).flatten()
    return idx[idx < vocab].tolist()

def is_allowed(mask_row, t):
    return (int(mask_row[t // 32]) >> (t % 32)) & 1 == 1

def walk(g, steps, rng, prefer):
    """grammar-guided walk: take the preferred token when legal, else a random legal one."""
    out = []
    m = g.allocate_vocab_mask(vocab_size=vocab, batch_size=1, device="cpu")
    pi = 0
    for _ in range(steps):
        if g.is_terminated():
            break
        g.fill_vocab_mask(m, 0)
        t = prefer[pi] if pi < len(prefer) else None
        if t is None or not is_allowed(m[0], t):
            al = allowed(m[0]); t = al[rng.randrange(len(al))]
        else:
            pi += 1
        g.accept_token(t); out.append(t)
    return out

text = ("Let me think about lighthouses and ships.</think>Lighthouses guide ships past rocks. "
        "They were once lit by oil lamps.")
prefer = tok.encode(text, add_special_tokens=False)

def run(label, make_grammar):
    rng = random.Random(1234)
    n_cmp = 0; n_rows_constrained = 0; n_reject = 0; bad = []
    for trial in range(40):
        g0 = make_grammar()
        seq = walk(g0.copy() if hasattr(g0, "copy") else g0, 60, rng, prefer)
        # snapshot states at several positions; each req in the batch is a fresh grammar advanced to p
        positions = sorted(rng.sample(range(1, max(2, len(seq) - 8)), k=min(3, max(1, len(seq) - 9))))
        reqs, chains = [], []
        for p in positions:
            g = make_grammar()
            for t in seq[:p]:
                g.accept_token(t)
            drafts = list(seq[p:p + 6]) + [rng.randrange(vocab)] * max(0, 6 - len(seq[p:p + 6]))
            if rng.random() < 0.6:  # corrupt one draft position with a random (usually disallowed) token
                j = rng.randrange(6); drafts[j] = rng.randrange(vocab)
            chains.append([seq[p - 1]] + drafts)
            reqs.append(types.SimpleNamespace(grammar=g, rid=f"{label}-{trial}-{p}"))
        full = torch.tensor(chains, dtype=torch.int64, device=dev)
        # state fingerprint before
        fp_before = []
        for r in reqs:
            m = r.grammar.allocate_vocab_mask(vocab_size=vocab, batch_size=1, device="cpu"); r.grammar.fill_vocab_mask(m, 0)
            fp_before.append(m.clone())
        m7, _ = generate_token_bitmask(reqs, *GrammarTree.from_linear_chain(full).resolve(), vocab)
        m7 = m7.view(len(reqs), 7, -1).clone()
        for K in (5, 4, 3, 2):
            vk = full[:, :K].contiguous()
            mK, _ = generate_token_bitmask(reqs, *GrammarTree.from_linear_chain(vk).resolve(), vocab)
            mK = mK.view(len(reqs), K, -1)
            n_cmp += 1
            if not torch.equal(mK, m7[:, :K]):
                bad.append((trial, K))
        for i, r in enumerate(reqs):
            m = r.grammar.allocate_vocab_mask(vocab_size=vocab, batch_size=1, device="cpu"); r.grammar.fill_vocab_mask(m, 0)
            if not torch.equal(m, fp_before[i]):
                bad.append((trial, "state_changed"))
            for row in range(7):
                if int((m7[i, row] != -1).sum()) > 0:
                    n_rows_constrained += 1
            # rejected draft = first chain position whose token the parent row disallows
            for row in range(1, 7):
                if not is_allowed(m7[i, row - 1], int(chains[i][row])):
                    n_reject += 1; break
    check(f"T5{label} K-chain mask == first K rows of 7-chain mask", not bad and n_cmp > 0,
          f"comparisons={n_cmp} mismatches={bad[:5]}")
    check(f"T5{label} test is meaningful (constrained rows>0, rejected drafts>0)",
          n_rows_constrained > 0 and n_reject > 0, f"constrained_rows={n_rows_constrained} rejected_chains={n_reject}")

run("", lambda: base.copy())

def make_reasoner():
    g = ReasonerGrammarObject(base.copy(), think_end_ids=[think_end])
    g.maybe_init_reasoning(True)
    return g
try:
    make_reasoner()
    ReasonerGrammarObject.copy  # noqa
    run("r", make_reasoner)
except Exception as e:  # report, don't hide
    check("T5r reasoner-wrapped grammar", False, f"{type(e).__name__}: {e}")

print("W3A2_GRAMMAR " + ("ALL_PASS" if ok else "SOME_FAIL"), flush=True)
sys.exit(0 if ok else 1)
