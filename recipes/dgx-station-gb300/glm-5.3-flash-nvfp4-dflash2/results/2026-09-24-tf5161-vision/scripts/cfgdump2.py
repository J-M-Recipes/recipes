import json, transformers
from sglang.srt.utils.hf_transformers_utils import get_config
c = get_config("/model", trust_remote_code=True)
def norm(v):
    if isinstance(v, type):
        return repr(v)
    if hasattr(v, "to_dict"):
        v = v.to_dict()
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    return v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)
def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    else:
        out[p] = o
    return out
d = norm(c.to_dict())
f = flat(d)
f["__class__"] = f"{type(c).__module__}.{type(c).__name__}"
tc = c.get_text_config() if hasattr(c, "get_text_config") else None
f["__text_class__"] = f"{type(tc).__module__}.{type(tc).__name__}" if tc is not None else None
# attributes the model code reads, whether or not they are serialized
for obj, tag in ((c, "top"), (tc, "text")):
    if obj is None:
        continue
    for k in sorted(set(dir(obj))):
        if k.startswith("_") or callable(getattr(type(obj), k, None)):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v) or hasattr(v, "to_dict"):
            continue
        f[f"__attr_{tag}.{k}"] = norm(v) if not isinstance(v, (dict, list, tuple)) else json.dumps(norm(v), sort_keys=True, default=str)[:400]
json.dump(f, open("/out/sgl-%s.json" % transformers.__version__, "w"), indent=1, sort_keys=True, default=str)
print(transformers.__version__, f["__class__"], f["__text_class__"], len(f))
