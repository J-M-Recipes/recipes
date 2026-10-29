# sitecustomize.py for Card TR2 — bind-mounted over /usr/lib/python3.12/sitecustomize.py.
# Does exactly what the pin-hot hook's sitecustomize does (apport hook + PIN_HOOK install), then installs the
# L2 top-k sharing overlay from $L2_HOOK when L2_SHARE is set. With L2_SHARE unset nothing L2-related loads.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os
import sys


def _load(name, path, install=True):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(name, path)
        _m = _ilu.module_from_spec(_spec)
        sys.modules[name] = _m
        _spec.loader.exec_module(_m)
        if install:
            _m.install()
        return _m
    except Exception as _e:
        sys.stderr.write("%s FAILED to install: %r\n" % (name, _e))
        return None


_MODE = os.environ.get("PIN_MODE", "").strip().lower()
if _MODE in ("count", "off", "split", "adaptive"):
    _load("pin_hot_experts_hook", os.environ.get("PIN_HOOK", "/w/pin_hot_experts_hook.py"))

if os.environ.get("L2_SHARE", "").strip() != "":
    _load("share_topk", os.environ.get("L2_HOOK", "/l2/share_topk.py"))