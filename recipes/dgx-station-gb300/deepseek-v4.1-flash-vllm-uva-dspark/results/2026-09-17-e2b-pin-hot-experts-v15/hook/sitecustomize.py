# sitecustomize.py — bind-mounted over /usr/lib/python3.12/sitecustomize.py
# Keeps Ubuntu's apport hook, then installs pin-hot-experts count/off hook.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os
import sys

_MODE = os.environ.get("PIN_MODE", "").strip().lower()
if _MODE in ("count", "off", "split"):
    _hook_path = os.environ.get("PIN_HOOK", "/w/pin_hot_experts_hook.py")
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("pin_hot_experts_hook", _hook_path)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
    except Exception as _e:
        sys.stderr.write(f"PIN_HOT hook FAILED to install: {_e!r}\n")
