"""Thin shim so app/05_dashboard.py can import run_query without digit-prefixed module name."""
import importlib.util as _util
from pathlib import Path as _Path

_spec = _util.spec_from_file_location("agent_04", _Path(__file__).parent / "04_agent.py")
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_query = _mod.run_query
DEMO_QUERIES = _mod.DEMO_QUERIES
