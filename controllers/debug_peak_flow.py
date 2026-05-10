# controllers/debug_peak_flow.py
"""
Re-exports the root-level debug_peak_flow module so that any code that
does `from controllers.debug_peak_flow import open_debugger` keeps working.
"""
from ParseHMDBNames import PeakFlowDebugger, open_debugger  # noqa: F401
