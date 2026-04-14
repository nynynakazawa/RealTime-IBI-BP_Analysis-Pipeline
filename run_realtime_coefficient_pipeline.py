from __future__ import annotations

"""Experimental coefficient-search entrypoint.

This wrapper intentionally stays thin:
- source of truth is `Analysis/Data/realtime_sessions/`
- coefficient candidates remain exploratory until `Analysis/AROB/` re-evaluates them
"""

from BP_Analysis.fit_realtime_map_pp_coefficients import main


if __name__ == "__main__":
    raise SystemExit(main())
