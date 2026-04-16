from __future__ import annotations

"""Experimental coefficient-search entrypoint.

This wrapper intentionally stays thin:
- source of truth is `Analysis/Data/realtime_sessions/`
- active methods are fixed to `RTBP / SinBP_D / SinBP_M`
- baseline-family experimental branches are preserved in code but disabled by default
"""

from BP_Analysis.fit_realtime_map_pp_coefficients import main


if __name__ == "__main__":
    raise SystemExit(main())
