from __future__ import annotations

import sys
from pathlib import Path


BP_ANALYSIS_DIR = Path(__file__).resolve().parent / "BP_Analysis"
sys.path.insert(0, str(BP_ANALYSIS_DIR))

from fit_realtime_map_pp_coefficients import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
