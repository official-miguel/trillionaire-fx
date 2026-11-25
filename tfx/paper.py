"""Paper-trading mode: at least 8 weeks, compared with the backtest's expected range.

"Ready for small live use" is shown ONLY when (a) paper trading has run >= 8 weeks,
(b) there are enough closed paper trades to judge, and (c) the paper expectancy
lies inside the backtest's 5-95% expectancy range. Even then: small account first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def paper_status(paper_r: list[float], started: datetime | None, now: datetime,
                 expected_ci: tuple[float, float] | None, min_weeks: int = 8,
                 min_trades: int = 10) -> dict[str, Any]:
    weeks = (now - started).days / 7 if started else 0.0
    exp = float(np.mean(paper_r)) if paper_r else None
    reasons = []
    if weeks < min_weeks:
        reasons.append(f"Paper trading has run {weeks:.1f} of the required {min_weeks} weeks.")
    if len(paper_r) < min_trades:
        reasons.append(f"{len(paper_r)} closed paper trades (need at least {min_trades}).")
    inside = None
    if expected_ci is None:
        reasons.append("No validated strategy provides an expected range yet.")
    elif exp is not None:
        lo, hi = expected_ci
        inside = lo <= exp <= hi
        if not inside:
            reasons.append(f"Paper expectancy {exp:+.2f}R is outside the backtest range "
                           f"[{lo:+.2f}R, {hi:+.2f}R].")
    ready = not reasons
    return {"weeks": round(weeks, 1), "closed_trades": len(paper_r),
            "paper_expectancy_r": exp, "expected_range_r": expected_ci, "inside_range": inside,
            "ready_for_small_live_use": ready,
            "message": ("Paper results are inside the expected range. If you go live, start "
                        "with a SMALL account only." if ready else
                        "NOT ready for live use. " + " ".join(reasons))}
