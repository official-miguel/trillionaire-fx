"""Consensus engine: combine per-strategy signals for one market.

Only strategies whose validator status ``can_vote`` (current PASS, matching rule
hash) are counted. Opposite votes => WAIT ("validated strategies disagree").
Deterministic: votes are sorted by strategy name before combining.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tfx.validation.validators import ValidatorStatus


@dataclass(frozen=True)
class Vote:
    strategy: str
    direction: int        # +1 / -1
    score: float          # 0-100 setup score


@dataclass(frozen=True)
class Consensus:
    direction: int                    # +1, -1, or 0 (no consensus)
    voters: list[str]
    excluded: dict[str, str] = field(default_factory=dict)   # strategy -> why it cannot vote
    conflict: bool = False
    leader: str | None = None         # strategy whose setup is shown

    @property
    def text(self) -> str:
        if self.conflict:
            return "Validated strategies disagree on direction: WAIT."
        if not self.voters:
            return "No validated strategy has a signal: WAIT."
        side = "long" if self.direction == 1 else "short"
        return f"{len(self.voters)} validated strateg{'y' if len(self.voters) == 1 else 'ies'} " \
               f"agree: {side} ({', '.join(self.voters)})."


def consensus(votes: list[Vote], statuses: dict[str, ValidatorStatus]) -> Consensus:
    eligible, excluded = [], {}
    for v in sorted(votes, key=lambda x: x.strategy):
        st = statuses.get(v.strategy)
        if st is None or not st.can_vote:
            excluded[v.strategy] = st.state if st else "NO VALIDATOR"
            continue
        eligible.append(v)
    if not eligible:
        return Consensus(0, [], excluded)
    dirs = {v.direction for v in eligible}
    if len(dirs) > 1:
        return Consensus(0, [v.strategy for v in eligible], excluded, conflict=True)
    leader = max(eligible, key=lambda v: (v.score, v.strategy))
    return Consensus(eligible[0].direction, [v.strategy for v in eligible], excluded,
                     leader=leader.strategy)
