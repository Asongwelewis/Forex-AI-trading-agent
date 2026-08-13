"""The router and `session_breakout` must permit exactly the same hours.

This is the property that was violated before the two were coupled: the router gated on
London local time while the strategy used fixed UTC hours, so every summer it permitted
07:00-10:59 UTC while the strategy only spoke from 08:00 — losing the first permitted hour
and offering a signal at 11:00 UTC that the router refused.

Testing the property rather than the hours is the point. Asserting "the window is 07:00-10:59
in July" would re-encode the same constants a third time and pass happily if both sides moved
together in the wrong direction. Asserting the two *agree*, at every hour, in three different
DST regimes, is a statement about the coupling and survives any retuning of the window.

The trend gate is held satisfied throughout, so a disagreement can only come from the session
window — which is what is under test.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from fxagent.adapters.base import Bar, BarSeries
from fxagent.regime.router import SESSION_BREAKOUT, RegimeRouter, RouterConfig
from fxagent.regime.sessions import Session, SessionOpening
from fxagent.strategies.base import MarketContext
from fxagent.strategies.session_breakout import ASIAN_HOURS, SessionBreakout
from tests.regime.builders import regime_at
from tests.strategies.builders import BASE_PRICE, bar, flat_bar, h1_series

TRENDING = 30.0
BAND = 0.0010
#: Above the Asian range high of BASE_PRICE + BAND, so the close breaks out.
BREAK_PRICE = BASE_PRICE + 3 * BAND

#: Three DST regimes. In March the US has sprung forward and the UK has not, so London is
#: still on GMT — the breakout window matches January while the overlap does not.
DAYS = [
    pytest.param(date(2026, 7, 15), id="july-london-on-bst"),
    pytest.param(date(2026, 1, 15), id="january-london-on-gmt"),
    pytest.param(date(2026, 3, 12), id="march-gap-week-london-still-gmt"),
]


def _breakout_day(day: date, hour: int) -> BarSeries:
    """A series ending at `hour` on `day`, engineered so a breakout is available there.

    Two prior days of flat bars satisfy the 48-bar history requirement and settle ATR. The
    day's own 00:00-06:00 bars form a complete Asian range, every bar between the range and
    `hour` closes back inside it so the first-break rule is not already spent, and the final
    bar closes above the range. If the strategy stays silent on this series, the reason is
    the session window and nothing else.
    """
    bars: list[Bar] = []
    for days_back in (2, 1):
        previous = day - timedelta(days=days_back)
        bars.extend(
            flat_bar(datetime.combine(previous, time(h), tzinfo=UTC), band=BAND) for h in range(24)
        )

    for h in range(hour + 1):
        moment = datetime.combine(day, time(h), tzinfo=UTC)
        if h == hour and h > max(ASIAN_HOURS):
            bars.append(
                bar(
                    moment,
                    open_=BASE_PRICE,
                    high=BREAK_PRICE,
                    low=BASE_PRICE - BAND,
                    close=BREAK_PRICE,
                )
            )
        else:
            bars.append(flat_bar(moment, band=BAND))

    return h1_series(bars)


def _permits(router: RegimeRouter, moment: datetime) -> bool:
    return router.weights(regime_at(moment, trend_strength=TRENDING))[SESSION_BREAKOUT] > 0.0


def _speaks(strategy: SessionBreakout, day: date, hour: int) -> bool:
    return strategy.generate(_breakout_day(day, hour), MarketContext.neutral()) is not None


@pytest.mark.parametrize("day", DAYS)
def test_the_router_and_the_strategy_permit_the_same_hours(day: date) -> None:
    router, strategy = RegimeRouter(), SessionBreakout()

    disagreements = [
        (hour, permitted, spoke)
        for hour in range(24)
        for permitted, spoke in [
            (
                _permits(router, datetime.combine(day, time(hour), tzinfo=UTC)),
                _speaks(strategy, day, hour),
            )
        ]
        if permitted != spoke
    ]

    assert not disagreements, f"on {day} the router and session_breakout disagree at " + ", ".join(
        f"{hour:02d}:00 UTC (router={'permits' if p else 'blocks'}, "
        f"strategy={'speaks' if s else 'silent'})"
        for hour, p, s in disagreements
    )


@pytest.mark.parametrize("day", DAYS)
def test_the_agreement_is_not_vacuous(day: date) -> None:
    """Both sides saying "no" to all 24 hours would satisfy the test above trivially."""
    router, strategy = RegimeRouter(), SessionBreakout()
    permitted = [
        hour
        for hour in range(24)
        if _permits(router, datetime.combine(day, time(hour), tzinfo=UTC))
    ]
    spoke = [hour for hour in range(24) if _speaks(strategy, day, hour)]

    assert len(permitted) >= 3, f"only {permitted} permitted; the fixture is not exercising much"
    assert permitted == spoke


def test_july_and_january_permit_different_utc_hours() -> None:
    """The regression itself: the agreed window is not the same set of UTC hours."""
    router = RegimeRouter()
    july = [h for h in range(24) if _permits(router, datetime(2026, 7, 15, h, tzinfo=UTC))]
    january = [h for h in range(24) if _permits(router, datetime(2026, 1, 15, h, tzinfo=UTC))]

    assert july == [7, 8, 9, 10]
    assert january == [8, 9, 10, 11]
    assert july != january


def test_the_property_test_has_teeth() -> None:
    """Deliberately decouple the two and confirm the assertion above would catch it.

    Without this, a change that made both sides silent everywhere — or that accidentally
    compared something to itself — would leave the suite green and the coupling unproven.
    """
    router = RegimeRouter(RouterConfig(breakout_window=SessionOpening(Session.LONDON, 12)))
    misaligned = SessionBreakout(SessionOpening(Session.LONDON, 10))
    day = date(2026, 7, 15)

    disagreements = [
        hour
        for hour in range(24)
        if _permits(router, datetime.combine(day, time(hour), tzinfo=UTC))
        != _speaks(misaligned, day, hour)
    ]
    assert disagreements, "a mismatched window must be detectable, or the test proves nothing"
