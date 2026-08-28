"""The reported risk fraction must never sit above the cap, not even by a float ulp.

Found while wiring the trader: a maximally-sized EURUSD trade reported
`risk_fraction == 0.005000000000000004` against a cap of `0.005`. Four femto-percent of equity
is not a risk problem, and it is a sharp reporting problem — `ExecutionPlan.over_trade_cap` is
a strict `>`, so every normally-sized trade would have been flagged as breaching hard rule 8
and the risk officer would have said so on every card until nobody read them.

The clamp in `position_size` is sound rather than cosmetic: `usable_volume` rounds DOWN to the
lot step, so `volume * loss_per_lot <= risk_budget` holds as real arithmetic and any excess is
error introduced by the divide-then-multiply round trip.

These cases use prices whose differences are inexact in binary — 1.1000 - 1.0980 is
0.0020000000000000018, not 0.002 — because that is where the error comes from and a test on
tidy numbers would pass without exercising anything.
"""

from __future__ import annotations

import pytest

from fxagent.risk.sizing import MAX_RISK_PER_TRADE, position_size
from fxagent.risk.symbols import SymbolSpec

SPEC = SymbolSpec.forex("EURUSD")


@pytest.mark.parametrize("equity", [1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 33_333.33])
@pytest.mark.parametrize(
    ("entry", "stop"),
    [
        (1.1000, 1.0980),  # the case that surfaced it
        (1.1000, 1.0975),
        (1.23456, 1.23001),
        (0.9, 0.8993),
    ],
)
def test_reported_fraction_never_exceeds_the_cap(
    equity: float, entry: float, stop: float
) -> None:
    size = position_size(equity, MAX_RISK_PER_TRADE, entry, stop, SPEC)
    if size is None:
        pytest.skip("not sizeable at this equity, which is a different behaviour")

    assert size.risk_fraction <= MAX_RISK_PER_TRADE
    assert size.risk_amount <= equity * MAX_RISK_PER_TRADE


def test_the_execution_plan_does_not_flag_a_normal_trade_as_over_cap() -> None:
    """The consequence the clamp exists to prevent, asserted at the place it would show up."""
    from fxagent.agents.schemas import ExecutionPlan
    from fxagent.risk.exposure import MAX_TOTAL_RISK

    size = position_size(100_000.0, MAX_RISK_PER_TRADE, 1.1000, 1.0980, SPEC)
    assert size is not None

    plan = ExecutionPlan(
        volume=size.volume,
        risk_fraction=size.risk_fraction,
        risk_amount=size.risk_amount,
        stop_distance=size.stop_distance,
        total_open_risk=size.risk_fraction,
        max_risk_per_trade=MAX_RISK_PER_TRADE,
        max_total_risk=MAX_TOTAL_RISK,
    )

    assert not plan.over_trade_cap, (
        "a trade sized exactly at the cap must not report as breaching it; that is the "
        "normal case, and a warning that fires on the normal case trains people to ignore it"
    )


def test_the_clamp_does_not_hide_a_genuine_overshoot() -> None:
    """Sanity: the clamp discards error, and there is no path that could produce a real one.

    Rounding down is what makes that true, so this asserts the rounding rather than trusting
    the comment. A volume above the unrounded ideal would be a real breach, and would mean the
    clamp was masking something instead of tidying it.
    """
    equity, entry, stop = 100_000.0, 1.1000, 1.0980
    size = position_size(equity, MAX_RISK_PER_TRADE, entry, stop, SPEC)
    assert size is not None

    ideal = (equity * MAX_RISK_PER_TRADE) / (
        SPEC.money_per_lot(abs(entry - stop)) * size.quote_to_account_rate
    )
    assert size.volume <= ideal + 1e-12, "volume must never round up past the risk budget"
