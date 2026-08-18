"""The report: intervals everywhere, both equity modes, and the caveats stated out loud.

`test_no_metric_is_a_bare_number` is the structural one. Every field that could be a float is an
`Estimate`, so a report cannot be assembled that quietly drops an interval — the type system
carries the rule rather than a reviewer.
"""

from __future__ import annotations

import pytest

from fxagent.backtest.folds import purged_walk_forward
from fxagent.backtest.replay import ReplayConfig, replay
from fxagent.backtest.report import AMBIGUITY_CONCERN, build_report
from fxagent.costs import CostConfig
from fxagent.risk.sizing import RiskConfig
from fxagent.risk.symbols import SymbolSpec
from fxagent.stats.performance import INSUFFICIENT_DATA, Estimate
from tests.backtest.builders import FullWeightRouter, agreeing_pair, bar, series

CONFIG = ReplayConfig(
    spec=SymbolSpec.forex("EURUSD"),
    risk=RiskConfig(reference_equity=100_000.0),
    costs=CostConfig(),
    history_bars=50,
    max_bars_held=10,
)

#: Fire often enough, past the 115-bar classifier warm-up, to get a handful of trades.
FIRE_AT = {130 + step * 20 for step in range(12)}


def choppy(count: int = 420) -> object:
    """Alternating drift, so some trades reach the target and others the stop."""
    prices, price = [], 1.1000
    for index in range(count):
        price += 0.0006 if (index // 13) % 2 == 0 else -0.0006
        prices.append(price)
    return series(
        [
            bar(index, open_=price, high=price + 0.0008, low=price - 0.0008, close=price)
            for index, price in enumerate(prices)
        ]
    )


def a_run():  # noqa: ANN201 - local helper
    return replay(choppy(), CONFIG, strategies=agreeing_pair(FIRE_AT), router=FullWeightRouter())


def a_report():  # noqa: ANN201 - local helper
    result = a_run()
    return result, build_report(
        result, CONFIG.costs, risk_fraction=CONFIG.risk.risk_fraction, paths=200
    )


class TestNoBarePointEstimates:
    def test_no_metric_is_a_bare_number(self) -> None:
        _, report = a_report()
        for metric in (
            report.expectancy_r,
            report.profit_factor,
            report.sharpe,
            report.sortino,
            report.win_rate,
        ):
            assert isinstance(metric, Estimate)
            assert metric.lower_ci <= metric.estimate <= metric.upper_ci

    def test_the_rendered_report_shows_every_interval(self) -> None:
        _, report = a_report()
        text = report.describe()
        assert text.count("[") >= 5
        for name in ("expectancy (R)", "profit factor", "sharpe/trade", "sortino/trade"):
            assert name in text

    def test_win_rate_is_labelled_as_information_only(self) -> None:
        """It is reported because a human asks for it, and it gates nothing."""
        _, report = a_report()
        assert "gates nothing" in report.describe()


class TestBothEquityModes:
    def test_both_drawdown_modes_are_present_and_named(self) -> None:
        _, report = a_report()
        text = report.describe()
        assert "compounded" in text
        assert "additive-R" in text

    def test_the_two_drawdowns_are_separate_distributions(self) -> None:
        _, report = a_report()
        assert "compounded" in report.compounded_drawdown.name
        assert "additive-R" in report.additive_drawdown.name

    def test_the_drawdown_is_a_distribution_not_the_realised_path(self) -> None:
        """One equity curve is one draw; its max drawdown is a sample of size one."""
        _, report = a_report()
        percentiles = report.compounded_drawdown.percentiles
        assert percentiles[5] <= percentiles[50] <= percentiles[95]

    def test_the_budget_guidance_names_the_mode_to_read(self) -> None:
        _, report = a_report()
        assert "should be read against" in report.describe()


class TestCaveats:
    def test_a_fixed_spread_run_says_the_result_is_only_as_good_as_the_guess(self) -> None:
        _, report = a_report()
        assert "configured" in report.spread_note
        assert "guess" in report.spread_note

    def test_an_unconfigured_swap_is_declared_rather_than_assumed_zero(self) -> None:
        _, report = a_report()
        assert "NOT configured" in report.swap_note

    def test_a_configured_swap_says_so(self) -> None:
        result = a_run()
        report = build_report(
            result,
            CostConfig(swap_long_per_lot=-7.0),
            risk_fraction=CONFIG.risk.risk_fraction,
            paths=200,
        )
        assert report.swap_note == "configured and charged"

    def test_the_ambiguity_rate_is_reported_with_its_concern_threshold(self) -> None:
        _, report = a_report()
        assert "intrabar ambiguity" in report.describe()
        assert 0.0 <= report.ambiguity_rate <= 1.0
        assert AMBIGUITY_CONCERN == 0.05

    def test_fold_counts_appear_and_a_zero_purge_is_called_out(self) -> None:
        result = a_run()
        folds = purged_walk_forward(list(result.trades), folds=3)
        report = build_report(
            result,
            CONFIG.costs,
            risk_fraction=CONFIG.risk.risk_fraction,
            folds=folds,
            paths=200,
        )
        assert "purged" in (report.fold_note or "")


class TestInsufficientData:
    def test_a_short_run_is_headlined_as_insufficient(self) -> None:
        _, report = a_report()
        assert report.trades < 100
        assert report.verdict == INSUFFICIENT_DATA
        assert INSUFFICIENT_DATA in report.describe()

    def test_an_empty_run_refuses_to_produce_a_report(self) -> None:
        """Zeros would bury the finding; the replay ledger is where it lives."""
        empty = replay(choppy(), CONFIG, strategies=agreeing_pair(set()))
        with pytest.raises(ValueError, match="produced no trades"):
            build_report(empty, CONFIG.costs, risk_fraction=0.005)
