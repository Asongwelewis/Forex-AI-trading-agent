"""The spread sampler: when it samples, what it records, and what it must not reach.

Driven with a fake terminal rather than a real one. MT5 is Windows-only and needs a running
platform, so a suite that required it would be a suite that only one machine can run — and the
logic worth testing here is the window, the dedup key and the record shape, none of which are
about MetaTrader.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, time
from pathlib import Path
from types import SimpleNamespace

import pytest

import fxagent
from fxagent.spreadwatch import (
    LONDON_WINDOW_END_UTC,
    LONDON_WINDOW_START_UTC,
    SpreadSample,
    calibrate,
    in_window,
    sample_once,
    subscribe,
)

#: 18 Aug 2026 is a Tuesday; 22 Aug is the Saturday after it.
TUESDAY = datetime(2026, 8, 18, tzinfo=UTC)
SATURDAY = datetime(2026, 8, 22, tzinfo=UTC)


class FakeAdapter:
    """Applies the Exness suffix, which is the only thing `sample_once` asks an adapter for."""

    def _broker_symbol(self, symbol: str) -> str:
        return f"{symbol}m"


class FakeMT5:
    """A terminal that answers for the symbols it was given and nothing else."""

    def __init__(self, quotes: dict[str, tuple[float, float]], *, spread_points: int = 8) -> None:
        self._quotes = quotes
        self._spread = spread_points
        self.selected: list[str] = []

    def symbol_select(self, name: str, enable: bool) -> bool:  # noqa: FBT001 - mirrors the MT5 API
        self.selected.append(name)
        return name in self._quotes

    def symbol_info_tick(self, name: str):  # noqa: ANN201 - mirrors the MT5 API
        if name not in self._quotes:
            return None
        bid, ask = self._quotes[name]
        return SimpleNamespace(bid=bid, ask=ask)

    def symbol_info(self, name: str):  # noqa: ANN201 - mirrors the MT5 API
        if name not in self._quotes:
            return None
        return SimpleNamespace(spread=self._spread, point=1e-05, spread_float=True)


def at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute)


class TestTheWindow:
    def test_the_window_is_the_london_open_and_the_hour_before_it(self) -> None:
        assert time(7, 0) == LONDON_WINDOW_START_UTC
        assert time(11, 0) == LONDON_WINDOW_END_UTC

    @pytest.mark.parametrize("hour", [7, 8, 9, 10])
    def test_hours_inside_the_window_sample(self, hour: int) -> None:
        assert in_window(at(TUESDAY, hour))

    @pytest.mark.parametrize("hour", [6, 11, 12, 21])
    def test_hours_outside_it_do_not(self, hour: int) -> None:
        assert not in_window(at(TUESDAY, hour))

    def test_the_start_is_inclusive_and_the_end_is_not(self) -> None:
        assert in_window(at(TUESDAY, 7, 0))
        assert not in_window(at(TUESDAY, 11, 0))

    def test_weekends_are_excluded_rather_than_filtered_later(self) -> None:
        """The terminal keeps reporting a stale quote when the market is shut, and a few
        hundred frozen Saturday rows would sit in the middle of the distribution looking like
        an unusually tight London morning."""
        assert SATURDAY.weekday() == 5
        assert not in_window(at(SATURDAY, 9))

    def test_a_non_utc_instant_is_converted_before_it_is_judged(self) -> None:
        from datetime import timedelta, timezone

        eight_utc = at(TUESDAY, 8).astimezone(timezone(timedelta(hours=5)))
        assert in_window(eight_utc)


class TestSampling:
    def test_a_sample_records_both_the_reported_and_derived_spread(self) -> None:
        """They should agree. Where they do not, the symbol's point is misconfigured, and
        storing both is what makes that visible instead of reconciling it away."""
        mt5 = FakeMT5({"EURUSDm": (1.15772, 1.15780)}, spread_points=8)
        samples = sample_once(FakeAdapter(), ("EURUSD",), mt5)

        assert len(samples) == 1
        sample = samples[0]
        assert sample.spread_points == 8
        assert sample.derived_points == pytest.approx(8.0)
        assert sample.broker_symbol == "EURUSDm"

    def test_a_symbol_without_a_tick_is_skipped_not_fatal(self) -> None:
        """The poller must survive a week unattended; one dying on a null tick measures
        nothing at all."""
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578)})
        samples = sample_once(FakeAdapter(), ("EURUSD", "EURGBP"), mt5)
        assert [s.symbol for s in samples] == ["EURUSD"]

    def test_a_non_positive_quote_is_refused(self) -> None:
        mt5 = FakeMT5({"EURUSDm": (0.0, 1.1578)})
        assert sample_once(FakeAdapter(), ("EURUSD",), mt5) == []

    def test_the_timestamp_is_truncated_to_the_minute(self) -> None:
        """The dedup key is (symbol, instant); seconds would make every restart a new row."""
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578)})
        sample = sample_once(FakeAdapter(), ("EURUSD",), mt5)[0]
        assert sample.sampled_at.second == 0
        assert sample.sampled_at.microsecond == 0

    def test_the_row_carries_the_broker_symbol_not_just_the_pair(self) -> None:
        """The suffix identifies the account type, and two Exness account types quote the same
        pair differently — a sample that forgot which one cannot be compared."""
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578)})
        row = sample_once(FakeAdapter(), ("EURUSD",), mt5)[0].as_row()
        assert row["broker_symbol"] == "EURUSDm"
        assert row["symbol"] == "EURUSD"
        assert row["source"] == "mt5_exness"


class TestSubscription:
    def test_every_symbol_is_added_to_market_watch_once(self) -> None:
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578), "GBPUSDm": (1.30, 1.3001)})
        assert subscribe(FakeAdapter(), ("EURUSD", "GBPUSD"), mt5) == ["EURUSDm", "GBPUSDm"]

    def test_subscription_happens_before_sampling_not_during_it(self) -> None:
        """Measured: EURGBPm returned no tick when selected and read in the same instant, and
        ticked normally seconds later. Selecting inside the loop loses the first reading of
        every symbol and every reading of a symbol whose first tick is slow."""
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578)})
        sample_once(FakeAdapter(), ("EURUSD",), mt5)
        assert mt5.selected == []

    def test_a_symbol_that_cannot_be_selected_is_reported(self) -> None:
        mt5 = FakeMT5({"EURUSDm": (1.1577, 1.1578)})
        assert subscribe(FakeAdapter(), ("EURUSD", "NOPE"), mt5) == ["EURUSDm"]


class TestItIsMeasurementOnly:
    """The sampler records dealing conditions. It must never become an input to a decision."""

    PACKAGE_ROOT = Path(fxagent.__file__).parent
    DECISION_PACKAGES = ("strategies", "regime", "risk", "patterns", "indicators", "backtest")

    @staticmethod
    def _imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
        return found

    def test_no_deciding_module_imports_the_sampler(self) -> None:
        """A spread sample has no publication time and no bar behind it. It is a measurement of
        the venue, not a feature — and the moment a strategy reads one, it is both."""
        for package in self.DECISION_PACKAGES:
            for module in sorted((self.PACKAGE_ROOT / package).rglob("*.py")):
                reached = {n for n in self._imports(module) if "spreadwatch" in n}
                assert not reached, f"{module.relative_to(self.PACKAGE_ROOT)} imports {reached}"

    def test_the_sampler_reaches_no_analysis_code(self) -> None:
        """It talks to the terminal and the store, and to nothing that forms an opinion."""
        forbidden = (
            "fxagent.strategies",
            "fxagent.regime",
            "fxagent.indicators",
            "fxagent.backtest",
        )
        reached = {
            name
            for name in self._imports(self.PACKAGE_ROOT / "spreadwatch.py")
            if name.startswith(forbidden)
        }
        assert not reached, f"fxagent/spreadwatch.py imports {sorted(reached)}"


class TestTheRecordShape:
    def test_an_inverted_quote_is_representable_but_the_database_rejects_it(self) -> None:
        """The dataclass does not validate; migration 0012's check constraint does. Asserted here
        so nobody adds a second, softer validation in Python and assumes it is the one enforcing."""
        sample = SpreadSample(
            symbol="EURUSD",
            broker_symbol="EURUSDm",
            sampled_at=at(TUESDAY, 9),
            bid=1.1580,
            ask=1.1570,
            spread_points=-10,
            point=1e-05,
            spread_float=True,
        )
        assert sample.derived_points < 0

    def test_a_zero_point_does_not_divide_by_zero(self) -> None:
        sample = SpreadSample(
            symbol="EURUSD",
            broker_symbol="EURUSDm",
            sampled_at=at(TUESDAY, 9),
            bid=1.1577,
            ask=1.1578,
            spread_points=8,
            point=0.0,
            spread_float=True,
        )
        assert sample.derived_points == 0.0


class TestCalibration:
    def _sample(self, spread_pips: float, *, symbol: str = "EURUSD") -> SpreadSample:
        bid = 1.1000
        return SpreadSample(
            symbol=symbol,
            broker_symbol=f"{symbol}m",
            sampled_at=at(TUESDAY, 9),
            bid=bid,
            ask=bid + spread_pips * 0.0001,
            spread_points=round(spread_pips * 10),
            point=1e-05,
            spread_float=True,
        )

    def test_calibration_keeps_the_tail_percentiles(self) -> None:
        report = calibrate([self._sample(value) for value in (1.0, 1.0, 2.0, 10.0)])
        assert report.samples == 4
        assert report.p50_pips == pytest.approx(1.5)
        assert report.p90_pips > report.p50_pips
        assert report.max_pips == pytest.approx(10.0)
        assert "p95=" in report.render()

    def test_calibration_does_not_pool_symbols_or_sources(self) -> None:
        with pytest.raises(ValueError, match="exactly one symbol and source"):
            calibrate([self._sample(1.0), self._sample(1.0, symbol="GBPUSD")])
