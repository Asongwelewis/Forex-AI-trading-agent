"""Regime measurement, and the warm-up honesty that keeps it from inventing a flat market."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fxagent.regime.classifier import ClassifierConfig, RegimeClassifier
from fxagent.regime.sessions import Session
from tests.strategies.builders import flat_run, h1_series

#: A Thursday inside the London session, so the session fields have something to report.
LONDON_MORNING = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)


class TestWarmUp:
    def test_indicators_are_none_before_they_can_be_computed(self) -> None:
        """None means unknown. Returning 0.0 here would be indistinguishable from a flat market."""
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=20)))
        assert regime.trend_strength is None
        assert regime.volatility_percentile is None

    def test_unknown_strength_is_neither_ranging_nor_trending(self) -> None:
        """The gated strategies must stay shut during warm-up rather than guess."""
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=20)))
        assert regime.is_ranging is False
        assert regime.is_trending is False

    def test_required_bars_covers_the_longest_chain(self) -> None:
        """The percentile needs a full window of ATR, and ATR needs its own warm-up first."""
        config = ClassifierConfig()
        assert config.required_bars == config.atr_period + config.volatility_lookback + 1

    def test_enough_history_populates_every_field(self) -> None:
        classifier = RegimeClassifier()
        bars = flat_run(end=LONDON_MORNING, count=classifier.required_bars)
        regime = classifier.classify(h1_series(bars))
        assert regime.trend_strength is not None
        assert regime.volatility_percentile is not None


class TestMeasurement:
    def test_a_flat_market_reads_as_ranging(self) -> None:
        """Identical bars produce no directional movement at all, so ADX is exactly zero."""
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=130)))
        assert regime.trend_strength == pytest.approx(0.0)
        assert regime.is_ranging is True
        assert regime.is_trending is False

    def test_a_reading_between_the_thresholds_is_neither_state(self) -> None:
        """The gap is deliberate: a market can be measured and still be called neither.

        Thresholds are moved around the known ADX of 0.0 rather than hunting for bars that
        produce a mid-range value, so the assertion stays exact.
        """
        straddling = ClassifierConfig(ranging_below=0.0, trending_above=5.0)
        regime = RegimeClassifier(straddling).classify(
            h1_series(flat_run(end=LONDON_MORNING, count=130))
        )
        assert regime.trend_strength == pytest.approx(0.0)
        assert regime.is_ranging is False, "0.0 is not strictly below the 0.0 threshold"
        assert regime.is_trending is False

    def test_constant_volatility_ranks_at_the_top_of_its_own_window(self) -> None:
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=130)))
        assert regime.volatility_percentile == pytest.approx(100.0)


class TestPurity:
    def test_symbol_and_timestamp_come_from_the_bars(self) -> None:
        """No clock is read, so a replayed bar classifies exactly as it did live."""
        bars = h1_series(flat_run(end=LONDON_MORNING, count=130), symbol="GBPUSD")
        regime = RegimeClassifier().classify(bars)
        assert regime.symbol == "GBPUSD"
        assert regime.timestamp == LONDON_MORNING

    def test_classifying_the_same_bars_twice_gives_the_same_regime(self) -> None:
        bars = h1_series(flat_run(end=LONDON_MORNING, count=130))
        classifier = RegimeClassifier()
        assert classifier.classify(bars) == classifier.classify(bars)


class TestSessionFields:
    def test_the_session_label_is_derived_from_the_last_bar(self) -> None:
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=130)))
        assert regime.sessions == (Session.LONDON,)
        assert regime.session is Session.LONDON
        assert regime.in_session(Session.LONDON)
        assert not regime.in_session(Session.OVERLAP)

    def test_a_weekend_bar_reports_a_shut_market_and_no_session(self) -> None:
        saturday = datetime(2026, 1, 17, 12, 0, tzinfo=UTC)
        regime = RegimeClassifier().classify(h1_series(flat_run(end=saturday, count=130)))
        assert regime.market_open is False
        assert regime.sessions == ()
        assert regime.session is None
        assert regime.minutes_until_weekly_close == 0

    def test_minutes_until_close_is_carried_on_the_regime(self) -> None:
        """Thursday 09:00 UTC to the Friday close, which in January is 22:00 UTC.

        37 hours, not 36: the week ends at 17:00 in New York, and New York is on EST here.
        """
        regime = RegimeClassifier().classify(h1_series(flat_run(end=LONDON_MORNING, count=130)))
        assert regime.minutes_until_weekly_close == 37 * 60


class TestConfigValidation:
    def test_overlapping_thresholds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must sit under"):
            ClassifierConfig(ranging_below=30.0, trending_above=25.0)

    def test_thresholds_are_tunable_without_touching_the_classifier(self) -> None:
        strict = ClassifierConfig(ranging_below=5.0, trending_above=6.0)
        regime = RegimeClassifier(strict).classify(
            h1_series(flat_run(end=LONDON_MORNING, count=130))
        )
        assert regime.trend_strength == pytest.approx(0.0)
        assert regime.is_ranging is True
