"""The four promises every indicator in the package makes, checked against all of them.

Adding an indicator without adding it to INDICATORS is the failure mode this file is
built to catch, so `test_every_public_indicator_is_covered` asserts the registry matches
`fxagent.indicators.__all__`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

import fxagent.indicators as indicators
from fxagent.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    rolling_percentile,
    rolling_zscore,
    true_range,
)

PERIOD = 5

#: Every public indicator, bound to the same period so warm-up lengths are comparable.
#:
#: An indicator returning several aligned series gets **one entry per series**, keyed
#: `name.band`. Registering a representative band instead would leave the other two
#: unchecked, and the look-ahead promise is not a property of a function — it is a property
#: of every number that function emits.
INDICATORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "adx": lambda f: adx(f["high"], f["low"], f["close"], PERIOD),
    "atr": lambda f: atr(f["high"], f["low"], f["close"], PERIOD),
    "bollinger_bands.upper": lambda f: bollinger_bands(f["close"], PERIOD).upper,
    "bollinger_bands.middle": lambda f: bollinger_bands(f["close"], PERIOD).middle,
    "bollinger_bands.lower": lambda f: bollinger_bands(f["close"], PERIOD).lower,
    "ema": lambda f: ema(f["close"], PERIOD),
    "rolling_percentile": lambda f: rolling_percentile(f["close"], PERIOD),
    "rolling_zscore": lambda f: rolling_zscore(f["close"], PERIOD),
    "true_range": lambda f: true_range(f["high"], f["low"], f["close"]),
}

#: Index of the first non-NaN value each indicator may produce, derived from its formula.
FIRST_VALID: dict[str, int] = {
    "adx": 2 * PERIOD - 1,  # `period` bars to seed DI, then `period` DX values to average
    "atr": PERIOD,  # TR[0] is undefined, so the seed covers TR[1..period]
    "bollinger_bands.upper": PERIOD - 1,
    "bollinger_bands.middle": PERIOD - 1,
    "bollinger_bands.lower": PERIOD - 1,
    "ema": PERIOD - 1,
    "rolling_percentile": PERIOD - 1,
    "rolling_zscore": PERIOD - 1,
    "true_range": 1,  # needs one previous close
}

#: Indicators taking a `period`, and the callable to invoke one with a bad value.
PERIODIC: dict[str, Callable[[pd.DataFrame, int], pd.Series]] = {
    "adx": lambda f, p: adx(f["high"], f["low"], f["close"], p),
    "atr": lambda f, p: atr(f["high"], f["low"], f["close"], p),
    "bollinger_bands": lambda f, p: bollinger_bands(f["close"], p).middle,
    "ema": lambda f, p: ema(f["close"], p),
    "rolling_percentile": lambda f, p: rolling_percentile(f["close"], p),
    "rolling_zscore": lambda f, p: rolling_zscore(f["close"], p),
}


def _public_name(registry_key: str) -> str:
    """`bollinger_bands.upper` -> `bollinger_bands`; every other key is already public."""
    return registry_key.split(".", 1)[0]


def ohlc(rows: int, *, seed: int = 20260810) -> pd.DataFrame:
    """Deterministic, well-formed OHLC bars on a UTC hourly index."""
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0015, rows))
    half_range = np.abs(rng.normal(0.0, 0.0010, rows)) + 0.0002
    open_ = np.concatenate([[close[0]], close[:-1]]) if rows else close

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + half_range,
            "low": np.minimum(open_, close) - half_range,
            "close": close,
        },
        index=pd.date_range("2026-08-03", periods=rows, freq="h", tz=UTC),
    )


def test_every_public_indicator_is_covered() -> None:
    assert {_public_name(key) for key in INDICATORS} == set(indicators.__all__)


# --- 1. warm-up is NaN, and the series is never truncated ----------------------


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_result_keeps_the_callers_index_and_length(name: str) -> None:
    frame = ohlc(60)
    result = INDICATORS[name](frame)

    assert len(result) == len(frame)
    assert result.index.equals(frame.index)
    assert result.dtype == np.dtype("float64")


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_warm_up_is_nan_and_ends_exactly_where_the_formula_says(name: str) -> None:
    frame = ohlc(60)
    result = INDICATORS[name](frame)
    boundary = FIRST_VALID[name]

    assert result.iloc[:boundary].isna().all(), f"{name} produced a value during warm-up"
    assert not np.isnan(result.iloc[boundary]), f"{name} is still NaN at index {boundary}"


# --- 2. short input is a data condition, not an error --------------------------


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_input_shorter_than_the_warm_up_is_all_nan_and_does_not_raise(name: str) -> None:
    """Every length from empty up to one bar short of the boundary, so off-by-one shows."""
    for rows in range(FIRST_VALID[name] + 1):
        result = INDICATORS[name](ohlc(rows))

        assert len(result) == rows, f"{name} changed length with {rows} bars"
        assert result.isna().all(), f"{name} produced a value with only {rows} bars"


# --- 3. no look-ahead ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INDICATORS))
@pytest.mark.parametrize("cut", [1, 6, 11, 25, 47])
def test_truncating_the_input_reproduces_the_prefix_exactly(name: str, cut: int) -> None:
    """The load-bearing test of the package.

    If any value at index i drew on data after i, computing on a prefix would disagree
    with computing on the whole series. Equality is exact, not approximate: identical
    inputs run through identical arithmetic in identical order, so anything less than
    bit-for-bit agreement means the computation saw something it should not have.
    """
    frame = ohlc(60)

    whole = INDICATORS[name](frame).iloc[:cut]
    prefix = INDICATORS[name](frame.iloc[:cut])

    pd.testing.assert_series_equal(whole, prefix, check_exact=True, check_names=False)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_appending_a_future_bar_does_not_change_any_earlier_value(name: str) -> None:
    """The same guarantee from the other direction: new bars only ever append."""
    frame = ohlc(41)

    before = INDICATORS[name](frame.iloc[:-1])
    after = INDICATORS[name](frame).iloc[:-1]

    pd.testing.assert_series_equal(before, after, check_exact=True, check_names=False)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_a_changed_future_bar_cannot_reach_backwards(name: str) -> None:
    """Rewriting the last bar to an absurd value must leave every earlier value alone."""
    frame = ohlc(40)
    tampered = frame.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 5.0

    original = INDICATORS[name](frame).iloc[:-1]
    altered = INDICATORS[name](tampered).iloc[:-1]

    pd.testing.assert_series_equal(original, altered, check_exact=True)


# --- 4. purity -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_mutate_its_input(name: str) -> None:
    frame = ohlc(40)
    untouched = frame.copy(deep=True)

    INDICATORS[name](frame)

    pd.testing.assert_frame_equal(frame, untouched)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_is_deterministic(name: str) -> None:
    frame = ohlc(40)
    pd.testing.assert_series_equal(INDICATORS[name](frame), INDICATORS[name](frame))


# --- period validation ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PERIODIC))
@pytest.mark.parametrize("bad", [0, -1, -20])
def test_a_non_positive_period_raises(name: str, bad: int) -> None:
    with pytest.raises(ValueError, match="period must be"):
        PERIODIC[name](ohlc(40), bad)


@pytest.mark.parametrize("name", sorted(PERIODIC))
def test_a_non_integer_period_raises(name: str) -> None:
    with pytest.raises(TypeError, match="period must be an integer"):
        PERIODIC[name](ohlc(40), 5.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", sorted(PERIODIC))
def test_a_boolean_period_is_not_accepted_as_an_integer(name: str) -> None:
    with pytest.raises(TypeError, match="period must be an integer"):
        PERIODIC[name](ohlc(40), True)  # type: ignore[arg-type]
