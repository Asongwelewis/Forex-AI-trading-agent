"""Proof that 0.5% cannot be raised — not by a config object, not by the environment.

This file exists because hard rule 8 is only a rule if there is no way round it. A cap that
lives in a default argument is a suggestion: someone passes 0.02 during a backtest sweep, the
value ends up in a `.env` on the executor machine, and nothing in the test suite notices. So
every route into the number gets a test here, and adding a new one means adding a case.

Two behaviours, deliberately different, and both are asserted:

* `RiskConfig` **raises** on an over-cap value. Startup is where a risk misconfiguration should
  stop a service, loudly, before anything is sized.
* `position_size` **clamps** an over-cap argument down. It has callers that never built a
  config — backtests, future modules — and it is the last thing between a number and a volume.

Neither of them raises the cap. That is the invariant; the difference in how they refuse is a
matter of where the failure is most useful.
"""

from __future__ import annotations

import dataclasses

import pytest

from fxagent.risk.sizing import MAX_RISK_PER_TRADE, RiskConfig, position_size
from fxagent.risk.symbols import SymbolSpec

EURUSD = SymbolSpec.forex("EURUSD")

#: Every way someone might try to ask for more than half a percent.
OVER_CAP = (0.0051, 0.01, 0.02, 0.05, 0.5, 1.0, 100.0)


class TestConfigCannotRaiseIt:
    def test_the_default_is_the_cap(self) -> None:
        assert RiskConfig().risk_fraction == MAX_RISK_PER_TRADE
        assert MAX_RISK_PER_TRADE == 0.005

    @pytest.mark.parametrize("requested", OVER_CAP)
    def test_constructing_a_config_above_the_cap_raises(self, requested: float) -> None:
        with pytest.raises(ValueError, match="exceeds the absolute cap"):
            RiskConfig(risk_fraction=requested)

    @pytest.mark.parametrize("requested", OVER_CAP)
    def test_the_environment_cannot_raise_it_either(self, requested: float) -> None:
        with pytest.raises(ValueError, match="exceeds the absolute cap"):
            RiskConfig.from_env({"FX_RISK_FRACTION": str(requested)})

    def test_an_absent_environment_variable_gives_the_cap_not_a_larger_default(self) -> None:
        assert RiskConfig.from_env({}).risk_fraction == MAX_RISK_PER_TRADE

    def test_a_malformed_environment_value_raises_rather_than_falling_back(self) -> None:
        """Falling back would run all day against a number nobody chose."""
        with pytest.raises(ValueError, match="invalid risk configuration"):
            RiskConfig.from_env({"FX_RISK_FRACTION": "aggressive"})

    def test_the_config_is_frozen_so_it_cannot_be_raised_after_construction(self) -> None:
        config = RiskConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.risk_fraction = 0.05  # type: ignore[misc]

    def test_replace_goes_back_through_validation(self) -> None:
        """`dataclasses.replace` is the other constructor, and it must not be a side door."""
        with pytest.raises(ValueError, match="exceeds the absolute cap"):
            dataclasses.replace(RiskConfig(), risk_fraction=0.05)

    def test_a_config_at_or_below_the_cap_is_accepted(self) -> None:
        """Sizing *down* is always allowed — hard rule 9 only forbids the other direction."""
        assert RiskConfig(risk_fraction=0.001).risk_fraction == 0.001
        assert RiskConfig(risk_fraction=MAX_RISK_PER_TRADE).risk_fraction == MAX_RISK_PER_TRADE


class TestTheSizerCannotBeTalkedIntoIt:
    @pytest.mark.parametrize("requested", OVER_CAP)
    def test_an_over_cap_argument_sizes_at_the_cap(self, requested: float) -> None:
        over = position_size(100_000.0, requested, 1.1000, 1.0980, EURUSD)
        at_cap = position_size(100_000.0, MAX_RISK_PER_TRADE, 1.1000, 1.0980, EURUSD)
        assert over is not None and at_cap is not None
        assert over.volume == at_cap.volume
        assert over.risk_amount == at_cap.risk_amount

    @pytest.mark.parametrize("requested", OVER_CAP)
    def test_no_over_cap_argument_ever_risks_more_than_the_cap(self, requested: float) -> None:
        size = position_size(100_000.0, requested, 1.1000, 1.0980, EURUSD)
        assert size is not None
        assert size.risk_amount <= 100_000.0 * MAX_RISK_PER_TRADE + 1e-9

    def test_the_config_path_and_the_bare_call_agree(self) -> None:
        config = RiskConfig(reference_equity=5_000.0)
        through_config = config.size(1.1000, 1.0980, EURUSD)
        directly = position_size(5_000.0, MAX_RISK_PER_TRADE, 1.1000, 1.0980, EURUSD)
        assert through_config is not None and directly is not None
        assert through_config.volume == directly.volume
