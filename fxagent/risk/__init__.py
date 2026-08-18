"""Position sizing and exposure caps.

Sizes are derived from stop distance in risk units, never fixed lots.
Caps are absolute: <=0.5% equity risked per trade, <=2% total open risk.

Two halves, and they behave differently on purpose.

`sizing` **enforces**. There is no argument, config key or environment variable that raises
`MAX_RISK_PER_TRADE`, and a setup that would need more than the cap allows returns `None`
rather than a slightly-too-large volume.

`exposure` **annotates**. It stamps `EXPOSURE_WARNING` past 2% and `CORRELATION_WARNING` on a
book that is one bet three times, and then hands the plan back untouched. This process holds no
broker connection, so there is nothing here to block; the permission layer gates execution and
nothing else does.
"""

from __future__ import annotations

from fxagent.risk.exposure import (
    CORRELATION_WARNING,
    EXPOSURE_WARNING,
    MAX_TOTAL_RISK,
    UNBOUNDED_RISK,
    CurrencyExposure,
    ExposureReport,
    OpenRisk,
    RiskAnnotation,
    assess_exposure,
    currency_exposure,
    total_open_risk,
)
from fxagent.risk.sizing import (
    MAX_RISK_PER_TRADE,
    NOT_SIZEABLE,
    PositionSize,
    RiskConfig,
    format_money,
    position_size,
)
from fxagent.risk.symbols import SymbolSpec, round_down_to_step

__all__ = [
    "CORRELATION_WARNING",
    "EXPOSURE_WARNING",
    "MAX_RISK_PER_TRADE",
    "MAX_TOTAL_RISK",
    "NOT_SIZEABLE",
    "UNBOUNDED_RISK",
    "CurrencyExposure",
    "ExposureReport",
    "OpenRisk",
    "PositionSize",
    "RiskAnnotation",
    "RiskConfig",
    "SymbolSpec",
    "assess_exposure",
    "currency_exposure",
    "format_money",
    "position_size",
    "round_down_to_step",
    "total_open_risk",
]
