"""Repositories — the only place that builds SQL.

Nothing outside this package composes a query. Strategies, the regime router and the agents
receive plain records and never learn that Postgres exists, which is what allows the store to
be swapped or mocked in tests without touching them.

Two repositories enforce point-in-time correctness structurally rather than by convention:
`EventRepository` and `WindowRepository` take `as_of` as a required first argument on every
read, and neither offers an unfiltered alternative.
"""

from __future__ import annotations

from fxagent.store.repositories.bars import BarRepository
from fxagent.store.repositories.base import Repository, require_utc
from fxagent.store.repositories.evaluations import EvaluationRecord, EvaluationRepository
from fxagent.store.repositories.events import HIGH_IMPACT, EventRecord, EventRepository
from fxagent.store.repositories.heartbeats import HeartbeatRecord, HeartbeatRepository
from fxagent.store.repositories.observations import ObservationRecord, ObservationRepository
from fxagent.store.repositories.trades import BARRIERS, MODES, TradeRecord, TradeRepository
from fxagent.store.repositories.windows import Neighbour, WindowRecord, WindowRepository

__all__ = [
    "BARRIERS",
    "HIGH_IMPACT",
    "MODES",
    "BarRepository",
    "EvaluationRecord",
    "EvaluationRepository",
    "EventRecord",
    "EventRepository",
    "HeartbeatRecord",
    "HeartbeatRepository",
    "Neighbour",
    "ObservationRecord",
    "ObservationRepository",
    "Repository",
    "TradeRecord",
    "TradeRepository",
    "WindowRecord",
    "WindowRepository",
    "require_utc",
]
