"""The loop around `run_cycle`: fetch, decide, journal, notify. Never execute.

**What this owns, and it is deliberately little.** Where the bars come from, where the ledger
goes, when to wake, and what to do when a part of the pass fails. The decision itself is
`cycle.run_cycle`, which is pure and shared with the backtest.

**Order matters and is not arbitrary.** Journal first, narrate second, notify third. The ledger
row is the observation, and it is the one thing that cannot be recomputed later — a bar not
recorded is gone the way a collection window is gone. Narration is commentary on a decision
already made, and a notification is a convenience. So a Groq outage or a Telegram 500 costs a
message, never a row.

**Every symbol is independent.** One symbol's failure is caught, counted and journalled as an
error; the others still run. A pass that aborted on the first bad symbol would make the
completeness of the ledger depend on alphabetical order.

**Nothing here can place an order.** Not "does not by default" — cannot. This module imports no
order-placing call, and `tests/trader/test_trader_cannot_execute.py` inspects the import graph
with AST to keep it that way, the same way the collector is held to being data-only. Execution
waits on `fxagent/permission/`, which does not exist (GATE A in CLAUDE.md). When it lands, it
gets its own module and its own pre-flight, and this file gains one call to it — not a flag.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from fxagent.adapters.base import BarSeries
from fxagent.backtest.replay import default_strategies
from fxagent.regime.classifier import RegimeClassifier
from fxagent.regime.router import RegimeRouter
from fxagent.regime.selection import SleeveSelector
from fxagent.risk.symbols import SymbolSpec
from fxagent.store.engine import Database
from fxagent.store.repositories import (
    BarRepository,
    EvaluationRepository,
    HeartbeatRepository,
    TradeRepository,
)
from fxagent.strategies.carry_divergence import TIMEFRAME as DAILY_TIMEFRAME
from fxagent.trader.cycle import (
    CycleConfig,
    CycleResult,
    ledger_row,
    paper_trade,
    run_cycle,
)

__all__ = ["TraderConfig", "TraderService", "TraderStats"]

logger = logging.getLogger(__name__)

SERVICE_NAME = "trader"


class Notifier(Protocol):
    """Somewhere to send a decision. Narrow on purpose — it is an output, not a channel back.

    There is no `receive`, no callback and no approval hook, and that is the whole point.
    A notifier that could carry an instruction back would be a second authorisation path
    beside the permission grant, authenticated by whatever the messaging app happens to
    check. See ADR-005 and `fxagent/alerts/telegram.py`.
    """

    async def send(self, text: str) -> bool: ...


@dataclass(frozen=True)
class TraderConfig:
    """What to watch and how often to look."""

    symbols: tuple[str, ...] = ("EURUSD",)
    timeframe: str = "H1"
    source: str = ""
    #: Bars handed to the pipeline. Must equal `ReplayConfig.history_bars`; see `CycleConfig`.
    history_bars: int = 300
    #: How long to wait between passes. A bar-close-aligned wake would be nicer and is not
    #: worth the complexity yet: the cycle is idempotent per (cycle_id, symbol), so an extra
    #: pass over an unchanged bar updates one row rather than writing a second.
    poll_interval: timedelta = timedelta(minutes=5)
    heartbeat_interval: timedelta = timedelta(minutes=5)
    #: Specs per symbol. A symbol with no spec is refused at startup rather than sized against
    #: a guessed contract size, which would be wrong quietly and in the direction of larger.
    specs: dict[str, SymbolSpec] = field(default_factory=dict)
    reference_equity: float = 1000.0
    #: Read the D1 series for the directional bias. Off means the filter reports "no daily view
    #: was supplied" and suppresses nothing — a materially different system, so it is recorded.
    use_daily_bias: bool = True


@dataclass
class TraderStats:
    """Counters for the heartbeat payload and the final log line."""

    passes: int = 0
    evaluations: int = 0
    fired: int = 0
    actionable: int = 0
    notified: int = 0
    errors: int = 0
    last_error: str = ""

    def as_detail(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "evaluations": self.evaluations,
            "fired": self.fired,
            "actionable": self.actionable,
            "notified": self.notified,
            "errors": self.errors,
            "last_error": self.last_error[:500],
        }


class TraderService:
    """Wakes, evaluates every configured symbol, records what it found, and sleeps."""

    def __init__(
        self,
        *,
        database: Database,
        config: TraderConfig,
        notifier: Notifier | None = None,
        now: Any = None,
    ) -> None:
        missing = [symbol for symbol in config.symbols if symbol not in config.specs]
        if missing:
            raise ValueError(
                f"no SymbolSpec for {', '.join(missing)}. Sizing needs the contract size and "
                "lot step; guessing them is wrong quietly, and in the direction of larger."
            )
        if not config.source:
            raise ValueError(
                "source is required. A bar's identity includes where it came from, and a "
                "trader reading one feed while the backtest measured another is not the "
                "system that was measured."
            )

        self._database = database
        self._config = config
        self._notifier = notifier
        self._now = now or (lambda: datetime.now(UTC))
        self._stats = TraderStats()
        self._stopping = asyncio.Event()
        self._started_at = self._now()

        # Built once. They hold configuration, not state — but rebuilding them per pass would
        # invite someone to make them per-symbol, and a router tuned differently per symbol is
        # a different system per symbol.
        self._classifier = RegimeClassifier()
        self._router = RegimeRouter()
        self._selector = SleeveSelector()
        self._strategies = default_strategies(config.timeframe)

    @property
    def stats(self) -> TraderStats:
        return self._stats

    def __repr__(self) -> str:
        return (
            f"TraderService(symbols={list(self._config.symbols)}, "
            f"timeframe={self._config.timeframe!r}, source={self._config.source!r})"
        )

    # -- lifecycle -------------------------------------------------------------

    def request_stop(self) -> None:
        if not self._stopping.is_set():
            logger.info("trader stop requested; finishing the pass in flight")
            self._stopping.set()

    def install_signal_handlers(self) -> None:
        """Ctrl-C finishes the bar in flight rather than dying mid-write.

        This runs on a home desktop that gets closed, slept and rebooted for updates, so an
        orderly stop is the common case rather than the exceptional one.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                # Windows' proactor loop has no add_signal_handler. This is a Windows service.
                signal.signal(sig, lambda *_: self.request_stop())

    async def run(self) -> TraderStats:
        logger.info(
            "trader starting: symbols=%s timeframe=%s source=%s",
            list(self._config.symbols),
            self._config.timeframe,
            self._config.source,
        )
        await self._heartbeat()
        while not self._stopping.is_set():
            await self.pass_once()
            await self._heartbeat()
            await self._sleep_until_next_pass()
        await self._heartbeat(final=True)
        logger.info("trader stopped cleanly after %d passes", self._stats.passes)
        return self._stats

    async def _sleep_until_next_pass(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._stopping.wait(), timeout=self._config.poll_interval.total_seconds()
            )

    # -- the pass --------------------------------------------------------------

    async def pass_once(self, *, dry_run: bool = False) -> list[CycleResult]:
        """Evaluate every symbol once. One cycle id for the pass, one row per symbol.

        `dry_run` skips the writes and the notification, and skips nothing else — the fetch and
        the whole decision still happen, because a dry run that short-circuits the decision
        proves only that the arguments parse.
        """
        cycle_id = uuid4()
        results: list[CycleResult] = []
        self._stats.passes += 1

        for symbol in self._config.symbols:
            if self._stopping.is_set():
                break
            try:
                result = await self._evaluate(symbol, cycle_id=cycle_id)
                if result is None:
                    continue
                if dry_run:
                    logger.info("dry run: %s", _one_line(result))
                else:
                    # Inside the try, and before `results`. A decision that was made and not
                    # recorded is an observation lost — so a journal failure must not be
                    # reported as a completed evaluation, and must not end the pass for the
                    # symbols after it either.
                    await self._journal(result)
            except Exception as exc:  # noqa: BLE001 - one symbol must not end the pass
                self._stats.errors += 1
                self._stats.last_error = f"{symbol}: {exc}"
                logger.exception("evaluation failed for %s", symbol)
                continue

            results.append(result)
            self._stats.evaluations += 1
            self._stats.fired += int(result.fired)
            self._stats.actionable += int(result.actionable)

            # Outside the try on purpose: the row is safe by now, and a notifier is a
            # convenience. `_notify` swallows its own failures for the same reason.
            if not dry_run:
                await self._notify(result)

        return results

    async def _evaluate(self, symbol: str, *, cycle_id) -> CycleResult | None:
        bars = await self._fetch(symbol, self._config.timeframe, self._config.history_bars)
        if not len(bars):
            logger.warning(
                "no %s %s bars from %s; nothing to evaluate",
                symbol,
                self._config.timeframe,
                self._config.source,
            )
            return None

        daily: BarSeries | None = None
        if self._config.use_daily_bias:
            # A failure to read D1 must not lose the intraday evaluation. The bias then reports
            # "no daily view was supplied", which is recorded — an absent filter is visible in
            # the ledger rather than looking like a filter that had no opinion.
            try:
                daily = await self._fetch(symbol, DAILY_TIMEFRAME, 120)
            except Exception:  # noqa: BLE001
                logger.exception("could not read the daily series for %s", symbol)

        config = CycleConfig(
            spec=self._config.specs[symbol],
            history_bars=self._config.history_bars,
        )
        return run_cycle(
            bars,
            config=config,
            equity=self._config.reference_equity,
            strategies=self._strategies,
            classifier=self._classifier,
            router=self._router,
            selector=self._selector,
            daily_bars=daily,
            cycle_id=cycle_id,
        )

    async def _fetch(self, symbol: str, timeframe: str, count: int) -> BarSeries:
        async with self._database.session() as session:
            return await BarRepository(session).latest_bars(
                symbol, timeframe, count, source=self._config.source
            )

    async def _journal(self, result: CycleResult) -> None:
        """The row goes in first and is the only step whose failure is worth raising on.

        A cycle that decided and was not recorded is an observation lost the way a missed
        collection window is lost, so this does not swallow. Everything after it does.

        An actionable cycle also opens a **paper** trade in the same transaction, so the
        evaluation and its trade land together or neither does — the property the store speaks
        Postgres directly to be able to express. Without that row the resolver has nothing to
        close, which means no outcome, no expectancy and no label: the journal would accumulate
        recommendations forever and never learn whether any of them were right.

        `mode="ADVISORY"` is not a formality. `trades_no_live_mode` is a database constraint,
        so a row claiming to be live is rejected by Postgres and not merely by this code.
        """
        async with self._database.begin() as session:
            evaluation_id = await EvaluationRepository(session).record(**ledger_row(result))

            if result.actionable:
                await TradeRepository(session).open_trade(
                    evaluation_id=evaluation_id,
                    **paper_trade(result, timeframe=self._config.timeframe),
                )

        object.__setattr__(result, "evaluation_id", evaluation_id)

    async def _notify(self, result: CycleResult) -> None:
        """Only actionable cycles are sent. Every cycle is journalled.

        The ledger is exhaustive precisely so the notifications do not have to be: a message on
        every silent bar is a message nobody reads, and the refusals are still all there to be
        queried. This is the only place the two audiences differ.
        """
        if self._notifier is None or not result.actionable:
            return
        try:
            if await self._notifier.send(_card(result)):
                self._stats.notified += 1
        except Exception:  # noqa: BLE001 - a notifier is a convenience, never the record
            logger.exception("could not send the notification for %s", result.symbol)

    # -- liveness --------------------------------------------------------------

    async def _heartbeat(self, *, final: bool = False) -> None:
        """Uptime is part of the record, not decoration.

        ADR-005 accepted that this only runs while the desktop is on. A journal that does not
        say when the agent was awake will read as though every unrecorded hour was a quiet
        market, and every metric derived from it would be silently conditioned on the machine
        happening to be running.
        """
        detail = self._stats.as_detail()
        detail["final"] = final
        detail["symbols"] = list(self._config.symbols)
        try:
            async with self._database.begin() as session:
                await HeartbeatRepository(session).beat(
                    SERVICE_NAME,
                    now=self._now(),
                    started_at=self._started_at,
                    detail=detail,
                )
        except Exception:  # noqa: BLE001 - a missed beat must not stop the trading loop
            logger.exception("heartbeat failed")


def _one_line(result: CycleResult) -> str:
    verdict = "FIRED" if result.fired else "no trade"
    if result.fired and result.plan is None:
        verdict = f"FIRED but not sizeable ({result.sizing_note})"
    return (
        f"{result.symbol} {result.timeframe} @ {result.timestamp.isoformat()}: "
        f"{verdict} — {result.selection.diagnostics.get('reason', '')}"
    )


def _card(result: CycleResult) -> str:
    """The message a human reads. Advisory wording, because that is what it is.

    It says what the system would do and states plainly that it has not done it. A card that
    read like a confirmation would be the first step towards someone treating it as one.
    """
    plan = result.plan
    signal = result.selection.signal
    if plan is None or signal is None:  # pragma: no cover - guarded by `actionable`
        return _one_line(result)

    primary = signal.primary
    lines = [
        f"{result.symbol} {signal.direction} — {primary.strategy_name}",
        f"entry {primary.entry_price:g}  stop {primary.stop_loss:g}  "
        f"target {primary.take_profit:g}",
        f"volume {plan.volume:g} lots · risking {plan.risk_fraction * 100:.2f}% "
        f"({plan.risk_amount:.2f})",
        f"confidence {signal.confidence:.2f} · router weight {signal.total_weight:.2f}",
        f"bar {result.timestamp.isoformat()}",
        "",
        result.selection.diagnostics.get("reason", ""),
        "",
        "ADVISORY — nothing has been placed. This system cannot execute.",
    ]
    return "\n".join(lines)
