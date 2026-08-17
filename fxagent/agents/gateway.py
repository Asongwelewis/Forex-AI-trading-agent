"""One provider-agnostic entry point for every LLM call this system makes.

Built for N providers rather than two. `DEFAULT_PROVIDERS` is a table, the fall-through walks
it in order, and a third or fourth entry is a line in that table plus an API key in the
environment — not a branch, and not a refactor. The same is true of the agents that call it:
this module knows about prompts and providers and has never heard of a chartist.

**Sequential, always.** `complete` holds a lock for the whole call, so two coroutines cannot
have requests in flight at once, and each provider carries a minimum interval that is enforced
before the request goes out. Requests per minute is the limit that actually bites on a free
tier — daily volume here is a couple of dozen calls — and a fan-out across ten symbols hits an
RPM ceiling in the first second. The lock is what makes "call them one at a time" a property
of the code rather than a rule every caller has to remember.

**429 is backed off, everything else falls through.** A rate limit means "the same provider,
later", so it is retried on an exponential schedule with full jitter. Any other failure means
"this provider is not going to serve this call", so the next one is tried immediately. Whoever
answers is logged by name, because "which provider produced this narration" is the first
question when one of them starts writing nonsense.

**A hard daily counter refuses call 51 and says so.** Not because the free tiers are tight —
they are not, at this volume — but because a bug that puts an LLM call inside a tight loop is
the one way to burn a free tier in ten minutes, and the counter is the only thing standing
between that bug and the bill. It counts *outbound requests*, so a call that falls through
three providers spends three: those are three requests the provider saw, which is the quantity
that matters. It is per-process and resets with the process; the runaway loop it exists to stop
is a per-process failure, and a cap that survives a restart needs the store and a round trip
before every call.

**Nothing about a prompt is ever logged.** A briefing quotes plan levels and, in time, account
state. Log lines outlive the request, get shipped to a third party and get pasted into issues.
The agent name, the digest and the provider are logged; the content is not, and there is no
debug level at which it becomes loggable.

Responses are cached on the hash of their input. Two agents given the same briefing twice —
which is exactly what a re-run over stored data does — cost one call, not two.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_DAILY_CALL_LIMIT",
    "DEFAULT_PROVIDERS",
    "NVIDIA_NIM_BASE",
    "CallBudget",
    "Completion",
    "DailyCallLimitReached",
    "Gateway",
    "LiteLLMTransport",
    "Prompt",
    "PromptCache",
    "ProviderConfig",
    "ProviderError",
    "RateLimited",
    "RetryPolicy",
    "Transport",
    "providers_from_env",
]

logger = logging.getLogger(__name__)

#: CLAUDE.md's cap. See the module docstring: this is a runaway-loop backstop, not a quota.
DEFAULT_DAILY_CALL_LIMIT = 50


# -- configuration -------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """One provider in the fall-through chain.

    `model` is a LiteLLM model identifier and is configuration, not a constant: providers
    retire model names on their own schedule, so check the id against the provider's current
    catalogue rather than trusting the default below to still resolve.
    """

    name: str
    model: str
    api_key_env: str
    #: OpenAI-compatible base URL, where the provider needs one naming it explicitly. LiteLLM
    #: knows the well-known hosts from the model prefix; NVIDIA NIM is pinned here anyway so
    #: the endpoint this was built against is in the repository rather than in a default that
    #: can move under it.
    api_base: str | None = None
    #: Minimum seconds between two requests to this provider. Sized from the free tier's
    #: requests-per-minute allowance with headroom, because being throttled costs more time
    #: than waiting does.
    min_interval_seconds: float = 4.0
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.min_interval_seconds < 0:
            raise ValueError(
                f"min_interval_seconds must not be negative, got {self.min_interval_seconds}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        """The configured key, or `None` when this provider is simply not set up.

        An unset key is an ordinary state, not an error: the README promises that with no keys
        at all the pass completes on template explanations, and that promise is kept here by
        treating absence as "skip this provider" rather than as a failure to report.
        """
        source = os.environ if env is None else env
        value = str(source.get(self.api_key_env, "")).strip()
        return value or None


#: NVIDIA's OpenAI-compatible endpoint. Free, but it needs Developer Program membership and a
#: verified phone number, so an unset `NVIDIA_API_KEY` is the normal state until someone has
#: been through that.
NVIDIA_NIM_BASE = "https://integrate.api.nvidia.com/v1"

#: The chain, in preference order.
#:
#: Gemini first: its free tier is the most generous and the historian's analogue reasoning is
#: the longest prompt. Groq second, for speed.
#:
#: Then **three rungs on one NVIDIA endpoint**, sharing a key and a base URL and differing only
#: in the model. That is not redundancy for its own sake: a developer reported this endpoint
#: returning internal server errors on trivial prompts, and with a single entry that report
#: describes a total outage of the risk officer's provider. Three rungs make it a fall-through
#: the existing chain already handles, at no extra configuration — same key, no extra account.
#:
#: **The slugs after `nvidia_nim/` were checked against each model's own hosted code sample**
#: on build.nvidia.com, verified 17 Aug 2026. They are worth checking again rather than
#: trusting, because a wrong slug fails as an ordinary provider error and falls quietly to the
#: next rung — the right behaviour, and a silent one:
#:
#:   * `deepseek-ai/deepseek-v4-pro` — free at up to 40 RPM with no daily token cap.
#:   * `nvidia/nemotron-3-ultra-550b-a55b` — **not** `nvidia/nemotron-3-ultra`, which is the
#:     `--served-model-name` in NVIDIA's *self-hosted* NIM serving docs. Two different ids for
#:     one model, and only one of them resolves on the hosted endpoint.
#:   * `z-ai/glm-5.2` — **not** `zai-org/glm-5.2`. `zai-org` is the NGC catalogue team that
#:     publishes the container; `z-ai` is the namespace the hosted API answers to.
#:
#: Go straight to the model URL when re-checking. The catalogue runs past a hundred entries and
#: the homepage cards rotate, so browsing to it is slower and lands on the wrong page.
DEFAULT_PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="gemini",
        model="gemini/gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        min_interval_seconds=6.0,
    ),
    ProviderConfig(
        name="groq",
        model="groq/llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        min_interval_seconds=2.5,
    ),
    ProviderConfig(
        name="nvidia_deepseek",
        model="nvidia_nim/deepseek-ai/deepseek-v4-pro",
        api_key_env="NVIDIA_API_KEY",
        api_base=NVIDIA_NIM_BASE,
        # Free tier is 40 RPM, so 1.5s would clear it. Four is deliberate headroom: being
        # throttled costs more time than waiting does, and this agent makes 20 calls a day.
        min_interval_seconds=4.0,
    ),
    ProviderConfig(
        name="nvidia_nemotron",
        model="nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        api_key_env="NVIDIA_API_KEY",
        api_base=NVIDIA_NIM_BASE,
        min_interval_seconds=4.0,
    ),
    ProviderConfig(
        name="nvidia_glm",
        model="nvidia_nim/z-ai/glm-5.2",
        api_key_env="NVIDIA_API_KEY",
        api_base=NVIDIA_NIM_BASE,
        min_interval_seconds=4.0,
    ),
)


def providers_from_env(
    env: Mapping[str, str] | None = None,
    *,
    providers: Sequence[ProviderConfig] = DEFAULT_PROVIDERS,
) -> tuple[ProviderConfig, ...]:
    """Order the chain by `LLM_PRIMARY_PROVIDER` and `LLM_FALLBACK_PROVIDER`, keeping the rest.

    Named providers move to the front in the order given; everything else keeps its table
    order behind them. Nothing is dropped — a provider left out of the two variables is still
    reachable once the named ones fail, which is what makes adding a fourth key sufficient to
    put it in the chain.
    """
    source = os.environ if env is None else env
    by_name = {config.name: config for config in providers}

    ordered: list[ProviderConfig] = []
    for variable in ("LLM_PRIMARY_PROVIDER", "LLM_FALLBACK_PROVIDER"):
        name = str(source.get(variable, "")).strip().lower()
        if not name:
            continue
        config = by_name.get(name)
        if config is None:
            logger.warning("%s names unknown provider %r; ignoring it", variable, name)
        elif config not in ordered:
            ordered.append(config)

    ordered.extend(config for config in providers if config not in ordered)
    return tuple(ordered)


# -- errors --------------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """A provider did not serve this call. The chain moves on to the next one."""


class RateLimited(ProviderError):
    """A provider refused for rate reasons. The same provider, later — so back off and retry."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DailyCallLimitReached(RuntimeError):
    """The per-process daily call counter refused a request."""


# -- budget --------------------------------------------------------------------------------


class CallBudget:
    """Counts outbound LLM requests against a hard daily cap, and refuses the one over.

    The clock is injected so a test exercises real UTC rollover instead of sleeping through it,
    matching `fxagent.adapters.credits.CreditLedger` — same shape, same reason.
    """

    def __init__(
        self,
        *,
        daily_limit: int = DEFAULT_DAILY_CALL_LIMIT,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if daily_limit < 1:
            raise ValueError(f"daily_limit must be positive, got {daily_limit}")
        self._daily_limit = daily_limit
        self._now = now or (lambda: datetime.now(UTC))
        self._day: date | None = None
        self._spent = 0
        self._by_agent: dict[str, int] = {}
        #: The per-agent limit each agent declared, remembered on first sight. Two different
        #: limits for one agent in one process is a wiring bug, not a preference, and a cap
        #: that could be quietly widened by a later caller is not a cap.
        self._declared: dict[str, int] = {}

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @property
    def spent_today(self) -> int:
        self._roll(self._now())
        return self._spent

    @property
    def remaining_today(self) -> int:
        return max(0, self._daily_limit - self.spent_today)

    def spent_by(self, agent: str) -> int:
        """Outbound requests this agent has made today."""
        self._roll(self._now())
        return self._by_agent.get(agent, 0)

    def spend(
        self,
        *,
        agent: str | None = None,
        agent_limit: int | None = None,
        description: str = "llm call",
    ) -> None:
        """Record one outbound request against both caps, or refuse it.

        `agent_limit` is the per-agent ceiling the caller declares — it travels on the `Prompt`
        rather than being configured here, so the gateway enforces a number without having to
        know which agent it belongs to or why that agent is expensive. The tighter cap is
        checked first, because naming it is what makes the refusal useful.
        """
        self._roll(self._now())

        if agent is not None and agent_limit is not None:
            previous = self._declared.get(agent)
            if previous is not None and previous != agent_limit:
                raise ValueError(
                    f"{agent} has already declared a daily limit of {previous} in this process "
                    f"and is now declaring {agent_limit}; one of the two call sites is wrong, "
                    "and a cap that the next caller can widen is not a cap"
                )
            self._declared[agent] = agent_limit

            if self._by_agent.get(agent, 0) >= agent_limit:
                raise DailyCallLimitReached(
                    f"refusing {description}: {agent} has spent its cap of {agent_limit} call(s) "
                    f"today. Its own limit, tighter than the shared one, and reaching it means "
                    "it was asked far more often than a once-per-evaluation agent should be."
                )

        if self._spent >= self._daily_limit:
            raise DailyCallLimitReached(
                f"refusing {description}: the hard daily cap of {self._daily_limit} LLM "
                f"call(s) is already spent. This limit exists to stop a runaway loop, so "
                f"reaching it means something called far more than a handful of times today."
            )

        self._spent += 1
        if agent is not None:
            self._by_agent[agent] = self._by_agent.get(agent, 0) + 1

    def _roll(self, moment: datetime) -> None:
        today = moment.astimezone(UTC).date()
        if self._day != today:
            if self._day is not None:
                logger.info("llm call budget rolled over: spent %d on %s", self._spent, self._day)
            self._day = today
            self._spent = 0
            self._by_agent = {}


# -- prompts and cache ---------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    """One request, provider-agnostic. `digest` is the cache key and the only safe log handle."""

    agent: str
    system: str
    user: str
    #: Asks the provider for a JSON object where it supports it. Left switchable because not
    #: every model in the chain honours the parameter, and a provider that rejects an unknown
    #: argument would be knocked out of the chain by a formatting preference.
    json_object: bool = True
    #: Provider names to try before the rest of the chain, in order. CLAUDE.md pins an agent to
    #: a provider — chartist to Groq, historian to Gemini — and this is how that arrives without
    #: the gateway learning what a chartist is: a preference travels on the request, and an
    #: unconfigured or failing preferred provider simply falls through to the ordinary chain.
    prefer: tuple[str, ...] = ()
    #: Outbound requests this agent may make in a day, on top of the shared cap. Declared by
    #: the agent's own spec, because the number belongs beside the agent it governs — the
    #: gateway enforces it without knowing whose it is.
    daily_call_limit: int | None = None

    @property
    def digest(self) -> str:
        """Keyed on content only.

        `prefer` and `daily_call_limit` are deliberately excluded: two prompts with identical
        text are the same question, and letting a routing preference or a budget split the cache
        would mean a re-run over stored data paid twice for one answer.
        """
        material = "\0".join([self.agent, self.system, self.user, str(self.json_object)])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Completion:
    """What a provider returned, and who returned it."""

    text: str
    provider: str
    model: str
    generated_at: datetime
    cached: bool = False


class PromptCache:
    """A bounded LRU of completions keyed on the input hash.

    Bounded because the analyst is long-lived enough to accumulate: an unbounded dict keyed on
    every briefing it has ever seen is a slow leak that only shows up in a container's memory
    graph a fortnight later.
    """

    def __init__(self, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max = max_entries
        self._entries: OrderedDict[str, Completion] = OrderedDict()

    def get(self, digest: str) -> Completion | None:
        entry = self._entries.get(digest)
        if entry is None:
            return None
        self._entries.move_to_end(digest)
        return entry

    def put(self, digest: str, completion: Completion) -> None:
        self._entries[digest] = completion
        self._entries.move_to_end(digest)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


# -- transport -----------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """How a prompt actually reaches a provider. The seam every test substitutes at."""

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        """Return the raw response text, or raise `ProviderError` / `RateLimited`."""


#: A response wrapped in a markdown code fence. Stripping it happens here, in the transport,
#: and deliberately not in `schemas.validate_note`: a fence is an envelope the provider put
#: around the payload, where hard rule 5's ban on repair is about the payload itself. Keeping
#: the two apart means the validator has exactly one behaviour — parse or discard — and there
#: is no second place where a "small fix" to a response could be argued for.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


def _unwrap(text: str) -> str:
    match = _FENCE.match(text)
    return match.group("body") if match else text.strip()


class LiteLLMTransport:
    """The real transport. LiteLLM routes to whichever provider the model id names.

    LiteLLM is imported inside the call, not at module scope, and is an optional extra
    (`uv sync --extra llm`). Two reasons, both structural: the hourly Actions job and the
    Vercel bundle install neither it nor its dependency tree for a pass that may make no LLM
    call at all, and — more importantly — `fxagent.agents` stays importable without it, so the
    outage test that proves the template floor holds does not depend on the very library whose
    absence it is meant to survive.
    """

    async def complete(self, prompt: Prompt, provider: ProviderConfig, api_key: str) -> str:
        try:
            import litellm
        except ImportError as error:  # pragma: no cover - depends on the install profile
            raise ProviderError(
                "litellm is not installed; install the optional extra with "
                "`uv sync --extra llm` to enable LLM narration"
            ) from error

        arguments: dict[str, Any] = {
            "model": provider.model,
            "api_key": api_key,
            "timeout": provider.timeout_seconds,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        if provider.api_base is not None:
            arguments["api_base"] = provider.api_base
        if prompt.json_object:
            arguments["response_format"] = {"type": "json_object"}

        try:
            response = await litellm.acompletion(**arguments)
        except Exception as error:  # noqa: BLE001 - classified below, then re-raised as ours
            raise _classify(error) from error

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as error:
            raise ProviderError(f"{provider.name} returned an unreadable response shape") from error

        if not content:
            raise ProviderError(f"{provider.name} returned an empty response")
        return _unwrap(str(content))


def _classify(error: Exception) -> ProviderError:
    """Sort a provider exception into "retry this one" or "try the next one".

    Duck-typed rather than caught by class: importing LiteLLM's exception hierarchy here would
    put the optional dependency back at module scope, which is exactly what the lazy import in
    `LiteLLMTransport` exists to avoid. A status code of 429 and a class name mentioning a rate
    limit are both stable across every provider SDK LiteLLM wraps.
    """
    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    if status == 429 or "ratelimit" in name.replace("_", ""):
        retry_after = getattr(error, "retry_after", None)
        return RateLimited(
            f"{type(error).__name__}: rate limited",
            retry_after=float(retry_after) if isinstance(retry_after, (int, float)) else None,
        )
    return ProviderError(f"{type(error).__name__}: {error}")


# -- the gateway ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to press a rate-limited provider before moving to the next one."""

    attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")

    def delay_for(self, attempt: int) -> float:
        """Un-jittered delay before retrying `attempt` (1-based), capped."""
        if attempt < 1:
            raise ValueError(f"attempt is 1-based, got {attempt}")
        return min(
            self.base_delay_seconds * (self.multiplier ** (attempt - 1)), self.max_delay_seconds
        )


@dataclass
class _ProviderState:
    """When this provider was last called, so the minimum interval can be honoured."""

    last_call: datetime | None = field(default=None)


class Gateway:
    """Calls providers in order, one at a time, within a hard daily budget.

    Returns `None` rather than raising when no provider serves the call. That is the whole
    interface contract: every caller has a deterministic template to fall back to, and an
    exception here would make an unreachable free-tier API able to fail a pass whose actual
    output — the regime, the votes, the plan — never needed a model at all.
    """

    def __init__(
        self,
        providers: Sequence[ProviderConfig] | None = None,
        *,
        transport: Transport | None = None,
        budget: CallBudget | None = None,
        cache: PromptCache | None = None,
        policy: RetryPolicy | None = None,
        env: Mapping[str, str] | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._now = now or (lambda: datetime.now(UTC))
        self._providers = tuple(providers if providers is not None else providers_from_env(env))
        self._transport = transport or LiteLLMTransport()
        self._budget = budget or CallBudget(now=self._now)
        self._cache = cache if cache is not None else PromptCache()
        self._policy = policy or RetryPolicy()
        self._env = env
        self._sleep = sleep
        self._jitter = jitter
        self._state: dict[str, _ProviderState] = {p.name: _ProviderState() for p in self._providers}
        #: Held for the whole of `complete`. See the module docstring on sequencing.
        self._lock = asyncio.Lock()

    @property
    def providers(self) -> tuple[ProviderConfig, ...]:
        return self._providers

    @property
    def budget(self) -> CallBudget:
        return self._budget

    def configured(self) -> tuple[ProviderConfig, ...]:
        """The providers that actually have a key. Empty is a supported, documented state."""
        return tuple(p for p in self._providers if p.api_key(self._env) is not None)

    async def complete(self, prompt: Prompt) -> Completion | None:
        """Serve `prompt` from the cache or the first provider that answers, else `None`."""
        cached = self._cache.get(prompt.digest)
        if cached is not None:
            logger.debug("cache hit for %s prompt %s", prompt.agent, prompt.digest[:12])
            return Completion(
                text=cached.text,
                provider=cached.provider,
                model=cached.model,
                generated_at=cached.generated_at,
                cached=True,
            )

        async with self._lock:
            # Re-checked inside the lock: two agents queued on the same briefing would
            # otherwise both miss and both spend a call for one answer.
            cached = self._cache.get(prompt.digest)
            if cached is not None:
                return Completion(
                    text=cached.text,
                    provider=cached.provider,
                    model=cached.model,
                    generated_at=cached.generated_at,
                    cached=True,
                )
            return await self._call_chain(prompt)

    def _ordered_for(self, prompt: Prompt) -> tuple[ProviderConfig, ...]:
        """Configured providers, with this prompt's preferred ones moved to the front.

        A preference the environment has no key for is skipped silently rather than warned
        about: an agent pinned to a provider nobody has configured is the ordinary state on a
        machine with one key, and the chain behind it is the answer, not a defect.
        """
        candidates = self.configured()
        if not prompt.prefer:
            return candidates
        by_name = {config.name: config for config in candidates}
        preferred = [by_name[name] for name in prompt.prefer if name in by_name]
        return tuple(preferred) + tuple(c for c in candidates if c not in preferred)

    async def _call_chain(self, prompt: Prompt) -> Completion | None:
        candidates = self._ordered_for(prompt)
        if not candidates:
            logger.info(
                "no LLM provider is configured (%s); %s falls back to the deterministic template",
                ", ".join(p.api_key_env for p in self._providers) or "no providers in the chain",
                prompt.agent,
            )
            return None

        for provider in candidates:
            key = provider.api_key(self._env)
            if key is None:  # pragma: no cover - configured() already filtered these out
                continue
            completion = await self._call_provider(prompt, provider, key)
            if completion is not None:
                logger.info(
                    "%s prompt %s served by %s (%s)",
                    prompt.agent,
                    prompt.digest[:12],
                    provider.name,
                    provider.model,
                )
                self._cache.put(prompt.digest, completion)
                return completion

        logger.warning(
            "every configured provider failed for the %s prompt %s; falling back to the "
            "deterministic template",
            prompt.agent,
            prompt.digest[:12],
        )
        return None

    async def _call_provider(
        self, prompt: Prompt, provider: ProviderConfig, key: str
    ) -> Completion | None:
        for attempt in range(1, self._policy.attempts + 1):
            try:
                self._budget.spend(
                    agent=prompt.agent,
                    agent_limit=prompt.daily_call_limit,
                    description=f"{prompt.agent} call to {provider.name}",
                )
            except DailyCallLimitReached as refused:
                # ERROR, not WARNING: reaching a fifty-call ceiling at this volume means a loop,
                # and a loop that is only logged at warning is a loop nobody finds.
                logger.error("%s", refused)
                return None

            await self._space(provider)
            try:
                text = await self._transport.complete(prompt, provider, key)
            except RateLimited as limited:
                if attempt == self._policy.attempts:
                    logger.warning(
                        "%s rate-limited the %s prompt %d time(s); moving to the next provider",
                        provider.name,
                        prompt.agent,
                        attempt,
                    )
                    return None
                delay = self._backoff(attempt, limited.retry_after)
                logger.warning(
                    "%s rate-limited the %s prompt (attempt %d/%d); retrying in %.2fs",
                    provider.name,
                    prompt.agent,
                    attempt,
                    self._policy.attempts,
                    delay,
                )
                await self._sleep(delay)
            except ProviderError as failed:
                logger.warning(
                    "%s did not serve the %s prompt (%s); moving to the next provider",
                    provider.name,
                    prompt.agent,
                    failed,
                )
                return None
            else:
                return Completion(
                    text=text,
                    provider=provider.name,
                    model=provider.model,
                    generated_at=self._now(),
                )
        return None  # pragma: no cover - the loop returns on every path

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Full jitter over the scheduled delay, never shorter than the provider's own ask.

        Full jitter rather than a fixed wait for the same reason `store.retry` uses it: several
        agents rate-limited at the same instant should not march back in lockstep.
        """
        delay = self._jitter(0.0, self._policy.delay_for(attempt))
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    async def _space(self, provider: ProviderConfig) -> None:
        """Wait out this provider's minimum interval before letting a request go out."""
        state = self._state.setdefault(provider.name, _ProviderState())
        moment = self._now()
        if state.last_call is not None:
            elapsed = (moment - state.last_call) / timedelta(seconds=1)
            remaining = provider.min_interval_seconds - elapsed
            if remaining > 0:
                await self._sleep(remaining)
        state.last_call = self._now()
