"""LLM agent abstractions used by the forecasting system.

This module now supports two kinds of agents:

- LLMAgent (default, deterministic): a local, deterministic pseudo-agent that
  produces stable outputs without calling external APIs. Great for offline
  testing and debugging.
- RealLLMAgent: a thin wrapper over ``forecasting_tools.GeneralLlm`` that
  performs real API calls to providers (OpenAI, OpenRouter, Anthropic, etc.)
  depending on your model string and environment keys. It formats prompts so
  that outputs match the committee workflow's expected structures.

By default the committee uses the deterministic agents. To enable real API
calls for the committee engine, set the env var ``COMMITTEE_LLM_MODELS`` to a
comma-separated list of model identifiers (e.g. "openrouter/gpt-4.1-mini, openai/gpt-4o-mini").

Optional env vars for tuning the real agents:
- COMMITTEE_LLM_TEMPERATURE (float, default: None)
- COMMITTEE_LLM_TIMEOUT (int seconds, default: 40)
- COMMITTEE_LLM_TRIES (int, default: 2)
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
import asyncio
from typing import List, Sequence
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMAgent:
    """Simple deterministic agent used for simulation.

    The ``generate`` method returns text matching the expected structure for
    the different forecasting prompts.  A hash of the prompt and agent name is
    used to seed a local random generator so results are stable across runs.
    """

    name: str
    weight: float = 1.0

    async def generate(self, prompt: str, mode: str, **kwargs) -> str:
        seed_input = (self.name + mode + prompt).encode()
        seed = int(hashlib.md5(seed_input).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        logger.info(
            "[Committee][Agent] %s generate() called | mode=%s | seed=%s | kwargs=%s",
            self.name,
            mode,
            seed,
            kwargs,
        )
        # Log the exact prompt sent to this agent
        logger.info(
            "[Committee][Agent] %s PROMPT (%s):\n%s",
            self.name,
            mode,
            prompt,
        )

        if mode == "scoping":
            topic = kwargs.get("topic", "query")
            out = f"- {topic} history\n- {topic} recent developments"
            # Log the exact response
            logger.info(
                "[Committee][Agent] %s RESPONSE (%s):\n%s", self.name, mode, out
            )
            return out

        if mode == "binary":
            prob = rnd.randint(1, 99)
            out = f"Probability: {prob}%"
            logger.info(
                "[Committee][Agent] %s RESPONSE (%s):\n%s", self.name, mode, out
            )
            return out

        if mode == "multiple_choice":
            n = kwargs.get("n_options", 2)
            raw = [rnd.random() for _ in range(n)]
            total = sum(raw)
            percentages = [round(x / total * 100, 1) for x in raw]
            joined = ", ".join(f"{p}%" for p in percentages)
            out = f"Probabilities: [{joined}]"
            logger.info(
                "[Committee][Agent] %s RESPONSE (%s):\n%s", self.name, mode, out
            )
            return out

        if mode == "numeric":
            percentiles: Sequence[int] = kwargs.get("percentiles", (10, 25, 50, 75, 90))
            base = rnd.uniform(0, 100)
            lines = []
            for i, p in enumerate(percentiles):
                value = base + (i + 1) * rnd.uniform(5, 15)
                lines.append(f"Percentile {p}: {value:.1f}")
            out = "\n".join(lines)
            logger.info(
                "[Committee][Agent] %s RESPONSE (%s):\n%s", self.name, mode, out
            )
            return out

        return ""


@dataclass
class RealLLMAgent(LLMAgent):
    """Agent that calls real LLMs via forecasting_tools.GeneralLlm.

    The caller provides the ``name`` as a model identifier understood by
    forecasting_tools (e.g., "openrouter/gpt-4.1-mini", "openai/gpt-4o-mini").
    The ``generate`` method wraps the incoming prompt with strict formatting
    instructions for each mode ("scoping", "binary", "multiple_choice",
    "numeric") so outputs are compatible with the downstream parsers.
    """

    temperature: float | None = None
    timeout: int | None = 40
    allowed_tries: int = 2
    _llm: object | None = field(default=None, init=False, repr=False)

    def _is_gemini_model(self) -> bool:
        """Return True if this agent targets Gemini via google-genai.

        Currently supports: gemini-2.5-flash, gemini-2.5-pro
        """
        model = (self.name or "").strip().lower()
        return model in {"gemini-2.5-flash", "gemini-2.5-pro"}

    def _ensure_llm(self):
        # Gemini models are handled via google-genai, not GeneralLlm
        if self._is_gemini_model():
            return
        if self._llm is None:
            try:
                from forecasting_tools import GeneralLlm
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "[Committee][Agent] Could not import forecasting_tools.GeneralLlm: %s",
                    e,
                )
                raise
            self._llm = GeneralLlm(
                model=self.name,
                temperature=self.temperature,
                timeout=self.timeout,
                allowed_tries=self.allowed_tries,
            )

    def _gemini_invoke_blocking(self, final_prompt: str) -> str:
        """Synchronous call to Gemini using google-genai.

        Follows the pattern in `gemini_api.py`, but without the google_search tool.
        Reads the API key from GEMINI_API_KEY_1 (or GOOGLE_API_KEY fallback).
        """
        api_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY"),
        ]
        api_keys = [k for k in api_keys if k]
        if not api_keys:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY_1/2/3 or GOOGLE_API_KEY."
            )
        try:
            from google import genai
            from google.genai import types
        except Exception as e:  # noqa: BLE001
            logger.error("[Committee][Agent] google-genai import failed: %s", e)
            raise

        last_err: Exception | None = None
        for key in api_keys:
            try:
                client = genai.Client(api_key=key)
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=final_prompt)],
                    )
                ]
                config = types.GenerateContentConfig(response_mime_type="text/plain")

                full_response = ""
                try:
                    for chunk in client.models.generate_content_stream(
                        model=self.name,
                        contents=contents,
                        config=config,
                    ):
                        full_response += getattr(chunk, "text", "")
                except Exception:
                    # Fallback to non-streaming if streaming fails in this environment
                    result = client.models.generate_content(
                        model=self.name, contents=contents, config=config
                    )
                    # google-genai returns a response object with .text
                    full_response = getattr(result, "text", "")

                if full_response:
                    return full_response
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

        # If we exhausted keys without success, raise the last error
        if last_err is not None:
            raise last_err
        return ""

    @staticmethod
    def _wrap_prompt_for_mode(base_prompt: str, mode: str, **kwargs) -> str:
        """Attach strict output-format instructions on top of the base prompt.

        The base prompt already contains context like Research:/Question:/Peer analysis.
        We add concise instructions so the output fits our parsers in utils.py,
        while allowing a richer "initial" reasoning phase that does NOT return
        the final formatted line (to preserve parser behavior in the final phase).
        Use kwarg ``phase`` in {"initial", "final"}; default is "final".
        """
        phase = str(kwargs.get("phase", "final")).lower()

        if mode == "scoping":
            topic = kwargs.get("topic") or "the query"
            return (
                "You are generating web-search queries for forecasting research.\n"
                f"Topic: {topic}\n\n"
                "Return 4-8 concise queries, one per line, each starting with '- '.\n"
                "Cover a mix of types and tag each query at the start with one of: "
                "[base-rate], [definition], [contrary], [trend/market].\n"
                "- [base-rate]: reference classes, historical frequencies, survival/censoring.\n"
                "- [definition]: resolution criteria, synonyms, edge cases, timeline.\n"
                "- [contrary]: disconfirming evidence, critiques, failure modes.\n"
                "- [trend/market]: official statistics, expert/market signals, recent trend.\n"
                "Do not add any other text.\n\n"
                f"Context:\n{base_prompt}"
            )

        if mode == "binary":
            if phase == "initial":
                return (
                    "You are a professional forecaster preparing a forecast.\n"
                    "Write a brief, structured rationale only (no final probability line).\n"
                    "Include: (1) Prior/base rate (1-2 sentences), (2) 3-5 key drivers with direction,\n"
                    "(3) Evidence FOR and AGAINST (at least one contrary point), (4) Status quo baseline,\n"
                    "(5) Resolution criteria checks, (6) Proposed update description in small steps "
                    "(e.g., +/- a few percentage points or small log-odds changes).\n"
                    "Do NOT output any line that starts with 'Probability:' and do NOT end with '%'.\n\n"
                    f"Context:\n{base_prompt}"
                )
            # final
            return (
                "You are a professional forecaster. Based on the context, output ONLY one line:\n"
                "Probability: ZZ%\n"
                "Where ZZ is an integer or decimal from 0 to 100. Do not include any other text.\n\n"
                f"Context:\n{base_prompt}"
            )

        if mode == "multiple_choice":
            n = int(kwargs.get("n_options", 2))
            if phase == "initial":
                return (
                    "You are a professional forecaster preparing a multi-option forecast.\n"
                    "Write a brief, structured rationale only (no final probabilities line).\n"
                    "Include: (1) Baseline/reference shares per option, (2) 3-5 adjustment factors\n"
                    "with direction, (3) strongest/weakest options with 1-sentence reasons,\n"
                    "(4) at least one disconfirming consideration, (5) note any residual category risks.\n"
                    "Do NOT output any line that starts with 'Probabilities:' or percentages in brackets.\n\n"
                    f"Context:\n{base_prompt}"
                )
            # final
            return (
                "You are a professional forecaster. Based on the context, output ONLY one line:\n"
                "Probabilities: [p1%, p2%, ..., pN%]\n"
                f"Use exactly {n} values in order, percentages summing ~100. Do not include any other text.\n\n"
                f"Context:\n{base_prompt}"
            )

        if mode == "numeric":
            percentiles: Sequence[int] = kwargs.get("percentiles", (10, 25, 50, 75, 90))
            if phase == "initial":
                return (
                    "You are a professional forecaster preparing a numeric forecast.\n"
                    "Write a brief, structured rationale only (no percentile lines).\n"
                    "Include: (1) Units clarified and sanity-checked, (2) plausible min/max constraints,\n"
                    "(3) status-quo anchor and trend check, (4) distributional shape assumption\n"
                    "(e.g., symmetric vs. log-normal) and reason, (5) at least one disconfirming scenario.\n"
                    "Do NOT output lines beginning with 'Percentile'.\n\n"
                    f"Context:\n{base_prompt}"
                )
            # final
            parts = "\n".join(f"Percentile {p}: XX" for p in percentiles)
            return (
                "You are a professional forecaster. Based on the context, output ONLY the following lines, one per percentile, and nothing else:\n"
                f"{parts}\n\n"
                "Replace XX with numeric values (no units, no scientific notation).\n"
                "Always list the percentiles in increasing order.\n\n"
                f"Context:\n{base_prompt}"
            )

        # Fallback: return the base prompt (should not happen)
        return base_prompt

    async def generate(self, prompt: str, mode: str, **kwargs) -> str:  # type: ignore[override]
        self._ensure_llm()
        final_prompt = self._wrap_prompt_for_mode(prompt, mode, **kwargs)
        logger.info(
            "[Committee][Agent] %s REAL generate() | mode=%s | kwargs=%s",
            self.name,
            mode,
            kwargs,
        )
        logger.info(
            "[Committee][Agent] %s REAL PROMPT (%s):\n%s",
            self.name,
            mode,
            final_prompt,
        )

        tries = max(1, int(self.allowed_tries or 1))
        attempt = 0
        last_error: Exception | None = None
        while attempt < tries:
            try:
                if self._is_gemini_model():
                    # Run Gemini call without blocking the event loop
                    response = await asyncio.to_thread(
                        self._gemini_invoke_blocking, final_prompt
                    )
                else:
                    assert self._llm is not None
                    # mypy: GeneralLlm has .invoke
                    response = await getattr(self._llm, "invoke")(final_prompt)  # type: ignore[misc]
                logger.info(
                    "[Committee][Agent] %s REAL RESPONSE (%s):\n%s",
                    self.name,
                    mode,
                    response,
                )
                return response or ""
            except Exception as e:  # noqa: BLE001
                last_error = e
                attempt += 1
                logger.warning(
                    "[Committee][Agent] %s REAL call failed (mode=%s, attempt %s/%s): %s",
                    self.name,
                    mode,
                    attempt,
                    tries,
                    e,
                )
                if attempt < tries:
                    # Exponential backoff with jitter
                    delay = min(8.0, (2 ** (attempt - 1)))
                    await asyncio.sleep(delay)
                else:
                    break

        # Final fallback: deterministic generation to avoid bubbling exceptions
        logger.error(
            "[Committee][Agent] %s REAL call exhausted retries (mode=%s). Falling back to deterministic output. Last error: %s",
            self.name,
            mode,
            last_error,
        )
        fallback_agent = LLMAgent(name=self.name, weight=self.weight)
        return await fallback_agent.generate(prompt, mode, **kwargs)


def default_agent_committee() -> List[LLMAgent]:
    """Return the default set of agents.

    Behavior:
    - If ``COMMITTEE_LLM_MODELS`` is set, build REAL agents for those models.
    - Otherwise, build REAL agents for the default models:
      ``openrouter/claude-sonnet-4``, ``openrouter/gpt-5``,
      ``openrouter/claude-opus-4.1``, ``gemini-2.5-flash`` (``gemini-2.5-pro``
      is also supported if specified via env).

    Tuning env vars (applied in both cases):
    - ``COMMITTEE_LLM_TEMPERATURE`` (float or empty)
    - ``COMMITTEE_LLM_TIMEOUT`` (int, default 40)
    - ``COMMITTEE_LLM_TRIES`` (int, default 2)
    """
    models_csv = os.getenv("COMMITTEE_LLM_MODELS")

    # Parse tuning knobs
    def _parse_float(name: str) -> float | None:
        v = os.getenv(name)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _parse_int(name: str, default: int | None) -> int | None:
        v = os.getenv(name)
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    temperature = _parse_float("COMMITTEE_LLM_TEMPERATURE")
    timeout = _parse_int("COMMITTEE_LLM_TIMEOUT", 40)
    tries = _parse_int("COMMITTEE_LLM_TRIES", 2) or 2

    if models_csv:
        models = [m.strip() for m in models_csv.split(",") if m.strip()]
        committee: List[LLMAgent] = [
            RealLLMAgent(
                name=m,
                weight=1.0,
                temperature=temperature,
                timeout=timeout,
                allowed_tries=tries,
            )
            for m in models
        ]
        logger.info(
            "[Committee] Built REAL agent committee (env): %s",
            ", ".join(a.name for a in committee),
        )
        return committee

    # Default REAL models when env override is not set
    default_models = [
        "openrouter/anthropic/claude-sonnet-4",
        "openrouter/anthropic/claude-opus-4.1",
        "openrouter/anthropic/claude-opus-4.1",
        "openrouter/openai/gpt-5",
        "openrouter/openai/o3-pro",
        "openrouter/openai/o3-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-pro",
    ]
    committee = [
        RealLLMAgent(
            name=m,
            weight=1.0,
            temperature=temperature,
            timeout=timeout,
            allowed_tries=tries,
        )
        for m in default_models
    ]
    logger.info(
        "[Committee] Built REAL agent committee (default): %s",
        ", ".join(a.name for a in committee),
    )
    return committee
