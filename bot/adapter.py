"""Adapter to use the local `bot/` multi-agent workflow
within the `forecasting_tools` ForecastBot interface.

This lets you run the committee-based forecasts while
still benefiting from Metaculus question retrieval,
aggregation, and posting provided by forecasting_tools.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)

from .llm_calls import LLMAgent, default_agent_committee
from .binary import get_binary_forecast
from .multiple_choice import get_multiple_choice_forecast
from .numeric import get_numeric_forecast
from .search import integrated_research

logger = logging.getLogger(__name__)


class CommitteeForecastBot(ForecastBot):
    """ForecastBot implementation backed by the local committee workflow.

    This replaces LLM prompting with the deterministic, multi-agent
    implementation under `bot/`, but preserves the higher-level orchestration
    (question fetching, aggregation, reporting, posting to Metaculus).
    """

    def __init__(
        self,
        *,
        agents: Sequence[LLMAgent] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._agents: List[LLMAgent] = (
            list(agents) if agents is not None else default_agent_committee()
        )
        logger.info(
            "CommitteeForecastBot initialized with %s agents: %s",
            len(self._agents),
            ", ".join(a.name for a in self._agents),
        )

    async def run_research(self, question: MetaculusQuestion) -> str:  # type: ignore[override]
        # Minimal, local research using stubbed utilities; formatted as markdown.
        queries = [
            f"Historical perspective on: {question.question_text}",
            f"Current events for: {question.question_text}",
        ]
        logger.info(
            "[Committee] Running integrated research on %s queries for URL %s",
            len(queries),
            getattr(question, "page_url", "<unknown>"),
        )
        # Explicitly log the two initial queries inferred from the URL/context
        if queries:
            logger.info("[Committee] Initial queries inferred from URL/context:")
            for q in queries:
                logger.info("[Committee]   - %s", q)
        logger.debug("[Committee] Research queries: %s", queries)
        research = await integrated_research(queries)
        logger.info(
            "[Committee] Integrated research collected (%s chars)", len(research)
        )
        header = "Here are the scoped queries used for integrated research:\n" + "\n".join(
            f"- {q}" for q in queries
        )
        return f"{header}\n\n{research}"

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:  # type: ignore[override]
        # Use parent summarization, but log the full summary for traceability
        summary = await super().summarize_research(question, research)
        try:
            logger.info(
                "[Committee] Research summary (%s chars) for %s:\n%s",
                len(summary),
                getattr(question, "page_url", "<unknown>"),
                summary,
            )
        except Exception:
            # Avoid crashing if summary is not a string or encoding issues occur
            logger.info("[Committee] Research summary logged (content omitted due to formatting error)")
        return summary

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:  # type: ignore[override]
        logger.info(
            "[Committee] Forecasting binary question (agents=%s): %s",
            len(self._agents),
            question.question_text,
        )
        prob = await get_binary_forecast(question.question_text, self._agents)
        logger.info("[Committee] Aggregated binary probability (decimal): %.4f", prob)
        reasoning = (
            f"Committee of {len(self._agents)} agents produced an aggregated probability: "
            f"{round(prob * 100, 2)}%."
        )
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:  # type: ignore[override]
        logger.info(
            "[Committee] Forecasting multiple-choice (agents=%s) options=%s",
            len(self._agents),
            question.options,
        )
        probs = await get_multiple_choice_forecast(
            question.question_text, question.options, self._agents
        )
        predicted = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=float(p))
                for name, p in zip(question.options, probs)
            ]
        )
        logger.info(
            "[Committee] Aggregated MC probabilities: %s",
            {name: round(float(p), 4) for name, p in zip(question.options, probs)},
        )
        reasoning_lines = [
            f"- {name}: {round(p * 100, 1)}%"
            for name, p in zip(question.options, probs)
        ]
        reasoning = (
            f"Committee of {len(self._agents)} agents aggregated option probabilities:\n"
            + "\n".join(reasoning_lines)
        )
        return ReasonedPrediction(prediction_value=predicted, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:  # type: ignore[override]
        # Ask the local numeric workflow to generate a full quantile path (201 points)
        # based on a small set of guiding percentiles.
        guiding_percentiles = (10, 25, 50, 75, 90)
        logger.info(
            "[Committee] Forecasting numeric (agents=%s) with guiding percentiles %s",
            len(self._agents),
            guiding_percentiles,
        )
        quantile_values = await get_numeric_forecast(
            question.question_text, guiding_percentiles, self._agents
        )
        # Map the quantile path (values at evenly spaced probabilities) to declared percentiles
        # used by forecasting_tools. Grid is 0..1 with 201 points => index = int(round(p * 200)).
        sample_ps = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
        # Percentile objects expect integer percent values (e.g., 10, 20, ..., 90),
        # while we index into the 201-point quantile path using decimal probabilities.
        declared = [
            Percentile(
                percentile=int(round(p * 100)),
                value=float(quantile_values[int(round(p * 200))]),
            )
            for p in sample_ps
        ]
        distribution = NumericDistribution.from_question(declared, question)
        logger.info(
            "[Committee] Aggregated numeric distribution samples: %s",
            {int(p * 100): float(quantile_values[int(round(p * 200))]) for p in sample_ps},
        )
        reasoning = (
            "Committee aggregated numeric distribution (representative percentiles): "
            + ", ".join(
                f"P{int(p*100)}={float(quantile_values[int(round(p*200))]):.2f}"
                for p in sample_ps
            )
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)
