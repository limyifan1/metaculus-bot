import asyncio
import logging
from typing import Literal

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)

from .forecast import (
    forecast_binary,
    forecast_multiple_choice,
    forecast_numeric,
)
from .research import run_research as _run_research_impl

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):
    """A clearer, modularized version of the Fall 2025 template bot.

    This class delegates research and forecasting logic to separate modules
    for readability and easier modification.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:  # type: ignore[override]
        return await _run_research_impl(self, question)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:  # type: ignore[override]
        return await forecast_binary(self, question, research)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:  # type: ignore[override]
        return await forecast_multiple_choice(self, question, research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:  # type: ignore[override]
        return await forecast_numeric(self, question, research)

