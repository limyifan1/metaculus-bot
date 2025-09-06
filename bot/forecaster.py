"""Per-question-type orchestration for forecasts."""

from __future__ import annotations

from typing import Sequence

from .binary import get_binary_forecast
from .llm_calls import LLMAgent, default_agent_committee
from .multiple_choice import get_multiple_choice_forecast
from .numeric import get_numeric_forecast


async def forecast_question(question: dict, agents: Sequence[LLMAgent] | None = None):
    """Route a question dictionary to the appropriate workflow.

    Parameters
    ----------
    question: dict
        A mapping containing ``type`` and other fields depending on the type.
    agents: Sequence[LLMAgent] | None
        Optionally override the default agent committee.
    """
    agents = list(agents) if agents is not None else default_agent_committee()
    qtype = question["type"]
    if qtype == "binary":
        return await get_binary_forecast(question["text"], agents)
    if qtype == "multiple_choice":
        return await get_multiple_choice_forecast(
            question["text"], question["options"], agents
        )
    if qtype == "numeric":
        return await get_numeric_forecast(
            question["text"], question.get("percentiles", (10, 25, 50, 75, 90)), agents
        )
    raise ValueError(f"Unknown question type: {qtype}")
