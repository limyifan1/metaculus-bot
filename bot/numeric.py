"""Numeric question forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    PercentileForecast,
    extract_percentiles_from_response,
)


async def get_numeric_forecast(
    question: str, percentiles: Sequence[int], agents: List[LLMAgent]
) -> List[float]:
    """Return an aggregated 201 point CDF."""
    scope_agent = agents[0]
    hist = await scope_agent.generate(
        f"Historical perspective on: {question}", mode="scoping", topic=question
    )
    curr = await scope_agent.generate(
        f"Current events for: {question}", mode="scoping", topic=question
    )
    queries = hist.splitlines() + curr.splitlines()
    research_context = await integrated_research(queries)

    prompt1 = f"Research:\n{research_context}\nQuestion: {question}"
    initial = await asyncio.gather(
        *[
            agent.generate(prompt1, mode="numeric", percentiles=percentiles)
            for agent in agents
        ]
    )
    context_map = initial[1:] + initial[:1]
    final = await asyncio.gather(
        *[
            agent.generate(
                f"Research:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}",
                mode="numeric",
                percentiles=percentiles,
            )
            for agent, peer in zip(agents, context_map)
        ]
    )

    cdfs: List[List[float]] = []
    for resp in final:
        percentile_values: Dict[int, float] = extract_percentiles_from_response(resp)
        forecast = PercentileForecast(percentile_values)
        cdfs.append(forecast.generate_continuous_cdf())

    weights = np.array([a.weight for a in agents])[:, None]
    cdf_array = np.sum(weights * np.array(cdfs), axis=0) / np.sum(weights)
    return cdf_array.tolist()
