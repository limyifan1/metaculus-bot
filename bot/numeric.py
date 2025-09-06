"""Numeric question forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence
import logging

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    PercentileForecast,
    extract_percentiles_from_response,
    today_iso_utc,
)

logger = logging.getLogger(__name__)

async def get_numeric_forecast(
    question: str, percentiles: Sequence[int], agents: List[LLMAgent]
) -> List[float]:
    """Return an aggregated 201 point CDF."""
    logger.info(
        "[Committee][Numeric] Starting forecast. Agents=%s | Percentiles=%s | Question='%s'",
        len(agents),
        list(percentiles),
        question,
    )
    today = today_iso_utc()
    scope_agent = agents[0]
    # Run scoping prompts concurrently (same content/order preserved)
    hist_task = asyncio.create_task(
        scope_agent.generate(
            f"Date: {today} (UTC)\nHistorical perspective on: {question}",
            mode="scoping",
            topic=question,
        )
    )
    curr_task = asyncio.create_task(
        scope_agent.generate(
            f"Date: {today} (UTC)\nCurrent events for: {question}",
            mode="scoping",
            topic=question,
        )
    )
    hist = await hist_task
    curr = await curr_task
    queries = hist.splitlines() + curr.splitlines()
    logger.info("[Committee][Numeric] Scoped %s queries", len(queries))
    logger.debug("[Committee][Numeric] Scoped queries: %s", queries)
    research_context = await integrated_research(queries)
    logger.info(
        "[Committee][Numeric] Research context length: %s chars", len(research_context)
    )

    prompt1 = f"Date: {today} (UTC)\nResearch:\n{research_context}\nQuestion: {question}"
    logger.debug("[Committee][Numeric] Initial prompt:\n%s", prompt1)
    initial = await asyncio.gather(
        *[
            agent.generate(
                prompt1,
                mode="numeric",
                percentiles=percentiles,
                phase="initial",
            )
            for agent in agents
        ]
    )
    logger.info("[Committee][Numeric] Initial responses collected: %s", len(initial))
    logger.debug("[Committee][Numeric] Initial responses: %s", initial)
    context_map = initial[1:] + initial[:1]
    final = await asyncio.gather(
        *[
            agent.generate(
                f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}",
                mode="numeric",
                percentiles=percentiles,
                phase="final",
            )
            for agent, peer in zip(agents, context_map)
        ]
    )
    logger.info("[Committee][Numeric] Final responses collected: %s", len(final))
    logger.debug("[Committee][Numeric] Final responses: %s", final)

    cdfs: List[List[float]] = []
    for resp in final:
        percentile_values: Dict[int, float] = extract_percentiles_from_response(resp)
        logger.debug(
            "[Committee][Numeric] Extracted percentiles from response: %s",
            percentile_values,
        )
        forecast = PercentileForecast(percentile_values)
        cdfs.append(forecast.generate_continuous_cdf())

    weights = np.array([a.weight for a in agents])[:, None]
    cdf_array = np.sum(weights * np.array(cdfs), axis=0) / np.sum(weights)
    logger.info(
        "[Committee][Numeric] Aggregated CDF size: %s (first/last: %.3f, %.3f)",
        len(cdf_array),
        float(cdf_array[0]) if len(cdf_array) else float("nan"),
        float(cdf_array[-1]) if len(cdf_array) else float("nan"),
    )
    return cdf_array.tolist()
