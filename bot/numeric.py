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
    try:
        hist = await hist_task
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[Committee][Numeric] Scoping (historical) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        hist = ""
    try:
        curr = await curr_task
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[Committee][Numeric] Scoping (current) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        curr = ""
    queries = hist.splitlines() + curr.splitlines()
    logger.info("[Committee][Numeric] Scoped %s queries", len(queries))
    logger.debug("[Committee][Numeric] Scoped queries: %s", queries)
    research_context = await integrated_research(queries)
    logger.info(
        "[Committee][Numeric] Research context length: %s chars", len(research_context)
    )

    prompt1 = f"Date: {today} (UTC)\nResearch:\n{research_context}\nQuestion: {question}"
    logger.debug("[Committee][Numeric] Initial prompt:\n%s", prompt1)
    initial_results = await asyncio.gather(
        *[
            agent.generate(
                prompt1,
                mode="numeric",
                percentiles=percentiles,
                phase="initial",
            )
            for agent in agents
        ],
        return_exceptions=True,
    )
    active_agents: List[LLMAgent] = []
    initial_responses: List[str] = []
    for agent, result in zip(agents, initial_results):
        if isinstance(result, Exception):
            logger.warning(
                "[Committee][Numeric] Initial generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        active_agents.append(agent)
        initial_responses.append(result)
    logger.info(
        "[Committee][Numeric] Initial responses collected: %s (failed=%s)",
        len(initial_responses),
        len(agents) - len(initial_responses),
    )
    logger.debug("[Committee][Numeric] Initial responses: %s", initial_responses)

    if not active_agents:
        logger.error(
            "[Committee][Numeric] All agents failed during initial generation. Returning baseline CDF."
        )
        return np.linspace(0.0, 1.0, 201).tolist()

    context_map = initial_responses[1:] + initial_responses[:1]
    logger.debug("[Committee][Numeric] Peer context map: %s", context_map)
    final_results = await asyncio.gather(
        *[
            agent.generate(
                f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}",
                mode="numeric",
                percentiles=percentiles,
                phase="final",
            )
            for agent, peer in zip(active_agents, context_map)
        ],
        return_exceptions=True,
    )
    final_agents: List[LLMAgent] = []
    final_responses: List[str] = []
    for agent, result in zip(active_agents, final_results):
        if isinstance(result, Exception):
            logger.warning(
                "[Committee][Numeric] Final generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        final_agents.append(agent)
        final_responses.append(result)
    logger.info(
        "[Committee][Numeric] Final responses collected: %s (failed=%s)",
        len(final_responses),
        len(active_agents) - len(final_responses),
    )
    logger.debug("[Committee][Numeric] Final responses: %s", final_responses)

    cdfs: List[List[float]] = []
    filtered_agents: List[LLMAgent] = []
    for agent, resp in zip(final_agents, final_responses):
        percentile_values: Dict[int, float] = extract_percentiles_from_response(resp)
        logger.debug(
            "[Committee][Numeric] Extracted percentiles from response: %s",
            percentile_values,
        )
        if not percentile_values:
            logger.warning(
                "[Committee][Numeric] Skipping agent %s due to missing percentiles",
                agent.name,
            )
            continue
        try:
            forecast = PercentileForecast(percentile_values)
            cdfs.append(forecast.generate_continuous_cdf())
            filtered_agents.append(agent)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[Committee][Numeric] Skipping agent %s due to CDF generation error: %s",
                agent.name,
                e,
            )

    if not cdfs:
        logger.error(
            "[Committee][Numeric] No valid final responses remaining. Returning baseline CDF."
        )
        return np.linspace(0.0, 1.0, 201).tolist()

    weights = np.array([a.weight for a in filtered_agents])[:, None]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        logger.error(
            "[Committee][Numeric] Non-positive weight sum after filtering. Returning baseline CDF."
        )
        return np.linspace(0.0, 1.0, 201).tolist()

    cdf_array = np.sum(weights * np.array(cdfs), axis=0) / weight_sum
    logger.info(
        "[Committee][Numeric] Aggregated CDF size: %s (first/last: %.3f, %.3f)",
        len(cdf_array),
        float(cdf_array[0]) if len(cdf_array) else float("nan"),
        float(cdf_array[-1]) if len(cdf_array) else float("nan"),
    )
    return cdf_array.tolist()
