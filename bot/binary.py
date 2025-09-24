"""Binary question forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import List
import logging

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    extract_probability_from_response_as_percentage_not_decimal,
    today_iso_utc,
)

logger = logging.getLogger(__name__)

async def get_binary_forecast(question: str, agents: List[LLMAgent]) -> float:
    """Forecast a binary question using the multi-step, multi-agent workflow."""
    logger.info(
        "[Committee][Binary] Starting forecast. Agents=%s | Question='%s'",
        len(agents),
        question,
    )
    today = today_iso_utc()
    # Phase 1: scoping prompts to produce queries
    historical_prompt = f"Date: {today} (UTC)\nHistorical perspective on: {question}"
    current_prompt = f"Date: {today} (UTC)\nCurrent events for: {question}"
    scope_agent = agents[0]
    # Log the two initial queries inferred from the question text
    logger.info(
        "[Committee][Binary] Initial scoping queries: '%s' | '%s'",
        historical_prompt,
        current_prompt,
    )
    logger.debug(
        "[Committee][Binary] Scoping prompts -> historical='%s' | current='%s'",
        historical_prompt,
        current_prompt,
    )
    hist_task = asyncio.create_task(
        scope_agent.generate(historical_prompt, mode="scoping", topic=question)
    )
    curr_task = asyncio.create_task(
        scope_agent.generate(current_prompt, mode="scoping", topic=question)
    )
    try:
        hist = await hist_task
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[Committee][Binary] Scoping (historical) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        hist = ""
    try:
        curr = await curr_task
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[Committee][Binary] Scoping (current) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        curr = ""
    scoped_queries = hist.splitlines() + curr.splitlines()
    logger.info(
        "[Committee][Binary] Scoped %s queries from scoping step",
        len(scoped_queries),
    )
    logger.debug("[Committee][Binary] Scoped queries: %s", scoped_queries)

    # Phase 2: Integrated research
    research_context = await integrated_research(scoped_queries)
    logger.info(
        "[Committee][Binary] Research context length: %s chars",
        len(research_context),
    )

    # Phase 3: Initial forecast from each agent
    prompt1 = f"Date: {today} (UTC)\nResearch:\n{research_context}\nQuestion: {question}"
    logger.debug("[Committee][Binary] Initial prompt:\n%s", prompt1)
    initial_tasks = [
        asyncio.create_task(agent.generate(prompt1, mode="binary", phase="initial"))
        for agent in agents
    ]
    initial_results = await asyncio.gather(*initial_tasks, return_exceptions=True)
    active_agents: List[LLMAgent] = []
    initial_responses: List[str] = []
    for agent, result in zip(agents, initial_results):
        if isinstance(result, Exception):
            logger.warning(
                "[Committee][Binary] Initial generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        active_agents.append(agent)
        initial_responses.append(result)
    logger.info(
        "[Committee][Binary] Initial responses collected: %s (failed=%s)",
        len(initial_responses),
        len(agents) - len(initial_responses),
    )
    logger.debug("[Committee][Binary] Initial responses: %s", initial_responses)

    if not active_agents:
        logger.error(
            "[Committee][Binary] All agents failed during initial generation. Returning neutral 0.5 forecast."
        )
        return 0.5

    # Phase 3.5: create peer review map by rotating analyses
    context_map = initial_responses[1:] + initial_responses[:1]
    logger.debug("[Committee][Binary] Peer context map: %s", context_map)

    # Phase 4: Final synthesis prompts
    final_tasks = []
    for agent, peer_context in zip(active_agents, context_map):
        final_prompt = (
            f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer_context}\nQuestion: {question}"
        )
        final_tasks.append(
            asyncio.create_task(
                agent.generate(final_prompt, mode="binary", phase="final")
            )
        )
    final_results = await asyncio.gather(*final_tasks, return_exceptions=True)
    final_agents: List[LLMAgent] = []
    final_responses: List[str] = []
    for agent, result in zip(active_agents, final_results):
        if isinstance(result, Exception):
            logger.warning(
                "[Committee][Binary] Final generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        final_agents.append(agent)
        final_responses.append(result)
    logger.info(
        "[Committee][Binary] Final responses collected: %s (failed=%s)",
        len(final_responses),
        len(active_agents) - len(final_responses),
    )
    logger.debug("[Committee][Binary] Final responses: %s", final_responses)

    if not final_agents:
        logger.error(
            "[Committee][Binary] All agents failed during final generation. Returning neutral 0.5 forecast."
        )
        return 0.5

    # Extraction and aggregation
    probs = [
        extract_probability_from_response_as_percentage_not_decimal(r)
        for r in final_responses
    ]
    logger.info("[Committee][Binary] Extracted probs (decimals): %s", probs)
    weights = np.array([a.weight for a in final_agents])
    logger.debug("[Committee][Binary] Agent weights: %s", weights.tolist())
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        logger.error(
            "[Committee][Binary] Non-positive weight sum after filtering. Returning neutral 0.5 forecast."
        )
        return 0.5
    aggregate = float(np.sum(weights * np.array(probs)) / weight_sum)
    clipped = max(0.001, min(0.999, aggregate))
    logger.info(
        "[Committee][Binary] Aggregated (weighted) prob=%.4f | Clipped=%.4f",
        aggregate,
        clipped,
    )
    return clipped
