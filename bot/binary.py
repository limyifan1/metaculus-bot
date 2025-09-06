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
    hist = await hist_task
    curr = await curr_task
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
    initial_responses = await asyncio.gather(*initial_tasks)
    logger.info(
        "[Committee][Binary] Initial responses collected: %s",
        len(initial_responses),
    )
    logger.debug("[Committee][Binary] Initial responses: %s", initial_responses)

    # Phase 3.5: create peer review map by rotating analyses
    context_map = initial_responses[1:] + initial_responses[:1]
    logger.debug("[Committee][Binary] Peer context map: %s", context_map)

    # Phase 4: Final synthesis prompts
    final_tasks = []
    for agent, peer_context in zip(agents, context_map):
        final_prompt = (
            f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer_context}\nQuestion: {question}"
        )
        final_tasks.append(
            asyncio.create_task(
                agent.generate(final_prompt, mode="binary", phase="final")
            )
        )
    final_responses = await asyncio.gather(*final_tasks)
    logger.info(
        "[Committee][Binary] Final responses collected: %s",
        len(final_responses),
    )
    logger.debug("[Committee][Binary] Final responses: %s", final_responses)

    # Extraction and aggregation
    probs = [
        extract_probability_from_response_as_percentage_not_decimal(r)
        for r in final_responses
    ]
    logger.info("[Committee][Binary] Extracted probs (decimals): %s", probs)
    weights = np.array([a.weight for a in agents])
    logger.debug("[Committee][Binary] Agent weights: %s", weights.tolist())
    aggregate = float(np.sum(weights * probs) / np.sum(weights))
    clipped = max(0.001, min(0.999, aggregate))
    logger.info(
        "[Committee][Binary] Aggregated (weighted) prob=%.4f | Clipped=%.4f",
        aggregate,
        clipped,
    )
    return clipped
