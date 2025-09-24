"""Multiple-choice forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import List, Sequence
import logging

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    extract_option_probabilities_from_response,
    normalize_probabilities,
    today_iso_utc,
)

logger = logging.getLogger(__name__)

async def get_multiple_choice_forecast(
    question: str, options: Sequence[str], agents: List[LLMAgent]
) -> List[float]:
    """Return aggregated probabilities for each option."""
    logger.info(
        "[Committee][MC] Starting forecast. Agents=%s | Options=%s | Question='%s'",
        len(agents),
        list(options),
        question,
    )
    if not options:
        logger.error(
            "[Committee][MC] No options supplied. Returning empty probability list."
        )
        return []

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
            "[Committee][MC] Scoping (historical) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        hist = ""
    try:
        curr = await curr_task
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[Committee][MC] Scoping (current) failed for agent %s: %s",
            scope_agent.name,
            e,
        )
        curr = ""
    queries = hist.splitlines() + curr.splitlines()
    logger.info("[Committee][MC] Scoped %s queries", len(queries))
    logger.debug("[Committee][MC] Scoped queries: %s", queries)
    research_context = await integrated_research(queries)
    logger.info("[Committee][MC] Research context length: %s chars", len(research_context))

    prompt1 = (
        f"Date: {today} (UTC)\nResearch:\n{research_context}\nQuestion: {question}\nOptions: {options}"
    )
    logger.debug("[Committee][MC] Initial prompt:\n%s", prompt1)
    initial_results = await asyncio.gather(
        *[
            agent.generate(
                prompt1,
                mode="multiple_choice",
                n_options=len(options),
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
                "[Committee][MC] Initial generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        active_agents.append(agent)
        initial_responses.append(result)
    logger.info(
        "[Committee][MC] Initial responses collected: %s (failed=%s)",
        len(initial_responses),
        len(agents) - len(initial_responses),
    )
    logger.debug("[Committee][MC] Initial responses: %s", initial_responses)

    if not active_agents:
        logger.error(
            "[Committee][MC] All agents failed during initial generation. Returning uniform probabilities."
        )
        return [1.0 / len(options)] * len(options)

    context_map = initial_responses[1:] + initial_responses[:1]
    logger.debug("[Committee][MC] Peer context map: %s", context_map)
    final_results = await asyncio.gather(
        *[
            agent.generate(
                f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}\nOptions: {options}",
                mode="multiple_choice",
                n_options=len(options),
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
                "[Committee][MC] Final generation failed for agent %s: %s",
                agent.name,
                result,
            )
            continue
        final_agents.append(agent)
        final_responses.append(result)
    logger.info(
        "[Committee][MC] Final responses collected: %s (failed=%s)",
        len(final_responses),
        len(active_agents) - len(final_responses),
    )
    logger.debug("[Committee][MC] Final responses: %s", final_responses)

    prob_lists = []
    filtered_agents: List[LLMAgent] = []
    for agent, response in zip(final_agents, final_responses):
        extracted = extract_option_probabilities_from_response(response)
        if len(extracted) != len(options):
            logger.warning(
                "[Committee][MC] Skipping agent %s due to invalid probability list (len=%s, expected=%s)",
                agent.name,
                len(extracted),
                len(options),
            )
            continue
        normalized = normalize_probabilities(extracted)
        if len(normalized) != len(options):
            logger.warning(
                "[Committee][MC] Skipping agent %s due to normalization mismatch",
                agent.name,
            )
            continue
        prob_lists.append(normalized)
        filtered_agents.append(agent)

    if not prob_lists:
        logger.error(
            "[Committee][MC] No valid final responses remaining. Returning uniform probabilities."
        )
        return [1.0 / len(options)] * len(options)

    weights = np.array([a.weight for a in filtered_agents])[:, None]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        logger.error(
            "[Committee][MC] Non-positive weight sum after filtering. Returning uniform probabilities."
        )
        return [1.0 / len(options)] * len(options)

    probs = np.sum(weights * np.array(prob_lists), axis=0) / weight_sum
    logger.info(
        "[Committee][MC] Aggregated probabilities per option: %s",
        {name: round(float(p), 4) for name, p in zip(options, probs.tolist())},
    )
    return probs.tolist()
