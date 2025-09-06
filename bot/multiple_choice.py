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
    logger.info("[Committee][MC] Scoped %s queries", len(queries))
    logger.debug("[Committee][MC] Scoped queries: %s", queries)
    research_context = await integrated_research(queries)
    logger.info("[Committee][MC] Research context length: %s chars", len(research_context))

    prompt1 = (
        f"Date: {today} (UTC)\nResearch:\n{research_context}\nQuestion: {question}\nOptions: {options}"
    )
    logger.debug("[Committee][MC] Initial prompt:\n%s", prompt1)
    initial = await asyncio.gather(
        *[
            agent.generate(
                prompt1,
                mode="multiple_choice",
                n_options=len(options),
                phase="initial",
            )
            for agent in agents
        ]
    )
    logger.info("[Committee][MC] Initial responses collected: %s", len(initial))
    logger.debug("[Committee][MC] Initial responses: %s", initial)
    context_map = initial[1:] + initial[:1]
    final = await asyncio.gather(
        *[
            agent.generate(
                f"Date: {today} (UTC)\nResearch:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}\nOptions: {options}",
                mode="multiple_choice",
                n_options=len(options),
                phase="final",
            )
            for agent, peer in zip(agents, context_map)
        ]
    )
    logger.info("[Committee][MC] Final responses collected: %s", len(final))
    logger.debug("[Committee][MC] Final responses: %s", final)
    prob_lists = [
        normalize_probabilities(
            extract_option_probabilities_from_response(resp)
        )
        for resp in final
    ]
    logger.info("[Committee][MC] Extracted per-agent probabilities: %s", prob_lists)
    weights = np.array([a.weight for a in agents])[:, None]
    probs = np.sum(weights * np.array(prob_lists), axis=0) / np.sum(weights)
    logger.info(
        "[Committee][MC] Aggregated probabilities per option: %s",
        {name: round(float(p), 4) for name, p in zip(options, probs.tolist())},
    )
    return probs.tolist()
