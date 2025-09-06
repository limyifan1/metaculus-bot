"""Binary question forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    extract_probability_from_response_as_percentage_not_decimal,
)


async def get_binary_forecast(question: str, agents: List[LLMAgent]) -> float:
    """Forecast a binary question using the multi-step, multi-agent workflow."""
    # Phase 1: scoping prompts to produce queries
    historical_prompt = f"Historical perspective on: {question}"
    current_prompt = f"Current events for: {question}"
    scope_agent = agents[0]
    hist_task = asyncio.create_task(
        scope_agent.generate(historical_prompt, mode="scoping", topic=question)
    )
    curr_task = asyncio.create_task(
        scope_agent.generate(current_prompt, mode="scoping", topic=question)
    )
    scoped_queries = (await hist_task).splitlines() + (await curr_task).splitlines()

    # Phase 2: Integrated research
    research_context = await integrated_research(scoped_queries)

    # Phase 3: Initial forecast from each agent
    prompt1 = f"Research:\n{research_context}\nQuestion: {question}"
    initial_tasks = [
        asyncio.create_task(agent.generate(prompt1, mode="binary"))
        for agent in agents
    ]
    initial_responses = await asyncio.gather(*initial_tasks)

    # Phase 3.5: create peer review map by rotating analyses
    context_map = initial_responses[1:] + initial_responses[:1]

    # Phase 4: Final synthesis prompts
    final_tasks = []
    for agent, peer_context in zip(agents, context_map):
        final_prompt = (
            f"Research:\n{research_context}\nPeer analysis:\n{peer_context}\nQuestion: {question}"
        )
        final_tasks.append(asyncio.create_task(agent.generate(final_prompt, mode="binary")))
    final_responses = await asyncio.gather(*final_tasks)

    # Extraction and aggregation
    probs = [
        extract_probability_from_response_as_percentage_not_decimal(r)
        for r in final_responses
    ]
    weights = np.array([a.weight for a in agents])
    aggregate = float(np.sum(weights * probs) / np.sum(weights))
    return max(0.001, min(0.999, aggregate))
