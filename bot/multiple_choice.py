"""Multiple-choice forecasting workflow."""

from __future__ import annotations

import asyncio
from typing import List, Sequence

import numpy as np

from .llm_calls import LLMAgent
from .search import integrated_research
from .utils import (
    extract_option_probabilities_from_response,
    normalize_probabilities,
)


async def get_multiple_choice_forecast(
    question: str, options: Sequence[str], agents: List[LLMAgent]
) -> List[float]:
    """Return aggregated probabilities for each option."""
    scope_agent = agents[0]
    hist = await scope_agent.generate(
        f"Historical perspective on: {question}", mode="scoping", topic=question
    )
    curr = await scope_agent.generate(
        f"Current events for: {question}", mode="scoping", topic=question
    )
    queries = hist.splitlines() + curr.splitlines()
    research_context = await integrated_research(queries)

    prompt1 = f"Research:\n{research_context}\nQuestion: {question}\nOptions: {options}"
    initial = await asyncio.gather(
        *[
            agent.generate(
                prompt1, mode="multiple_choice", n_options=len(options)
            )
            for agent in agents
        ]
    )
    context_map = initial[1:] + initial[:1]
    final = await asyncio.gather(
        *[
            agent.generate(
                f"Research:\n{research_context}\nPeer analysis:\n{peer}\nQuestion: {question}\nOptions: {options}",
                mode="multiple_choice",
                n_options=len(options),
            )
            for agent, peer in zip(agents, context_map)
        ]
    )
    prob_lists = [
        normalize_probabilities(
            extract_option_probabilities_from_response(resp)
        )
        for resp in final
    ]
    weights = np.array([a.weight for a in agents])[:, None]
    probs = np.sum(weights * np.array(prob_lists), axis=0) / np.sum(weights)
    return probs.tolist()
