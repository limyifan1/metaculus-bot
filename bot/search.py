"""Simplified research utilities.

The real system would use AskNews and Gemini search APIs.  Here we simply
return canned strings describing the queries so that downstream prompts have
some context to work with.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List


async def _asknews(query: str) -> str:
    await asyncio.sleep(0)  # allow context switch
    return f"AskNews summary for '{query}'"


async def _gemini_search(query: str) -> str:
    await asyncio.sleep(0)
    return f"Gemini search result for '{query}'"


async def integrated_research(queries: Iterable[str]) -> str:
    """Run AskNews and Gemini search over the provided queries."""
    tasks: List[asyncio.Task[str]] = []
    for q in queries:
        tasks.append(asyncio.create_task(_asknews(q)))
        tasks.append(asyncio.create_task(_gemini_search(q)))
    results = await asyncio.gather(*tasks)
    return "\n".join(results)
