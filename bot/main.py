"""Entry point for running the multi-agent forecasting system.

This module provides a tiny harness that simulates fetching open questions
and runs the appropriate forecasting workflow for each.
"""

from __future__ import annotations

import asyncio
from typing import List

from .forecaster import forecast_question

# In the real system this would call the Metaculus API.  Here we simply return
# a small set of example questions of various types.


def get_open_question_ids_from_tournament(_tournament_id: int) -> List[dict]:
    return [
        {"id": 1, "type": "binary", "text": "Will it rain tomorrow?"},
        {
            "id": 2,
            "type": "multiple_choice",
            "text": "Who will win the match?",
            "options": ["Team A", "Team B", "Draw"],
        },
        {
            "id": 3,
            "type": "numeric",
            "text": "How many users will sign up next month?",
            "percentiles": (10, 25, 50, 75, 90),
        },
    ]


async def forecast_individual_question(question: dict) -> None:
    result = await forecast_question(question)
    print(f"Question {question['id']} ({question['type']}): {result}")


async def main() -> None:
    questions = get_open_question_ids_from_tournament(0)
    tasks = [forecast_individual_question(q) for q in questions]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
