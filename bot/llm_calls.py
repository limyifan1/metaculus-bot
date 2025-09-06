"""LLM agent abstractions used by the forecasting system.

The real system would interface with multiple model families.  For the
purposes of this repository the agents produce deterministic pseudo-random
outputs so the rest of the pipeline can be exercised without external
API calls.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class LLMAgent:
    """Simple deterministic agent used for simulation.

    The ``generate`` method returns text matching the expected structure for
    the different forecasting prompts.  A hash of the prompt and agent name is
    used to seed a local random generator so results are stable across runs.
    """

    name: str
    weight: float = 1.0

    async def generate(self, prompt: str, mode: str, **kwargs) -> str:
        seed_input = (self.name + mode + prompt).encode()
        seed = int(hashlib.md5(seed_input).hexdigest()[:8], 16)
        rnd = random.Random(seed)

        if mode == "scoping":
            topic = kwargs.get("topic", "query")
            return f"- {topic} history\n- {topic} recent developments"

        if mode == "binary":
            prob = rnd.randint(1, 99)
            return f"Probability: {prob}%"

        if mode == "multiple_choice":
            n = kwargs.get("n_options", 2)
            raw = [rnd.random() for _ in range(n)]
            total = sum(raw)
            percentages = [round(x / total * 100, 1) for x in raw]
            joined = ", ".join(f"{p}%" for p in percentages)
            return f"Probabilities: [{joined}]"

        if mode == "numeric":
            percentiles: Sequence[int] = kwargs.get(
                "percentiles", (10, 25, 50, 75, 90)
            )
            base = rnd.uniform(0, 100)
            lines = []
            for i, p in enumerate(percentiles):
                value = base + (i + 1) * rnd.uniform(5, 15)
                lines.append(f"Percentile {p}: {value:.1f}")
            return "\n".join(lines)

        return ""


def default_agent_committee() -> List[LLMAgent]:
    """Return a default set of three equal-weight agents."""
    return [
        LLMAgent("openrouter/claude-sonnet-4"),
        LLMAgent("openrouter/gpt-5"),
        LLMAgent("gemini-2.5-pro"),
    ]
