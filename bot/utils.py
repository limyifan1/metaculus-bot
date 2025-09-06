"""Utility functions for the multi-agent forecasting system."""

from __future__ import annotations

import re
from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


def today_iso_utc() -> str:
    """Return today's date in ISO format using UTC (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def extract_probability_from_response_as_percentage_not_decimal(text: str) -> float:
    """Extract a probability expressed as a percentage and return a decimal."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", text)
    if not match:
        logger.warning(
            "[Committee][Utils] No percentage found in binary response. Defaulting to 0.5. Text=%s",
            text,
        )
        return 0.5
    return float(match.group(1)) / 100.0


def extract_option_probabilities_from_response(text: str) -> List[float]:
    """Parse a list of probabilities from a string like ``[30%, 40%, 30%]``."""
    match = re.search(r"\[([^\]]+)\]", text)
    if not match:
        logger.warning(
            "[Committee][Utils] Could not find list in multiple choice response. Text=%s",
            text,
        )
        return []
    parts = match.group(1).split(",")
    probs = []
    for part in parts:
        number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", part)
        if number_match:
            probs.append(float(number_match.group(1)) / 100.0)
    return probs


def normalize_probabilities(probs: Iterable[float]) -> List[float]:
    """Normalize a list of probabilities so they sum to one."""
    arr = np.array(list(probs), dtype=float)
    total = arr.sum()
    if total <= 0:
        logger.warning(
            "[Committee][Utils] Non-positive probability sum detected. Uniform fallback applied. Values=%s",
            arr.tolist(),
        )
        return [1.0 / len(arr)] * len(arr)
    return list(arr / total)


def extract_percentiles_from_response(text: str) -> Dict[int, float]:
    """Extract ``{percentile: value}`` pairs from lines like ``Percentile 10: 45``."""
    result: Dict[int, float] = {}
    for line in text.splitlines():
        match = re.search(r"(\d+)[a-z]{0,2}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line, re.I)
        if match:
            result[int(match.group(1))] = float(match.group(2))
    return result


@dataclass
class PercentileForecast:
    percentiles: Dict[int, float]

    def generate_continuous_cdf(self) -> List[float]:
        """Generate a 201 point monotonic CDF using PCHIP interpolation."""
        if not self.percentiles:
            raise ValueError("No percentiles provided")
        items = sorted(self.percentiles.items())
        xs = np.array([p / 100.0 for p, _ in items])
        ys = np.array([v for _, v in items])
        interpolator = PchipInterpolator(xs, ys, extrapolate=True)
        grid = np.linspace(0.0, 1.0, 201)
        values = interpolator(grid)
        monotonic = np.maximum.accumulate(values).tolist()
        logger.debug(
            "[Committee][Utils] Generated continuous CDF with %s points (min=%.3f, max=%.3f)",
            len(monotonic),
            float(monotonic[0]) if monotonic else float("nan"),
            float(monotonic[-1]) if monotonic else float("nan"),
        )
        return monotonic
