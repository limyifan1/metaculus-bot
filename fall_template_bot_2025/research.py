import asyncio
import logging
import os

from forecasting_tools import AskNewsSearcher, GeneralLlm, MetaculusQuestion
from gemini_api import gemini_web_search

logger = logging.getLogger(__name__)


async def run_research(self, question: MetaculusQuestion) -> str:  # noqa: ANN001
    """Run multi-source research and combine into one report.

    Attempts all of the following in parallel:
    - AskNews news summaries
    - OpenRouter `openrouter/gpt-4o-search-preview`
    - Gemini web search (tries GEMINI_API_KEY_1..3)

    If any fails, we log and ignore it, returning a combined report from
    whatever succeeded. Never raises.
    """
    async with self._concurrency_limiter:  # type: ignore[attr-defined]
        from forecasting_tools import clean_indents
        from .prompts import research_prompt

        prompt = clean_indents(
            research_prompt(
                question.question_text,
                question.resolution_criteria,
                question.fine_print,
            )
        )

        # Prepare env once
        gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
        gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")
        gemini_api_key_3 = os.environ.get("GEMINI_API_KEY_3")

        # Define individual fetchers with internal error handling.
        async def fetch_asknews() -> str | None:
            try:
                logger.info(
                    "AskNews: fetching for URL %s | query preview: %s",
                    getattr(question, "page_url", "<unknown>"),
                    (
                        (question.question_text[:200] + "…")
                        if len(question.question_text) > 200
                        else question.question_text
                    ),
                )
                text = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
                logger.info(
                    "AskNews: got %s chars",
                    len(text) if isinstance(text, str) else 0,
                )
                return text
            except Exception as e:  # noqa: BLE001
                logger.warning("AskNews failed: %s", e)
                return None

        async def fetch_gpt4o() -> str | None:
            try:
                # Use explicit model to ensure search-capable variant is used.
                gpt4o = GeneralLlm(
                    model="openrouter/gpt-4o-search-preview",
                    temperature=None,
                    timeout=60,
                    allowed_tries=2,
                )
                logger.info("GPT-4o Search: invoking model for research")
                text = await gpt4o.invoke(prompt)
                logger.info(
                    "GPT-4o Search: got %s chars",
                    len(text) if isinstance(text, str) else 0,
                )
                return text
            except Exception as e:  # noqa: BLE001
                logger.warning("GPT-4o Search failed: %s", e)
                return None

        async def fetch_gemini() -> str | None:
            # Try keys in order; run in thread because client is sync.
            for idx, key in enumerate(
                [gemini_api_key_1, gemini_api_key_2, gemini_api_key_3], start=1
            ):
                if not key:
                    continue
                try:
                    logger.info("Gemini: trying API key %s", idx)
                    text = await asyncio.to_thread(gemini_web_search, prompt, key)
                    logger.info(
                        "Gemini: got %s chars",
                        len(text) if isinstance(text, str) else 0,
                    )
                    return text
                except Exception as e:  # noqa: BLE001
                    logger.warning("Gemini with key %s failed: %s", idx, e)
                    continue
            return None

        # Launch all three in parallel and gather results.
        asknews_task = asyncio.create_task(fetch_asknews())
        gpt4o_task = asyncio.create_task(fetch_gpt4o())
        gemini_task = asyncio.create_task(fetch_gemini())

        results = await asyncio.gather(asknews_task, gpt4o_task, gemini_task)
        asknews_res, gpt4o_res, gemini_res = results

        sources_included: list[str] = []
        parts: list[str] = []
        if asknews_res:
            sources_included.append("AskNews")
            parts.append(
                "AskNews Summary\n----------------\n" + asknews_res.strip()
            )
        if gpt4o_res:
            sources_included.append("GPT-4o Search Preview")
            parts.append(
                "OpenRouter GPT-4o Search Preview\n-----------------------------------\n"
                + gpt4o_res.strip()
            )
        if gemini_res:
            sources_included.append("Gemini Search")
            parts.append("Gemini Search\n-------------\n" + gemini_res.strip())

        # Combine into one integrated report. If nothing succeeded, return empty string.
        if not parts:
            logger.warning(
                "Research: all sources failed for URL %s; proceeding without research",
                getattr(question, "page_url", "<unknown>"),
            )
            return ""

        combined = (
            "Integrated Research Report\n============================\n\n"
            + "\n\n\n".join(parts)
        )

        logger.info(
            "Found research for URL %s | sources=%s",
            getattr(question, "page_url", "<unknown>"),
            ", ".join(sources_included) if sources_included else "none",
        )
        return combined

