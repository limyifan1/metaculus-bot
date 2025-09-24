"""Research utilities used by the committee engine.

This now integrates real AskNews lookups (if configured) similar to the
flow in `fall_template_bot_2025/research.py`. If credentials are not set,
it falls back to lightweight mocked outputs so the rest of the pipeline
still functions.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List
import logging
import os
import random

from asknews_sdk import AsyncAskNewsSDK
from gemini_api import gemini_web_search
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)


def _format_articles(articles) -> str:
    """Best-effort formatting for AskNews article objects or dicts."""
    try:
        sorted_articles = sorted(articles, key=lambda x: x.pub_date, reverse=True)
    except Exception:
        try:
            sorted_articles = sorted(
                articles, key=lambda x: x.get("pub_date"), reverse=True
            )
        except Exception:
            sorted_articles = articles

    out = ""
    for article in sorted_articles:
        try:
            eng_title = getattr(article, "eng_title", None) or article.get(
                "eng_title", ""
            )
            summary = getattr(article, "summary", None) or article.get("summary", "")
            language = getattr(article, "language", None) or article.get("language", "")
            source_id = getattr(article, "source_id", None) or article.get(
                "source_id", ""
            )
            article_url = getattr(article, "article_url", None) or article.get(
                "article_url", ""
            )
            pub_date = getattr(article, "pub_date", None) or article.get("pub_date")
            if hasattr(pub_date, "strftime"):
                pub_date_str = pub_date.strftime("%B %d, %Y %I:%M %p")
            else:
                pub_date_str = str(pub_date) if pub_date is not None else ""
        except Exception:
            eng_title = str(article)
            summary = ""
            language = ""
            source_id = ""
            article_url = ""
            pub_date_str = ""

        out += (
            f"**{eng_title}**\n{summary}\n"
            f"Original language: {language}\n"
            f"Publish date: {pub_date_str}\n"
            f"Source:[{source_id}]({article_url})\n\n"
        )
    return out


async def _asknews_single_search_formatted(query: str) -> str:
    """Call AskNews once and format results similarly to the template bot.

    Requires ASKNEWS_CLIENT_ID and ASKNEWS_SECRET. Strategy and n_articles
    can be controlled with env vars ASKNEWS_STRATEGY and ASKNEWS_N_ARTICLES.
    """
    client_id = os.getenv("ASKNEWS_CLIENT_ID")
    client_secret = os.getenv("ASKNEWS_SECRET")
    if not client_id or not client_secret:
        raise ValueError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")

    strategy = os.getenv("ASKNEWS_STRATEGY", "news knowledge")
    try:
        n_articles = int(os.getenv("ASKNEWS_N_ARTICLES", "10"))
    except Exception:
        n_articles = 10

    async with AsyncAskNewsSDK(
        client_id=client_id, client_secret=client_secret, scopes={"news"}
    ) as ask:
        response = await ask.news.search_news(
            query=query,
            n_articles=n_articles,
            return_type="both",
            strategy=strategy,
        )

    articles = getattr(response, "as_dicts", None)
    formatted = "Here are the relevant news articles:\n\n"
    if articles:
        formatted += _format_articles(articles)
    else:
        formatted += "No articles were found.\n\n"
    logger.info(
        "[Committee][AskNews] Query='%s' | Formatted result (%s chars):\n%s",
        query,
        len(formatted),
        formatted,
    )
    return formatted


async def _gemini_search(query: str) -> str:
    """Run a Gemini web search using available API keys with fallbacks.

    Tries env vars `GEMINI_API_KEY_1`, `_2`, `_3` in order. Returns the raw
    text output. Logs full content for traceability. Falls back to a mocked
    string if no key is set or all attempts fail.
    """
    api_keys = [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ]
    # Retry settings
    try:
        max_retries = int(os.getenv("RESEARCH_RETRIES", "2"))
    except Exception:
        max_retries = 2

    for i, key in enumerate(api_keys, start=1):
        if not key:
            continue
        attempt = 0
        while attempt <= max_retries:
            try:
                logger.info(
                    "[Committee][Gemini] Trying web search (key #%s, attempt %s/%s)",
                    i,
                    attempt + 1,
                    max_retries + 1,
                )
                response = await asyncio.to_thread(gemini_web_search, query, key)
                logger.info(
                    "[Committee][Gemini] Success (key #%s). Result (%s chars)",
                    i,
                    len(response),
                )
                return response
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[Committee][Gemini] Search failed (key #%s, attempt %s/%s): %s",
                    i,
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                attempt += 1
                if attempt <= max_retries:
                    # Exponential backoff with jitter
                    delay = min(8.0, (2 ** (attempt - 1)) + random.random())
                    await asyncio.sleep(delay)
                continue
    # Fallback
    result = f"[Mock Gemini] search result for '{query}'"
    logger.warning(
        "[Committee][Gemini] No API key available or all failed. Using mock. Result: %s",
        result,
    )
    return result


async def _gpt4o_search_preview(query: str) -> str:
    """Run GPT-4o Search (preview) via the same OpenRouter path as llm_calls.

    Uses forecasting_tools.GeneralLlm with model
    "openrouter/openai/gpt-4o-search-preview". Requires OPENROUTER_API_KEY.
    Falls back to a mock string if the key is missing or the call fails.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        result = f"[Mock GPT-4o-Search] search result for '{query}'"
        logger.warning(
            "[Committee][GPT-4o-Search] OPENROUTER_API_KEY not set. Using mock. Result: %s",
            result,
        )
        return result

    prompt = (
        "You are a research assistant. Perform fresh web search and return "
        "a concise, factual roundup for the following query. Include direct "
        "citations with titles and URLs.\n\n"
        f"Query: {query}"
    )

    # Retry settings
    try:
        max_retries = int(os.getenv("RESEARCH_RETRIES", "2"))
    except Exception:
        max_retries = 2

    attempt = 0
    last_error: Exception | None = None
    while attempt <= max_retries:
        llm = GeneralLlm(
            model="openai/gpt-4o-search-preview",
            temperature=None,
            timeout=40,
            allowed_tries=2,
        )
        try:
            logger.info(
                "[Committee][GPT-4o-Search] Invoking GeneralLlm (attempt %s/%s)",
                attempt + 1,
                max_retries + 1,
            )
            text = await llm.invoke(prompt)  # type: ignore[attr-defined]
            text = text or "(No text content returned)"
            logger.info(
                "[Committee][GPT-4o-Search] Success. Result (%s chars)", len(text)
            )
            return text
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "[Committee][GPT-4o-Search] Call failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries + 1,
                e,
            )
            attempt += 1
            if attempt <= max_retries:
                delay = min(8.0, (2 ** (attempt - 1)) + random.random())
                await asyncio.sleep(delay)
            continue

    result = f"[Mock GPT-4o-Search] search result for '{query}'" + (
        f" (error: {last_error})" if last_error else ""
    )
    logger.warning(
        "[Committee][GPT-4o-Search] All retries failed. Using mock. Error: %s",
        last_error,
    )
    return result


async def integrated_research(queries: Iterable[str]) -> str:
    """Run AskNews (single-shot), Gemini, and GPT-4o Search.

    Behavior change: AskNews is now called once with a single combined
    query built from all deduped queries (joined with OR) to use the API
    sparingly, while Gemini and GPT-4o Search retain per-query calls.
    """
    logger.info("[Committee] Starting integrated research over incoming queries")
    # Realize iterator for logging and reuse
    if not isinstance(queries, list):
        queries = list(queries)
    # Deduplicate and keep order
    seen = set()
    deduped: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    queries = deduped
    logger.info("[Committee] Total queries after dedupe: %s", len(queries))
    logger.debug("[Committee] Queries: %s", queries)
    client_id = os.getenv("ASKNEWS_CLIENT_ID")
    client_secret = os.getenv("ASKNEWS_SECRET")

    combined_chunks: List[str] = []
    if client_id and client_secret:
        if queries:
            # Build a single combined AskNews query like: (q1) OR (q2) OR (q3)
            joiner = os.getenv("ASKNEWS_COMBINE_JOINER", " OR ")
            parts = []
            for q in queries:
                q = (q or "").strip()
                if not q:
                    continue
                if any(ch.isspace() for ch in q):
                    parts.append(f"({q})")
                else:
                    parts.append(q)
            combined_query = joiner.join(parts) if parts else ""
            logger.info(
                "[Committee] Using AskNews (single-shot) over %s queries", len(parts)
            )
            try:
                header = (
                    f"## AskNews (combined) over {len(parts)} queries\n"
                    f"Combined query: {combined_query}\n"
                )
                chunk = await _asknews_single_search_formatted(combined_query)
                combined_chunks.append(header + chunk)
            except Exception as e:
                logger.warning(
                    "[Committee] AskNews combined query failed (len=%s): %s",
                    len(combined_query),
                    e,
                )
        else:
            logger.info("[Committee] No queries to send to AskNews.")
    else:
        logger.warning(
            "[Committee] ASKNEWS credentials not set. Skipping AskNews (single-shot)."
        )
        if queries:
            combined_chunks.append(
                f"## AskNews (mock combined) over {len(queries)} queries\nNo articles (mock).\n"
            )

    # Optional Gemini and GPT-4o Search sections with per-provider concurrency
    if queries:
        # Derive provider-specific caps, fall back to a global cap, then default
        def _cap(env_name: str, fallback_env: str, default_val: int) -> int:
            try:
                v = os.getenv(env_name)
                if v is not None:
                    return int(v)
            except Exception:
                pass
            try:
                v2 = os.getenv(fallback_env)
                if v2 is not None:
                    return int(v2)
            except Exception:
                pass
            return default_val

        gem_cap = _cap("RESEARCH_GEMINI_MAX_CONCURRENCY", "RESEARCH_MAX_CONCURRENCY", 4)
        gpt_cap = _cap("RESEARCH_GPT4O_MAX_CONCURRENCY", "RESEARCH_MAX_CONCURRENCY", 4)

        gem_sem = asyncio.Semaphore(gem_cap)
        gpt_sem = asyncio.Semaphore(gpt_cap)

        async def _with_gem_sem(q: str):
            async with gem_sem:
                return await _gemini_search(q)

        async def _with_gpt_sem(q: str):
            async with gpt_sem:
                return await _gpt4o_search_preview(q)

        gemini_tasks = [asyncio.create_task(_with_gem_sem(q)) for q in queries]
        gpt4o_tasks = [asyncio.create_task(_with_gpt_sem(q)) for q in queries]

        gemini_results, gpt4o_results = await asyncio.gather(
            asyncio.gather(*gemini_tasks), asyncio.gather(*gpt4o_tasks)
        )

        # Preserve output order by emitting sections in the original sequence
        for q, gem in zip(queries, gemini_results):
            chunk = f"## Gemini for query: {q}\n{gem}\n"
            logger.info(
                "[Committee][Gemini] Query='%s' | Result (%s chars):\n%s",
                q,
                len(chunk),
                chunk,
            )
            combined_chunks.append(chunk)

        for q, gpt4o in zip(queries, gpt4o_results):
            chunk = f"## GPT-4o Search (preview) for query: {q}\n{gpt4o}\n"
            logger.info(
                "[Committee][GPT-4o-Search] Query='%s' | Result (%s chars):\n%s",
                q,
                len(chunk),
                chunk,
            )
            combined_chunks.append(chunk)

    combined = "\n".join(combined_chunks)
    logger.info(
        "[Committee] Integrated research complete. Sections: %s | Length: %s chars",
        len(combined_chunks),
        len(combined),
    )
    return combined
