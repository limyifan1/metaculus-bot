import asyncio
import logging
import os

from forecasting_tools import GeneralLlm, MetaculusQuestion
from asknews_sdk import AsyncAskNewsSDK
from gemini_api import gemini_web_search

logger = logging.getLogger(__name__)


async def _asknews_single_search_formatted(query: str) -> str:
    """Call AskNews once and return formatted articles.

    Strategy and n_articles are configurable via env vars:
    - ASKNEWS_STRATEGY: one of AskNews strategies (default: "news knowledge")
    - ASKNEWS_N_ARTICLES: integer count (default: 10)
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
        client_id=client_id,
        client_secret=client_secret,
        scopes={"news"},
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
    return formatted


def _format_articles(articles) -> str:
    try:
        sorted_articles = sorted(articles, key=lambda x: x.pub_date, reverse=True)
    except Exception:
        # Fallback if objects are dict-like
        try:
            sorted_articles = sorted(
                articles,
                key=lambda x: x.get("pub_date"),
                reverse=True,
            )
        except Exception:
            sorted_articles = articles

    out = ""
    for article in sorted_articles:
        try:
            eng_title = getattr(article, "eng_title", None) or article.get("eng_title", "")
            summary = getattr(article, "summary", None) or article.get("summary", "")
            language = getattr(article, "language", None) or article.get("language", "")
            source_id = getattr(article, "source_id", None) or article.get("source_id", "")
            article_url = getattr(article, "article_url", None) or article.get("article_url", "")
            pub_date = getattr(article, "pub_date", None) or article.get("pub_date")
            if hasattr(pub_date, "strftime"):
                pub_date_str = pub_date.strftime("%B %d, %Y %I:%M %p")
            else:
                pub_date_str = str(pub_date) if pub_date is not None else ""
        except Exception:
            # If structure surprises us, best-effort stringify
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
        if isinstance(researcher, GeneralLlm):
            research_source = "general-llm"
            research = await researcher.invoke(prompt)
        elif researcher == "asknews/news-summaries":
            research_source = "asknews/news-summaries"
            logger.info(
                "Using AskNews researcher for URL %s | query preview: %s",
                getattr(question, "page_url", "<unknown>"),
                (
                    (question.question_text[:200] + "…")
                    if len(question.question_text) > 200
                    else question.question_text
                ),
            )
            # Handle AskNews rate limits (429) by respecting Retry-After
            max_retries = 3
            last_err: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        "Calling AskNews (single-search) (attempt %s/%s) for URL %s",
                        attempt,
                        max_retries,
                        getattr(question, "page_url", "<unknown>"),
                    )
                    research = await _asknews_single_search_formatted(
                        question.question_text
                    )
                    # Log what AskNews returned (summary + preview)
                    try:
                        preview_len = int(os.getenv("ASKNEWS_LOG_PREVIEW_CHARS", "800"))
                    except Exception:
                        preview_len = 800
                    preview = research[:preview_len] + (
                        "…" if len(research) > preview_len else ""
                    )
                    logger.info(
                        "AskNews returned formatted research (%s chars). Preview:\n%s",
                        len(research),
                        preview,
                    )
                    logger.debug("AskNews full research:\n%s", research)
                    break
                except (
                    Exception
                ) as e:  # noqa: BLE001 - want to catch SDK-specific errors too
                    last_err = e

                    def _extract_status_and_retry_after(
                        ex: Exception,
                    ) -> tuple[int | None, int | None]:
                        """Best-effort extraction of status and retry-after seconds.

                        Tries common headers: Retry-After, RateLimit-Reset, X-RateLimit-Reset,
                        and also attempts to parse HTTP-date values. Falls back to None.
                        """
                        status_code = None
                        retry_after_seconds = None
                        resp = getattr(ex, "response", None)
                        if resp is not None:
                            status_code = getattr(resp, "status_code", None)
                            headers = getattr(resp, "headers", {}) or {}

                            # Log all headers at debug level to help diagnose header naming
                            try:
                                logger.debug(
                                    "AskNews error headers: %s",
                                    dict(headers) if hasattr(headers, "items") else headers,
                                )
                            except Exception:
                                pass

                            # Case-insensitive getter helper
                            def _hget(h: str) -> str | None:
                                try:
                                    return (
                                        headers.get(h)
                                        or headers.get(h.lower())
                                        or headers.get(h.title())
                                    )
                                except Exception:
                                    return None

                            # 1) Standard Retry-After header (seconds or HTTP-date)
                            retry_after_val = _hget("Retry-After")

                            # 2) RFC RateLimit-Reset: seconds until reset
                            if not retry_after_val:
                                retry_after_val = _hget("RateLimit-Reset")

                            # 3) Common vendor headers: X-RateLimit-Reset (epoch seconds or ms)
                            xrate_reset = _hget("X-RateLimit-Reset") or _hget(
                                "X-Rate-Limit-Reset"
                            )

                            # 4) Sometimes providers expose exact reset time
                            reset_at = _hget("X-RateLimit-Reset-At") or _hget(
                                "RateLimit-Reset-At"
                            )

                            # Parse Retry-After / RateLimit-Reset if present
                            if retry_after_val:
                                s = str(retry_after_val).strip()
                                try:
                                    # Numeric seconds delta
                                    retry_after_seconds = int(s)
                                except ValueError:
                                    # HTTP-date
                                    try:
                                        dt = parsedate_to_datetime(s)
                                        if dt.tzinfo is None:
                                            dt = dt.replace(tzinfo=timezone.utc)
                                        retry_after_seconds = max(
                                            0,
                                            int(
                                                (
                                                    dt - datetime.now(timezone.utc)
                                                ).total_seconds()
                                            ),
                                        )
                                    except Exception:
                                        retry_after_seconds = None

                            # If still None, try X-RateLimit-Reset epoch parsing
                            if retry_after_seconds is None and xrate_reset:
                                try:
                                    s = str(xrate_reset).strip()
                                    ts = int(float(s))
                                    # Heuristic: treat very large values as ms
                                    if ts > 10_000_000_000:  # > ~Nov 2286 in seconds
                                        ts = ts // 1000
                                    now = int(datetime.now(timezone.utc).timestamp())
                                    retry_after_seconds = max(0, ts - now)
                                except Exception:
                                    pass

                            # If still None, try reset-at as datetime string
                            if retry_after_seconds is None and reset_at:
                                try:
                                    dt = parsedate_to_datetime(str(reset_at))
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=timezone.utc)
                                    retry_after_seconds = max(
                                        0,
                                        int(
                                            (
                                                dt - datetime.now(timezone.utc)
                                            ).total_seconds()
                                        ),
                                    )
                                except Exception:
                                    pass

                            # As a last resort, attempt to parse seconds from a JSON body message
                            if retry_after_seconds is None:
                                try:
                                    body = getattr(resp, "text", None)
                                    if body and isinstance(body, str):
                                        import json, re

                                        retry_after_seconds = None
                                        try:
                                            j = json.loads(body)
                                        except Exception:
                                            j = None
                                        if isinstance(j, dict):
                                            for k in (
                                                "retry_after",
                                                "retry-after",
                                                "reset",
                                                "reset_seconds",
                                                "reset_after",
                                            ):
                                                if k in j:
                                                    try:
                                                        retry_after_seconds = int(
                                                            float(str(j[k]))
                                                        )
                                                        break
                                                    except Exception:
                                                        pass
                                            if (
                                                retry_after_seconds is None
                                                and isinstance(j.get("detail"), str)
                                            ):
                                                m = re.search(
                                                    r"(?i)try again in\s+(\d+)\s*sec",
                                                    j["detail"],
                                                )
                                                if m:
                                                    retry_after_seconds = int(m.group(1))
                                    # If not JSON, try regex on plain text
                                    if retry_after_seconds is None and body:
                                        import re as _re

                                        m2 = _re.search(
                                            r"(?i)retry-?after\s*:\s*(\d+)", body
                                        )
                                        if m2:
                                            retry_after_seconds = int(m2.group(1))
                                except Exception:
                                    pass

                        if status_code is None:
                            status_code = getattr(ex, "status_code", None)
                        return status_code, retry_after_seconds

                    status, retry_after = _extract_status_and_retry_after(e)
                    is_rate_limit = False
                    if status == 429:
                        is_rate_limit = True
                    else:
                        name = e.__class__.__name__
                        msg = str(e)
                        if (
                            ("RateLimit" in name)
                            or ("429" in msg)
                            or ("Rate Limit Exceeded" in msg)
                        ):
                            is_rate_limit = True

                    if not is_rate_limit or attempt == max_retries:
                        status, retry_after = _extract_status_and_retry_after(e)
                        logger.warning(
                            "AskNews research failed (attempt %s/%s). Status: %s, Retry-After: %s, Error: %s",
                            attempt,
                            max_retries,
                            status,
                            retry_after,
                            e,
                        )
                        if attempt == max_retries:
                            raise
                        wait_s = 5
                        logger.info(f"Retrying AskNews in {wait_s}s (transient error)")
                        await asyncio.sleep(wait_s)
                        continue

                    wait_seconds = (
                        retry_after
                        if (retry_after is not None and retry_after >= 0)
                        else 60
                    )
                    logger.warning(
                        f"AskNews rate limited (429). Retry-After: {retry_after if retry_after is not None else 'N/A'}; waiting {wait_seconds}s before retry {attempt+1}/{max_retries}"
                    )
                    await asyncio.sleep(wait_seconds)
            else:
                raise (last_err if last_err else RuntimeError("Unknown AskNews error"))
        elif researcher == "gemini":
            gemini_success = False
            if gemini_api_key_1 and not gemini_success:
                try:
                    logger.info("Trying web search with Gemini API Key 1...")
                    response = await asyncio.to_thread(
                        gemini_web_search, prompt, gemini_api_key_1
                    )
                    logger.info(f"Search result: {response}")
                    research = response
                    research_source = "gemini"
                    gemini_success = True
                except Exception as e:
                    logger.warning(f"Gemini search with key 1 failed: {e}")
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

