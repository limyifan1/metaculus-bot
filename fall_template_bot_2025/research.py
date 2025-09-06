import asyncio
import logging
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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
    async with self._concurrency_limiter:  # type: ignore[attr-defined]
        research = ""
        research_source = "unknown"

        gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
        gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")
        gemini_api_key_3 = os.environ.get("GEMINI_API_KEY_3")

        researcher = self.get_llm("researcher")

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

            if gemini_api_key_2 and not gemini_success:
                try:
                    logger.info("Trying web search with Gemini API Key 2...")
                    response = await asyncio.to_thread(
                        gemini_web_search, prompt, gemini_api_key_2
                    )
                    logger.info(f"Search result: {response}")
                    research = response
                    research_source = "gemini"
                    gemini_success = True
                except Exception as e:
                    logger.warning(f"Gemini search with key 2 failed: {e}")

            if gemini_api_key_3 and not gemini_success:
                try:
                    logger.info("Trying web search with Gemini API Key 3...")
                    response = await asyncio.to_thread(
                        gemini_web_search, prompt, gemini_api_key_3
                    )
                    logger.info(f"Search result: {response}")
                    research = response
                    research_source = "gemini"
                    gemini_success = True
                except Exception as e:
                    logger.warning(f"Gemini search with key 3 failed: {e}")
        elif not researcher or researcher == "None":
            research = ""
            research_source = "none"
        else:
            llm = self.get_llm("researcher", "llm")
            research = await llm.invoke(prompt)
            research_source = "llm"

        logger.info(
            f"Found Research for URL {question.page_url} | source={research_source}:\n{research}"
        )
        return research
