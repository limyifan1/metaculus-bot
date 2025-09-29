import argparse
import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Literal

from forecasting_tools import GeneralLlm, MetaculusApi
from fall_template_bot_2025 import FallTemplateBot2025
from bot.adapter import CommitteeForecastBot


if __name__ == "__main__":
    # Configure logging to both console and a uniquely-named file per run
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    unique_suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    log_file_path = os.path.join(logs_dir, f"run_{unique_suffix}.log")

    class _ProgressOnlyFilter(logging.Filter):
        """Allow progress-tagged logs (or errors) through the console handler."""

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            return getattr(record, "is_progress", False) or record.levelno >= logging.ERROR

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    verbose_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (progress-only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.addFilter(_ProgressOnlyFilter())
    ch.setFormatter(logging.Formatter("%(message)s"))

    # File handler (full verbosity)
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(verbose_formatter)

    # Avoid duplicate handlers if re-run in same interpreter
    root_logger.handlers = []
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

    progress_logger = logging.getLogger("progress")
    progress_logger.propagate = True

    def log_progress(message: str, *args: object) -> None:
        """Emit a progress update that is visible in GitHub Actions output."""

        progress_logger.info(message, *args, extra={"is_progress": True})

    logging.getLogger(__name__).info("Logging to file: %s", log_file_path)

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["template", "committee"],
        default="committee",
        help="Choose forecasting engine: 'template' (FallTemplateBot2025) or 'committee' (bot/ adapter)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    engine: Literal["template", "committee"] = args.engine
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"
    log_progress("Step: Starting forecasting run (mode=%s, engine=%s)", run_mode, engine)
    if engine == "template":
        template_bot = FallTemplateBot2025(
            research_reports_per_question=1,
            predictions_per_research_report=5,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=True,
            llms={
                "default": GeneralLlm(
                    model="openrouter/gpt-4.1-mini",
                    temperature=None,
                    timeout=40,
                    allowed_tries=2,
                ),
                "summarizer": "openai/gpt-4.1-nano",
                # "researcher": GeneralLlm(
                #     model="openrouter/gpt-4o-search-preview",
                #     temperature=None,
                #     timeout=40,
                #     allowed_tries=2,
                # ),
                "researcher": "asknews/news-summaries",
            },
        )
    else:
        # CommitteeForecastBot uses local deterministic committee logic; keep posting behavior the same.
        template_bot = CommitteeForecastBot(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=True,
            # LLMs only used for optional summarization; leave defaults minimal.
            llms={
                "default": GeneralLlm(
                    model="openrouter/openai/gpt-4.1-mini",
                    temperature=None,
                    timeout=40,
                    allowed_tries=2,
                ),
                "summarizer": "openai/gpt-4.1-nano",
                "researcher": None,
            },
        )

    if run_mode == "tournament":
        log_progress(
            "Step: Forecasting current AI competition questions",
        )
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        log_progress("Step: Forecasting minibench questions")
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        log_progress("Step: Forecasting Metaculus Cup questions")
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            # "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            # "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            # "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
            "https://www.metaculus.com/questions/11112/us-military-response-to-invasion-of-taiwan/"
        ]
        log_progress("Step: Forecasting test questions (%s total)", len(EXAMPLE_QUESTIONS))
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
    total_reports = len(forecast_reports) if forecast_reports is not None else 0

    failures = 0
    for report in forecast_reports or []:
        if report is None:
            failures += 1
            continue
        if isinstance(report, BaseException):
            failures += 1
            continue
        if getattr(report, "exception", None):
            failures += 1
            continue

    successes = total_reports - failures
    log_progress(
        "Summary: %s total forecasts (%s succeeded, %s failed)",
        total_reports,
        successes,
        failures,
    )
    if failures:
        log_progress("Summary: %s failures recorded; check verbose artifact for details", failures)
    log_progress("Summary: Verbose log captured at %s", log_file_path)
