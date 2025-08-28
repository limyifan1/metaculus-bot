# Technical Design Document: Multi-Agent Forecasting System

## 1.0 Executive Summary & Core Design Philosophy
*   **1.1 High-Level Overview:** This document outlines the technical architecture of a sophisticated probabilistic forecasting system designed to compete in Metaculus tournaments. The system employs a multi-agent, multi-step reasoning process to analyze forecasting questions, conduct comprehensive research, and generate statistically aggregated predictions. It is architected to handle binary, multiple-choice, and numeric question types through specialized, asynchronous workflows, leveraging a heterogeneous "committee" of Large Language Models (LLMs) to produce robust and well-calibrated forecasts.

*   **1.2 Core Design Philosophy:** The system is built upon several key principles to maximize forecasting accuracy and robustness:
    *   **Multi-Agent Systems:** Instead of relying on a single model, the system uses a "committee" of five distinct LLM agents. This approach introduces cognitive diversity and reduces the risk of idiosyncratic errors or biases from a single model.
    *   **Ensemble Methods:** Individual forecasts from each agent are not treated equally. They are combined using a weighted average, a classic ensemble technique that leverages the strengths of different models to produce a more accurate and stable final prediction.
    *   **Prompt Chaining:** The forecasting process is not monolithic but is broken down into a sequence of distinct, interconnected phases guided by prompt chaining. Each phase's output serves as the input for the next, creating a structured cognitive workflow that guides the LLMs from broad research to a final, synthesized prediction.
    *   **Up-to-Date Research:** The system's reasoning is grounded in current information gathered via AskNews and Gemini 2.5 Flash with the `google_search` tool. It does not perform general web content extraction or agentic browsing.
    *   **Cognitive Diversity:** The use of different model families (Anthropic's Claude Sonnet 4 via OpenRouter, OpenAI's GPT-5 via OpenRouter, and Google's Gemini 2.5 Pro) is a deliberate strategy to ensure a variety of reasoning patterns and "world models" are applied to each question.
    *   **Iterative Refinement & Simulated Debate:** The system employs a multi-step forecasting process where initial analyses are generated and then "cross-examined." The reasoning of one agent is passed as context to a different agent in a subsequent step, simulating a peer-review or debate process that refines and challenges initial conclusions before a final forecast is made.

## 2.0 System Architecture & End-to-End Workflow
*   **2.1 Question Ingestion & Initialization:** The process originates in `Bot/main.py`. The `get_open_question_ids_from_tournament` function makes an API call to Metaculus, using a predefined `TOURNAMENT_ID` to fetch a list of all open questions. The main execution block then iterates through this list, initiating the `forecast_individual_question` asynchronous task for each question. This function checks if a forecast has already been made (`forecast_is_already_made`) and, if not, proceeds to call the appropriate forecasting orchestrator.

*   **2.2 Forecasting Orchestration (Per Question Type):** The script `Bot/forecaster.py` acts as the central router, delegating tasks to specialized modules based on the question type retrieved from the question's metadata. The `forecast_individual_question` function in `Bot/main.py` checks the `question_type` field and calls the corresponding high-level async function (e.g., `binary_forecast`, `multiple_choice_forecast`, `numeric_forecast`) defined in `Bot/forecaster.py`, which in turn call the detailed implementations in their respective modules.

    *   **2.2.1 Binary Question Workflow:** The end-to-end logic is implemented in `Bot/binary.py` within the `get_binary_forecast` function.
        1.  **Initial Research Scoping:** Two initial prompts, `BINARY_PROMPT_historical` and `BINARY_PROMPT_current`, are formatted with the question details and sent concurrently to generate targeted search directions.
        2.  **Integrated Research (AskNews + Gemini Search):** The scoped queries inform calls to AskNews and to Gemini 2.5 Flash with the `google_search` tool; their summarized results are carried directly into the next step. No web scraping, Google/Google News direct APIs, or agentic search loops are used.
        3.  **Initial Forecast Generation:** The integrated research context is used to format `BINARY_PROMPT_1`. This prompt is run concurrently across the multi-model committee to generate an "outside view" prediction.
        4.  **Simulated Peer Review:** The outputs from the first forecast are strategically rearranged. A `context_map` is created where the analysis from one agent is combined with the integrated research context and passed to a *different* agent for the next phase.
        5.  **Final Forecast Synthesis:** `BINARY_PROMPT_2` is formatted with this combined peer-review context and run across all agents.
        6.  **Aggregation:** The final probability from each agent's output is extracted using `extract_probability_from_response_as_percentage_not_decimal`. These probabilities are combined using a weighted average, and the result is clamped to the range [0.001, 0.999].

    *   **2.2.2 Multiple-Choice Question Workflow:** This flow, detailed in `Bot/multiple_choice.py`'s `get_multiple_choice_forecast` function, mirrors the binary workflow's structure but uses multiple-choice specific prompts (`MULTIPLE_CHOICE_PROMPT_historical`, `_current`, `_1`, `_2`).
        1.  It follows the same dual-perspective research scoping and integrated research process (AskNews + Gemini 2.5 Flash `google_search`), without a standalone RAG phase.
        2.  The initial and final forecasts generate probability distributions across the available options.
        3.  The function `extract_option_probabilities_from_response` is used to parse the list of probabilities from the LLM's structured output.
        4.  These probability lists are normalized (`normalize_probabilities`) to ensure they sum to 1.0.
        5.  The final aggregation computes a weighted average for each option's probability across all agents.

    *   **2.2.3 Numeric Question Workflow:** Implemented in `Bot/numeric.py`, this workflow also follows the established multi-phase pattern with numeric-specific prompts.
        1.  The research and initial analysis phases are analogous to the other types.
        2.  The key difference is the output format. The LLMs are prompted to produce a series of discrete percentile points (e.g., 10th, 25th, 50th, 75th, 90th).
        3.  The `extract_percentiles_from_response` function parses these key-value pairs from the text.
        4.  The core technical step is in `generate_continuous_cdf`, which takes these discrete points and uses `scipy.interpolate.PchipInterpolator` to create a smooth, monotonically increasing Cumulative Distribution Function (CDF) across 201 points, the format required for the Metaculus API.
        5.  The final CDFs from all agents are aggregated using a weighted average.

*   **2.3 The Multi-Agent "Committee" Model:**
    *   **2.3.1 Agent Composition:** The forecasting committee mixes the following models:
        *   `openrouter/claude-sonnet-4` via GeneralLLM
        *   `openrouter/gpt-5` via GeneralLLM
        *   `gemini-2.5-pro` via `genai.Client`

    *   **2.3.2 Rationale for Heterogeneity:** Using a mix of Anthropic (Claude Sonnet 4), OpenAI via OpenRouter (GPT-5), and Google (Gemini 2.5 Pro) is a deliberate design choice. This heterogeneity introduces cognitive diversity, helping to mitigate against the inherent biases or systematic reasoning flaws of any single model architecture and producing a more robust and balanced ensemble.

    *   **2.3.3 Agent Weighting:** Forecasts are aggregated with configurable weights. A simple starting point is equal weighting across agents, with the option to tune weights empirically based on validation.

*   **2.4 The Multi-Step Reasoning Process (Prompt Chaining):**
    *   **2.4.1 Phase 1: Dual-Perspective Scoping & Research Direction:** The process begins with two parallel prompts, for example `BINARY_PROMPT_historical` and `BINARY_PROMPT_current`. Their explicit goal is not to forecast but to scope the problem from two distinct viewpoints. The `_historical` prompt frames the problem from an "outside view," asking for historical precedents and base rates. The `_current` prompt takes an "inside view," focusing on recent events and key actors. The output of this phase is a structured list of targeted search queries for the next phase.

    *   **2.4.2 Phase 2: Integrated Research (No RAG):**
        *   **AskNews:** Retrieve relevant, structured news context with the AskNews SDK based on the scoped queries.
        *   **Gemini Search Tool:** Use Gemini 2.5 Flash with the `google_search` tool to surface recent, high-signal sources. No Google/Google News direct API usage, no agentic search loops, and no web content extraction (e.g., headless browsers or HTML scraping) are performed.
        *   The result is a concise, synthesized research context passed forward as-is; there is no separate RAG pipeline.

    *   **2.4.3 Phase 3: Initial Forecast & Simulated Peer Review:** The integrated research context is used to create the first forecast prompt (e.g., `BINARY_PROMPT_1`). Each agent generates an independent initial forecast. The critical step occurs in `Bot/binary.py`'s `context_map`, which swaps analyses between agents to simulate peer review across model families, encouraging a more robust synthesis of information.

    *   **2.4.4 Phase 4: Final Synthesis & Prediction:** The final context is assembled by combining the integrated research (Phase 2) with the peer-reviewed initial forecast (Phase 3). This is fed into the final prompt (e.g., `BINARY_PROMPT_2`), and each agent produces its refined forecast.

*   **2.5 Ensemble & Aggregation:**
    *   **2.5.1 Statistical Method:** Individual forecasts are aggregated using a weighted average. In files like `Bot/binary.py`, this is implemented using `numpy.sum` on the weighted probabilities, divided by the sum of the weights. This combines the five distinct predictions into a single, calibrated forecast.

    *   **2.5.2 Risk Management:** The final aggregated probability is explicitly clamped to a safe range, such as [0.001, 0.999] in `Bot/binary.py`. This is a critical risk management step that prevents the bot from making overly confident (0% or 100%) predictions, which are heavily penalized in most scoring systems if they are wrong.

*   **2.6 Output Generation & Submission:** The final steps are handled in `Bot/main.py`. The `create_forecast_payload` function formats the aggregated forecast into the specific JSON structure required by the Metaculus API for the given question type. The `post_question_prediction` function then submits this payload. In parallel, the combined reasoning from all agents is summarized by a final LLM call, and this concise explanation is posted as a private comment on the question page via `post_question_comment`.

*   **2.7 End-to-End Forecasting Flow (Summary):**
    *   **Ingest Question:** Fetch open questions and skip those already forecasted.
    *   **Route by Type:** Dispatch to `binary`, `multiple_choice`, or `numeric` pipelines.
    *   **Dual-Perspective Scoping:** Run `_historical` (outside view) and `_current` (inside view) prompts in parallel to produce targeted queries.
    *   **Integrated Research:** Use AskNews and Gemini 2.5 Flash `google_search` to gather and summarize recent, forecasting-relevant context (no RAG, scraping, or agentic browsing).
    *   **Initial Forecast (Committee):** Provide the research pack to `*_PROMPT_1` and run concurrently across the heterogeneous model committee for independent estimates.
    *   **Simulated Peer Review:** Swap initial analyses between agents (`context_map`) so each agent sees another’s reasoning plus the research pack.
    *   **Final Synthesis:** Run `*_PROMPT_2` with research + peer-reviewed context to get refined forecasts from all agents.
    *   **Ensemble Aggregation:** Extract structured outputs, apply per-agent weights, aggregate, and apply calibration safeguards (e.g., clamping/normalization as appropriate).
    *   **Submit Forecast:** Convert to Metaculus payload (binary probability, MC distribution, or numeric 201-point CDF) and post.

    *   **Shared Forecasting Mechanics:**
        *   **Multi-Agent Committee:** Mix model families to increase cognitive diversity and robustness.
        *   **Weights:** Start equal; configurable for empirical tuning by validation.
        *   **Prompt Scaffolding:** Enforce stepwise reasoning and strict, parseable output (e.g., "Probability: ZZ%", "[p1, p2, …]").
        *   **De-biasing:** Include explicit instructions (e.g., status-quo weighting) to counter recency/drama bias and improve calibration.

    *   **Per-Question Type Outputs:**
        *   **Binary:** Parse final percentage via `extract_probability_from_response_as_percentage_not_decimal`, aggregate with weights, and clamp to [0.001, 0.999].
        *   **Multiple-Choice:** Parse option list via `extract_option_probabilities_from_response`, normalize with `normalize_probabilities`, then aggregate per option.
        *   **Numeric:** Parse discrete percentiles via `extract_percentiles_from_response`, fit a smooth monotone CDF with `generate_continuous_cdf` (PCHIP, 201 points), and aggregate CDFs.

    *   **Concurrency & Orchestration:** Per-question async orchestration; within phases, concurrent committee calls. The peer-review step intentionally cross-pollinates reasoning across model families.

## 3.0 Key Modules & Technical Implementation Details
*   **3.1 LLM Interaction (`Bot/llm_calls.py`):**
    *   This layer abstracts LLM APIs, using GeneralLLM for OpenRouter models (`openrouter/claude-sonnet-4`, `openrouter/gpt-5`) and `genai.Client` for `gemini-2.5-pro`, decoupling forecasting logic from provider specifics.
    *   It utilizes `asyncio` and `aiohttp` to manage concurrent API requests across agents, significantly speeding up end-to-end forecasting.
    *   API calls implement exponential backoff for transient errors (e.g., 429/503), with bounded retries and jittered delays for resilience.

*   **3.2 Research & News (`Bot/search.py`):**
    *   Research is performed via AskNews and Gemini 2.5 Flash with the `google_search` tool, guided by scoped queries from Phase 1.
    *   There is no agentic search loop, no direct Google/Google News API usage, and no web content extraction or headless browsing. The focus is on concise, structured summaries suitable for prompt contexts.

*   **3.3 Numeric Distribution Generation (`Bot/numeric.py`):**
    *   This module contains the logic for handling numeric forecasting questions, which require a full probability distribution as output. The `extract_percentiles_from_response` function uses regular expressions to parse discrete percentile points (e.g., "Percentile 10: 50", "Percentile 90: 200") from the unstructured text output of an LLM.
    *   The core of the module is the `generate_continuous_cdf` function. It takes the sparse set of percentile points and uses `scipy.interpolate.PchipInterpolator` to perform a Piecewise Cubic Hermite Interpolating Polynomial interpolation. This method is specifically chosen because it preserves monotonicity, ensuring the resulting CDF is always non-decreasing. It generates a smooth 201-point CDF array, which is the precise format required by the Metaculus API for numeric question submissions.

    

## 4.0 Prompt Engineering Strategy
*   **4.1 Core Principles:** The prompt design across the system adheres to a consistent set of advanced principles:
    *   **Role-Playing:** Prompts consistently begin with instructions like, "You are a professional forecaster..." This primes the LLM to adopt a specific persona, activating the desired knowledge and reasoning patterns associated with that role.
    *   **Cognitive Scaffolding:** The prompts impose a mandatory, step-by-step reasoning structure. For instance, in `BINARY_PROMPT_TEMPLATE`, the model is forced to explicitly consider "(a) The time left...", "(b) The status quo outcome...", "(c) A brief description of a scenario that results in a No outcome...", etc. This scaffolding prevents the model from jumping to a conclusion and ensures a more thorough, structured analysis.
    *   **Structured Output:** The prompts enforce strict output formatting, such as "Probability: ZZ%" or "Probabilities: [Probability_1, Probability_2, ..., Probability_N]". This is not just for neatness; it is essential for reliable, programmatic parsing of the final prediction by functions like `extract_probability_from_response_as_percentage_not_decimal`.
    *   **Deliberate De-biasing:** Prompts include strategic instructions designed to mitigate known cognitive biases. The recurring phrase, "good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time," is a direct intervention to counteract the tendency of LLMs to over-index on recent news and predict dramatic changes, thereby improving the forecast's calibration.

*   **4.2 Prompt Breakdown by Phase:** For the binary workflow, the prompts form a logical chain, guiding the LLMs from research to prediction.
    *   **`_historical` and `_current`:** These initial prompts are for research direction. They do not ask for a prediction. Instead, they ask the LLM to think about the question from an "outside view" (historical) and an "inside view" (current events) and to generate a list of search queries appropriate for each perspective.
    *   **`assistant_prompt`:** Used to summarize outputs from AskNews and Gemini 2.5 Flash `google_search`, focusing only on forecasting-relevant information. No raw HTML extraction is performed.
    *   **`_1` (e.g., `BINARY_PROMPT_1`):** This is the first forecasting prompt. It provides the LLM with the integrated research context and asks for an initial analysis and prediction. This serves as the "outside view" or base-rate forecast.
    *   **`_2` (e.g., `BINARY_PROMPT_2`):** This is the final synthesis prompt. It provides the LLM with the integrated research context and the peer-reviewed analysis from another agent's output on `PROMPT_1`. It asks the LLM to integrate these sources of information to produce its final, most considered forecast.
