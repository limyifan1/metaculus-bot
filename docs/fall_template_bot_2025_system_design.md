# Technical Design Document: Multi-Agent Forecasting System

## 1.0 Executive Summary & Core Design Philosophy
*   **1.1 High-Level Overview:** This document outlines the technical architecture of a sophisticated probabilistic forecasting system designed to compete in Metaculus tournaments. The system employs a multi-agent, multi-step reasoning process to analyze forecasting questions, conduct comprehensive research, and generate statistically aggregated predictions. It is architected to handle binary, multiple-choice, and numeric question types through specialized, asynchronous workflows, leveraging a heterogeneous "committee" of Large Language Models (LLMs) to produce robust and well-calibrated forecasts.

*   **1.2 Core Design Philosophy:** The system is built upon several key principles to maximize forecasting accuracy and robustness:
    *   **Multi-Agent Systems:** Instead of relying on a single model, the system uses a "committee" of LLM agents. By default, three real models are used, and the set is configurable via environment variables. This approach introduces cognitive diversity and reduces the risk of idiosyncratic errors or biases from a single model.
    *   **Ensemble Methods:** Individual forecasts from each agent are combined using a weighted average (equal weights by default), a classic ensemble technique that leverages the strengths of different models to produce a more accurate and stable final prediction.
    *   **Prompt Chaining:** The forecasting process is broken down into a sequence of distinct, interconnected phases. Each phase's output serves as the input for the next, creating a structured workflow that guides the LLMs from scoped research to a final, synthesized prediction.
    *   **Up-to-Date Research:** The system's reasoning is grounded in current information gathered via AskNews and Gemini 2.5 Flash with the `google_search` tool, with an optional GPT‑4o Search (preview) integration via OpenRouter when configured. It does not perform general web content extraction or agentic browsing loops.
    *   **Cognitive Diversity:** The default committee mixes multiple model families and variants (e.g., Anthropic via OpenRouter and OpenAI via OpenRouter). Additional models can be added via configuration to further increase heterogeneity.
    *   **Iterative Refinement & Simulated Debate:** The system employs a multi-step forecasting process where initial analyses are generated and then "cross-examined." The reasoning of one agent is passed as context to a different agent in a subsequent step, simulating a peer-review process that refines and challenges initial conclusions before a final forecast is made.

## 2.0 System Architecture & End-to-End Workflow
*   **2.1 Question Ingestion & Initialization:** The process originates in `main.py` (repository root). You can select the engine via a CLI flag: the FallTemplateBot2025 reference engine (`--engine template`) or the committee engine (`--engine committee`). When using the committee engine, `CommitteeForecastBot` (`bot/adapter.py`) integrates with the `forecasting_tools` library to fetch open questions from Metaculus tournaments, orchestrate per-type forecasting calls, and post predictions and comments. Forecast reuse (e.g., skipping previously forecasted questions) is handled by `forecasting_tools` configuration.

*   **2.2 Forecasting Orchestration (Per Question Type):** The `CommitteeForecastBot` delegates to specialized modules in `bot/` based on question type. For binary, multiple choice, and numeric questions, it calls `bot/binary.py`, `bot/multiple_choice.py`, and `bot/numeric.py` respectively. These modules perform their own integrated research and multi-agent prompting for each question.

    *   **2.2.1 Binary Question Workflow:** The end-to-end logic is implemented in `bot/binary.py` within the `get_binary_forecast` function.
        1.  **Initial Research Scoping:** Two concise scoping prompts are generated in-line: "Historical perspective on: {question}" and "Current events for: {question}". These are sent (by the scope agent) to produce targeted search directions.
        2.  **Integrated Research (AskNews + Gemini + optional GPT‑4o Search):** The scoped queries inform a single combined AskNews request (grouping all queries with OR) to use the API sparingly, plus per‑query calls to Gemini 2.5 Flash with the `google_search` tool; when configured, an additional per‑query GPT‑4o Search (preview) section is added via OpenRouter. No web scraping, Google/Google News direct APIs, or agentic search loops are used. Results are summarized and carried into the next step.
        3.  **Initial Forecast Generation:** The integrated research context is included in a structured prompt and run concurrently across the multi-model committee to generate initial predictions.
        4.  **Simulated Peer Review:** The outputs from the first forecast are rotated; for each agent, a different agent’s initial analysis is included as "peer analysis", simulating peer review.
        5.  **Final Forecast Synthesis:** The final prompts include both the research context and the peer analysis; all agents produce refined forecasts.
        6.  **Aggregation:** The final probability from each agent's output is extracted using `extract_probability_from_response_as_percentage_not_decimal`. These probabilities are combined using a weighted average (equal weights by default), and the result is clamped to the range [0.001, 0.999].

    *   **2.2.2 Multiple-Choice Question Workflow:** Implemented in `bot/multiple_choice.py:get_multiple_choice_forecast`, this mirrors the binary structure with MC-specific extraction.
        1.  It follows the same dual-perspective research scoping and integrated research process (AskNews + Gemini, optional GPT‑4o Search), without a standalone RAG phase.
        2.  The initial and final forecasts generate probability vectors across the available options.
        3.  `extract_option_probabilities_from_response` parses the probabilities from each agent’s structured output, and `normalize_probabilities` ensures each list sums to 1.0.
        4.  A weighted average across agents yields the final per-option probabilities.

    *   **2.2.3 Numeric Question Workflow:** Implemented in `bot/numeric.py`, this workflow also follows the established multi-phase pattern with numeric-specific prompts.
        1.  The research and initial analysis phases are analogous to the other types.
        2.  The key difference is the output format. The LLMs are prompted to produce a series of discrete percentile points (e.g., 10th, 25th, 50th, 75th, 90th).
        3.  The `extract_percentiles_from_response` function parses these key-value pairs from the text.
        4.  The core technical step is in `generate_continuous_cdf`, which takes these discrete points and uses `scipy.interpolate.PchipInterpolator` to create a smooth, monotonically increasing Cumulative Distribution Function (CDF) across 201 points, the format required for the Metaculus API.
        5.  The final CDFs from all agents are aggregated using a weighted average.

*   **2.3 The Multi-Agent "Committee" Model:**
    *   **2.3.1 Agent Composition (Default):** By default, the committee uses three real models via OpenRouter:
        *   `openrouter/anthropic/claude-sonnet-4`
        *   `openrouter/openai/gpt-5`
        *   `openrouter/anthropic/claude-opus-4.1`
        These can be overridden by setting `COMMITTEE_LLM_MODELS` to a comma-separated list of model identifiers. Additional tuning env vars include `COMMITTEE_LLM_TEMPERATURE`, `COMMITTEE_LLM_TIMEOUT`, and `COMMITTEE_LLM_TRIES`.
        Note: Gemini 2.5 is used in the research stack by default, not as a forecasting agent, but can be added via configuration if desired.

    *   **2.3.2 Rationale for Heterogeneity:** Mixing model families (e.g., Anthropic variants and OpenAI via OpenRouter) introduces cognitive diversity, helping to mitigate systematic reasoning flaws of any single model architecture. Teams can expand diversity further by adding additional providers/models via configuration.

    *   **2.3.3 Agent Weighting:** Forecasts are aggregated with configurable weights (`LLMAgent.weight`). The default is equal weighting across agents, with the option to tune weights empirically.

*   **2.4 The Multi-Step Reasoning Process (Prompt Chaining):**
    *   **2.4.1 Phase 1: Dual-Perspective Scoping & Research Direction:** The process begins with two parallel prompts: a historical/outside-view framing and a current/inside-view framing (constructed inline as described in 2.2.1). Their goal is to scope targeted queries for research.

    *   **2.4.2 Phase 2: Integrated Research (No RAG):**
        *   **AskNews:** Retrieve relevant, structured news context with the AskNews SDK based on the scoped queries.
        *   **Gemini Search Tool:** Use Gemini 2.5 Flash with the `google_search` tool to surface recent, high-signal sources.
        *   **Optional GPT‑4o Search (preview):** When configured, add a third research section via OpenRouter.
        *   No Google/Google News direct API usage, no agentic search loops, and no web content extraction (e.g., headless browsers or HTML scraping) are performed.
        *   The result is a concise, synthesized research context passed forward as-is; there is no separate RAG pipeline.

    *   **2.4.3 Phase 3: Initial Forecast & Simulated Peer Review:** The integrated research context is used to create the first forecast prompts. Each agent generates an independent initial forecast. The critical step is the rotation (`context_map`) of analyses between agents to simulate peer review and encourage robust synthesis.

    *   **2.4.4 Phase 4: Final Synthesis & Prediction:** The final context combines integrated research with a peer analysis from a different agent; each agent produces a refined forecast.

*   **2.5 Ensemble & Aggregation:**
    *   **2.5.1 Statistical Method:** Individual forecasts are aggregated using a weighted average. In files like `bot/binary.py`, this is implemented using `numpy.sum` on the weighted probabilities, divided by the sum of the weights. By default, three agent predictions are combined, but the committee size is configurable.

    *   **2.5.2 Risk Management & Calibration:** Binary probabilities are clipped to [0.001, 0.999] as a risk management safeguard against extreme confidence. Multiple-choice probabilities are normalized per-agent before aggregation. Numeric distributions are aggregated at the CDF level. No additional calibration transform is applied by default beyond these safeguards.

*   **2.6 Output Generation & Submission:** When running the committee engine, `CommitteeForecastBot` (in `bot/adapter.py`) returns `ReasonedPrediction` objects to the `forecasting_tools` framework, which handles converting forecasts into the correct Metaculus payloads and posting them. The adapter logs research and forecast summaries for traceability; comments are posted via `forecasting_tools` as configured.

*   **2.7 End-to-End Forecasting Flow (Summary):**
    *   **Ingest Question:** Fetch open questions and skip those already forecasted.
    *   **Route by Type:** Dispatch to `binary`, `multiple_choice`, or `numeric` pipelines in `bot/`.
    *   **Dual-Perspective Scoping:** Run the historical (outside view) and current (inside view) scoping prompts in parallel to produce targeted queries.
    *   **Integrated Research:** Use AskNews and Gemini 2.5 Flash `google_search` (plus optional GPT‑4o Search) to gather and summarize recent, forecasting‑relevant context (no RAG, scraping, or agentic browsing).
    *   **Initial Forecast (Committee):** Provide the research pack to agents and run concurrently across the committee for independent estimates.
    *   **Simulated Peer Review:** Swap initial analyses between agents (`context_map`) so each agent sees another’s reasoning plus the research pack.
    *   **Final Synthesis:** Run final prompts with research + peer‑reviewed context to get refined forecasts from all agents.
    *   **Ensemble Aggregation:** Extract structured outputs, apply per‑agent weights, aggregate, and apply safeguards (e.g., clipping/normalization as appropriate).
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
