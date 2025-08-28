from datetime import datetime


def research_prompt(question_text: str, resolution_criteria: str, fine_print: str) -> str:
    return (
        "You are an assistant to a superforecaster.\n"
        "The superforecaster will give you a question they intend to forecast on.\n"
        "To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
        "You do not produce forecasts yourself.\n\n"
        f"Question:\n{question_text}\n\n"
        "This question's outcome will be determined by the specific criteria below:\n"
        f"{resolution_criteria}\n\n"
        f"{fine_print}\n"
    )


def binary_prompt(question_text: str, background: str, resolution_criteria: str, fine_print: str, research: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{research}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def multiple_choice_prompt(
    question_text: str,
    options: list[str],
    background: str,
    resolution_criteria: str,
    fine_print: str,
    research: str,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{research}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def numeric_prompt(
    question_text: str,
    background: str,
    resolution_criteria: str,
    fine_print: str,
    unit_of_measure: str | None,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    units = unit_of_measure if unit_of_measure else "Not stated (please infer this)"
    return f"""
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}

Your research assistant says:
{research}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

The last thing you write is your final answer as:
"\nPercentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX\n"
"""

