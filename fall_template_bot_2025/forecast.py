import logging
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from .bounds import create_upper_and_lower_bound_messages
from .prompts import binary_prompt, multiple_choice_prompt, numeric_prompt

logger = logging.getLogger(__name__)


async def forecast_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:  # noqa: ANN001
    prompt = clean_indents(
        binary_prompt(
            question_text=question.question_text,
            background=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
        )
    )
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
    binary_prediction: BinaryPrediction = await structure_output(
        reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
    )
    decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
    logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
    return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)


async def forecast_multiple_choice(
    self, question: MultipleChoiceQuestion, research: str
) -> ReasonedPrediction[PredictedOptionList]:  # noqa: ANN001
    prompt = clean_indents(
        multiple_choice_prompt(
            question_text=question.question_text,
            options=question.options,
            background=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
        )
    )
    parsing_instructions = clean_indents(
        f"""
        Make sure that all option names are one of the following:
        {question.options}
        The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
        """
    )
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
    predicted_option_list: PredictedOptionList = await structure_output(
        text_to_structure=reasoning,
        output_type=PredictedOptionList,
        model=self.get_llm("parser", "llm"),
        additional_instructions=parsing_instructions,
    )
    logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
    return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)


async def forecast_numeric(
    self, question: NumericQuestion, research: str
) -> ReasonedPrediction[NumericDistribution]:  # noqa: ANN001
    upper_bound_message, lower_bound_message = create_upper_and_lower_bound_messages(question)
    prompt = clean_indents(
        numeric_prompt(
            question_text=question.question_text,
            background=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            unit_of_measure=question.unit_of_measure,
            research=research,
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
        )
    )
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
    percentile_list: list[Percentile] = await structure_output(
        reasoning, list[Percentile], model=self.get_llm("parser", "llm")
    )
    prediction = NumericDistribution.from_question(percentile_list, question)
    logger.info(
        f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
    )
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

