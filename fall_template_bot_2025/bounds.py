from forecasting_tools import NumericQuestion


def create_upper_and_lower_bound_messages(question: NumericQuestion) -> tuple[str, str]:
    if question.nominal_upper_bound is not None:
        upper_bound_number = question.nominal_upper_bound
    else:
        upper_bound_number = question.upper_bound
    if question.nominal_lower_bound is not None:
        lower_bound_number = question.nominal_lower_bound
    else:
        lower_bound_number = question.lower_bound

    if question.open_upper_bound:
        upper_bound_message = (
            f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        )
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound_number}."

    if question.open_lower_bound:
        lower_bound_message = (
            f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        )
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound_number}."
    return upper_bound_message, lower_bound_message

