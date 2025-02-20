def get_p_value_label(p_value):
    if round(p_value,2) < 0.001:
        p_value_label = 'P<0.001'
    elif round(p_value,2) == 0.05 and p_value<0.05:
        p_value_label = f"P={str(p_value)[:6]}"
    else:
        p_value_label = f"P={round(p_value,4)}"
    return p_value_label

def interpret_hazard_ratio(hazard_ratio, baseline_group_name, other_group_name):
    """
    Interprets the hazard ratio (HR) for two groups.

    Parameters:
    hazard_ratio (float): The hazard ratio value.
    baseline_group_name (str): Name of the baseline/reference group.
    other_group_name (str): Name of the other group being compared.

    Returns:
    str: Interpretation of the hazard ratio.
    """
    if hazard_ratio == 1:
        interpretation = (f"The hazard ratio is 1. This means there is no difference in risk between "
                          f"the {other_group_name} group and the {baseline_group_name} group.")
    elif hazard_ratio > 1:
        risk_increase = (hazard_ratio - 1) * 100
        interpretation = (f"The hazard ratio is {hazard_ratio:.2f}. This means the {other_group_name} group has "
                          f"{risk_increase:.0f}% higher risk of the event compared to the {baseline_group_name} group.")
    elif hazard_ratio < 1:
        risk_reduction = (1 - hazard_ratio) * 100
        interpretation = (f"The hazard ratio is {hazard_ratio:.2f}. This means the {other_group_name} group has "
                          f"{risk_reduction:.0f}% lower risk of the event compared to the {baseline_group_name} group.")
    else:
        interpretation = "Invalid hazard ratio value. Please provide a numeric value greater than 0."

    return interpretation
