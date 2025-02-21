import pandas as pd
from .normality import Normality

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

class DescriptiveStatistics:
    def __init__(self):
        self.normality = Normality()

    def get_statistics(self, data:pd.core.series.Series, data_type:str):
        
        self.data = data.copy()
        
        assert self.data.ndim ==1, f"Expected data with ndim 1 but got {self.data.ndim}"
        
        

        if data_type == 'num':
            return self._get_num_stats()
        elif data_type == 'cat':
            raise NotImplementedError()
        else:
            raise ValueError(f"Expected one of ['num', 'cat'] as data_type but got {data_type}")

    def _get_num_stats(self):
        desc_dict = {}
        
        desc_dict["n_samples"] = len(self.data)
        desc_dict["n_samples_valid"] = self.data.count()
        desc_dict["missing_values"] = desc_dict["n_samples"]-desc_dict["n_samples_valid"]
        desc_dict["mean"] = self.data.mean()
        desc_dict['sem'] = self.data.sem() #standard error of mean
        desc_dict["std"] = self.data.std()
        desc_dict["min"] = self.data.min()
        desc_dict["q1"] = self.data.quantile(q=0.25)
        desc_dict["median"] = self.data.median()
        desc_dict["q3"] = self.data.quantile(q=0.75)
        desc_dict["max"] = self.data.max()
        desc_dict["skew"] = self.data.skew()
        desc_dict["kurt"] = self.data.kurt()
        self.normality.set_data(data=self.data)
        normality_report = self.normality.get_normality_report_default()
        
        desc_dict["normal"] = normality_report["normal"]
        desc_dict["normality_test"] = normality_report["test_type"]
        desc_dict["normality_p"] = normality_report['p']

        return desc_dict