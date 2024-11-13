import pandas as pd
import matplotlib.pyplot as plt


from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
from lifelines import KaplanMeierFitter, CoxPHFitter

"""
Note:
    any variable named with lowerCamelCase convention is a string variable and represents a column name in a dataframe. 
"""

def get_median_survival(km_object:KaplanMeierFitter, label=None):
    '''
    Computes the median survival time and its 95% confidence interval from a Kaplan-Meier survival object.

    Parameters:
    ----------
    km_object : KaplanMeierFitter
        A fitted KaplanMeierFitter object from the lifelines library.
    
    label : str, optional
        An optional label for the median survival output. If not provided, the label from the `km_object` will be used.

    Returns:
    --------
    med_surv_dict : dict
        A dictionary containing:
        - 'variable': The label of the variable (either the provided label or the one from the KaplanMeierFitter object).
        - 'med_sruv': The median survival time along with its 95% confidence interval in the format `median (95% CI, lower - upper)`.

    Example:
    --------
    kmf = KaplanMeierFitter()
    kmf.fit(durations=[5, 6, 6, 2.5], event_observed=[1, 0, 1, 1])
    
    # Get the median survival time with default label
    result = get_median_survival(kmf)
    
    # Get the median survival time with custom label
    result = get_median_survival(kmf, label='Treatment A')
    
    # Access the results
    print(result['variable'])  # 'Treatment A' or the label from km_object
    print(result['med_sruv'])  # 'median_survival_time (95% CI, lower - upper)'
    '''
    
    median_survival = km_object.median_survival_time_
    median_confidence_interval = median_survival_times(km_object.confidence_interval_)


    median_ci_lower = median_confidence_interval[f'{km_object.label}_lower_0.95'][0.5]
    median_ci_upper = median_confidence_interval[f'{km_object.label}_upper_0.95'][0.5]
    
    if label:
        med_surv_dict = {'variable':label,
                         'med_surv':f"{round(median_survival,2)} (95% CI, {median_ci_lower:.2f} - {median_ci_upper:.2f})"
                        }
    else:
        med_surv_dict = {'variable':km_object.label,
                         'med_sruv':f"{round(median_survival,2)} (95% CI, {median_ci_lower:.2f} - {median_ci_upper:.2f})"
                        }
    
    return med_surv_dict


def get_km_binary(group_0:pd.DataFrame, group_1:pd.DataFrame, t2e_var):
    
    '''
    Computes Kaplan-Meier survival curves for two groups and performs a log-rank test.

    Parameters:
    ----------
    group_0 : pd.DataFrame
        The DataFrame representing the first group. Must contain time-to-event and event indicator columns.
    
    group_1 : pd.DataFrame
        The DataFrame representing the second group. Must also contain time-to-event and event indicator columns.
    
    t2e_var : dict
        A dictionary with two keys:
        - 'time': Name of the column representing the time-to-event variable.
        - 'event': Name of the column representing the event indicator variable (1 if event occurred, 0 if censored).

    Returns:
    --------
    results : dict
        A dictionary containing:
        - 'km_objects': A list of KaplanMeierFitter objects for the two groups, where the first entry is for `group_0` and the second for `group_1`.
        - 'logrank_pval': The p-value from the log-rank test comparing the survival curves of the two groups.
    
    Example:
    --------
    group_0 = pd.DataFrame({'time': [1, 2, 3], 'event': [1, 0, 1]})
    group_1 = pd.DataFrame({'time': [2, 3, 4], 'event': [0, 1, 1]})
    t2e_var = {'time': 'time', 'event': 'event'}
    
    result = get_km_binary(group_0, group_1, t2e_var)
    
    # Access Kaplan-Meier fitters and p-value
    kmf0, kmf1 = result['km_objects']
    p_value = result['logrank_pval']
    '''
    
    kmf0 = KaplanMeierFitter()
    kmf1 = KaplanMeierFitter()

    logrank_result = logrank_test(
        group_0[t2e_var['time']], group_1[t2e_var['time']],
        
        event_observed_A=group_0[t2e_var['event']],
        event_observed_B=group_1[t2e_var['event']]
        )

    kmf0.fit(group_0[t2e_var['time']], event_observed=group_0[t2e_var['event']])
    kmf1.fit(group_1[t2e_var['time']], event_observed=group_1[t2e_var['event']])

    results = {}
    results['km_objects'] = [kmf0, kmf1]
    results['logrank_pval'] = logrank_result.p_value

    return results

def plot_km_binary(km_binary_dict,
                   plot_title,
                   labels,
                   figsize = (8, 6),
                   hazard_ratio=None,
                   hazard_ratio_ci=None,
                   xlab = 'Time (months)', 
                   ylab = 'Survival Probability',
                   legend_title = 'Survival Groups',
                   save_path = None,
                   plot = True
                  ):
    '''
    Plot Kaplan-Meier survival curves for two groups and display at-risk counts, hazard ratio, and p-value.
    
    Parameters:
    - km_binary_dict (dict): A dictionary containing the Kaplan-Meier fitter objects and log-rank p-value.
        - 'km_objects' (list): List of two Kaplan-Meier fitter objects (kmf0 and kmf1).
        - 'logrank_pval' (float): P-value from the log-rank test comparing the two groups.
    - plot_title (str): The title of the plot.
    - labels (list): A list of labels for the two survival groups.
    - hazard_ratio (float, optional): The hazard ratio to be displayed on the plot. Defaults to None.
    - xlab (str, optional): The label for the x-axis. Defaults to 'Time (months)'.
    - ylab (str, optional): The label for the y-axis. Defaults to 'Survival Probability'.
    - legend_title (str, optional): The title for the legend. Defaults to 'Survival Groups'.
    - save_path (str, optional): File path to save the plot as an image. If None, the plot will not be saved.
    - plot (bool, optional): If True, display the plot. If False, do not show the plot but prepare it for saving. Defaults to True.
    
    Returns:
    None: Displays the Kaplan-Meier survival curves plot and/or saves it as an image based on the parameters.
    '''


    kmf0, kmf1 = km_binary_dict['km_objects']
    p_value = km_binary_dict['logrank_pval']

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(0, 1)
    
    kmf0.plot(ax=ax, at_risk_counts=False, ci_show=False, label=labels[0])
    kmf1.plot(ax=ax, at_risk_counts=False, ci_show=False, label=labels[1])
    
    legend = ax.legend()  # Get the generated legend object
    legend.set_title(legend_title)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(plot_title)
    
    add_at_risk_counts(kmf0, kmf1, ax=ax,  labels=labels)

    if hazard_ratio:
        if hazard_ratio_ci:
            plt.text(0.05, 0.15,
                     f'Hazard ratio, {round(hazard_ratio,2)} (95% CI, {round(hazard_ratio_ci[0],2)} - {round(hazard_ratio_ci[1],2)})',
                     transform=ax.transAxes, fontsize=9, verticalalignment='top'
                    )
        else:
            plt.text(0.05, 0.15,
                     f'Hazard ratio, {round(hazard_ratio,2)}',
                     transform=ax.transAxes, fontsize=9, verticalalignment='top'
                    )
            
    if round(p_value,2) < 0.001:
        p_value_to_print = 'P<0.001'
    elif round(p_value,2) == 0.05 and p_value<0.05:
        p_value_to_print = f"P={str(p_value)[:6]}"
    else:
        p_value_to_print = f"P={round(p_value,4)}"
    plt.text(0.05, 0.10,f'{p_value_to_print}', transform=ax.transAxes, fontsize=9, verticalalignment='top')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    if plot:
        plt.show()
    else:
        plt.close()

def get_hazard_binary(surv, t2e_var, categoriesVarName):
    '''
    Calculate the hazard ratio and its confidence interval for binary categories using the Cox proportional hazards model.
    
    Parameters:
    - surv (DataFrame): A pandas DataFrame containing survival data, including time and event indicators.
    - t2e_var (dict): A dictionary specifying the column names for time and event.
        - 'time' (str): The name of the column representing the duration until the event or censoring.
        - 'event' (str): The name of the column indicating whether the event of interest occurred (1) or was censored (0).
    - categories_ (str): The category for which to retrieve the hazard ratio from the Cox model.
    
    Returns:
    - hazard_ratio (float): The calculated hazard ratio for the specified category.
    - hazard_ratio_ci (tuple): A tuple containing the lower and upper bounds of the 95% confidence interval for the hazard ratio.
    '''
    cph = CoxPHFitter()
    cph.fit(surv, duration_col=t2e_var['time'], event_col=t2e_var['event'])
    
    hazard_ratio = cph.summary.loc[categoriesVarName, 'exp(coef)']
    hazard_ratio_ci = (cph.summary.loc[categoriesVarName]['exp(coef) lower 95%'],
                       cph.summary.loc[categoriesVarName]['exp(coef) upper 95%']
                       )
    return hazard_ratio, hazard_ratio_ci
