import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time

class SingleClass:
    def __init__(self, df, surv_dict, km_label):
        self.kmf = KaplanMeierFitter()
        self.df = df
        self.surv_dict = surv_dict
        self.surv_df = self.df[[self.surv_dict["time"], self.surv_dict["event"]]].dropna()
        self.km_label = km_label
        self.max_followup_time = self.surv_df[self.surv_dict["time"]].max()
        self.kmf.fit(
            durations=self.surv_df[self.surv_dict["time"]],
            event_observed=self.surv_df[self.surv_dict["event"]],
            label=self.km_label,
        )
        self.descriptives = {}
        self.descriptives['median_survival'] = self._get_median_survival()
        self.descriptives['median_follow_up'] = self._get_median_follow_up()
        self.descriptives[f"Surv Probability"] = {}

    def _get_median_survival(self):
        median_survival = self.kmf.median_survival_time_
        median_ci = median_survival_times(self.kmf.confidence_interval_)

        ci_lower = median_ci[f'{self.kmf.label}_lower_0.95'][0.5]
        ci_upper = median_ci[f'{self.kmf.label}_upper_0.95'][0.5]
        
        return f"{round(median_survival,2)} (95% CI, {ci_lower:.2f} - {ci_upper:.2f})"

    def _get_survival_probability(self, time_point):
        ci_df = self.kmf.confidence_interval_
        time_points = ci_df.index
        
        lower_interpolator = interp1d(time_points, ci_df[f'{self.kmf.label}_lower_0.95'], fill_value="extrapolate")
        upper_interpolator = interp1d(time_points, ci_df[f'{self.kmf.label}_upper_0.95'], fill_value="extrapolate")
        
        if time_point < time_points[0] or time_point > time_points[-1]:
            #"Time point is out of the range of observed data."
            survival_prob = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        
        else:
            survival_prob = self.kmf.predict(time_point)
        
            ci_lower = lower_interpolator(time_point).item()
            ci_upper = upper_interpolator(time_point).item()
        
        self.descriptives[f"Surv Probability"][time_point] =  f"{round(survival_prob,2)} (95% CI, {ci_lower:.2f} - {ci_upper:.2f})"

    def _get_median_follow_up(self):
        label = f"{self.km_label} Follow-Up"
        followup_kmf = KaplanMeierFitter()
        
        followup_kmf.fit(
            durations=self.surv_df[self.surv_dict["time"]],
            event_observed=~self.df[self.surv_dict["event"]],
            label=label,
        )
        
        median_followup = followup_kmf.median_survival_time_
        
        median_ci = median_survival_times(followup_kmf.confidence_interval_)
        
        ci_lower = median_ci[f'{label}_lower_0.95'][0.5]
        ci_upper = median_ci[f'{label}_upper_0.95'][0.5]
    
        return f"{median_followup:.2f} (95% CI, {ci_lower:.2f} - {ci_upper:.2f})"

    def get_rmst(
        self, 
        restricted_time = None, 
        bootstrap_ci=False, 
        n_bootstraps=1000,
        show_progress=True,
        random_seed = None,
    ):
        if restricted_time is None:
            restricted_time = self.max_followup_time
    
        rmst_dict = {}
        rmst_dict['restricted_time'] = restricted_time
        
        rmst = restricted_mean_survival_time(self.kmf, t=restricted_time)
        rmst_dict['rmst'] = rmst
            
        if bootstrap_ci:
            rmst_samples = []
            if show_progress:
                iterator = tqdm(range(n_bootstraps))
            else:
                iterator = range(n_bootstraps)
                
            for idx in iterator:
                if random_seed is not None:
                    random_state = random_seed+idx
                else:
                    random_seed=None
                
                bootstrap_sample = self.surv_df.sample(n=len(self.surv_df), replace=True, random_state=random_state)
                kmf_bootstrap = KaplanMeierFitter()
                kmf_bootstrap.fit(bootstrap_sample[self.surv_dict['time']], event_observed=bootstrap_sample[self.surv_dict['event']])
                rmst_bootstrap = restricted_mean_survival_time(kmf_bootstrap, t=restricted_time)
                rmst_samples.append(rmst_bootstrap)
                
            rmst_dict['95% ci'] = np.percentile(rmst_samples, [2.5, 97.5])
            rmst_dict['mean_rmst'] = np.mean(rmst_samples)
            rmst_dict['bootstrap_label'] = f"{rmst_dict['mean_rmst']:.2f} (95% CI, {rmst_dict['95% ci'][0]:.2f} - {rmst_dict['95% ci'][1]:.2f})"            
            rmst_dict['label'] = f"{rmst_dict['rmst']:.2f} (95% CI, {rmst_dict['95% ci'][0]:.2f} - {rmst_dict['95% ci'][1]:.2f})" 
            
        self.descriptives['rmst'] = rmst_dict
        
    def plot_km_curve(
        self, 
        table_title = 'Risk Table', 
        xlabel = 'Time (Months)', 
        ylabel = 'Survival Probability', 
        plot = True, 
        title = None, 
        savepath = None, 
        plot_whole_y_axis = True,
        add_risk_table = True
    ):
        self.kmf.plot_survival_function(ci_show=False, legend=False)
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if title is not None:
            plt.title(title)
            
        if plot_whole_y_axis:
            plt.ylim(0, 1) 
        
        plt.grid(True)
    
        if add_risk_table:
            add_at_risk_counts(self.kmf, labels=[table_title])
    
        plt.tight_layout()
    
        if savepath is not None:
            plt.savefig(savepath)
    
        if plot:
            plt.show()
        else:
            plt.close()
