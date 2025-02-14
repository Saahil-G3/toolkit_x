import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

class MinimumPValue:
    def __init__(self, project_id, df):
        self.project_id = project_id
        self.df = df
        self._init_dirs()

    def _init_dirs(self):
        self.dirs = {}
        self.dirs['mpv'] = Path(f"analytics/{self.project_id}/survival/mpv")
        self.dirs['mpv'].mkdir(exist_ok=True, parents=True)

        self.paths = {}

    def run_mpv(self, num_col_name, surv_dict, surv_label, show_progress =True, continuous_range=True):
        
        self.paths[f'mpv_{num_col_name}'] = self.dirs['mpv']/f"{num_col_name} ({surv_label}).csv"
        surv_df = self.df[[surv_dict["time"], surv_dict["event"], num_col_name]].dropna()
    
        if continuous_range:
            num_col = surv_df[num_col_name].dropna().unique()
            cutoffs = np.arange(num_col.min(), num_col.max(),0.5)
        else:
            cutoffs = surv_df[num_col_name].dropna().unique()
            cutoffs.sort()
        
        if show_progress:
            iterator = tqdm(cutoffs)
        else:
            iterator = cutoffs
        minimum_p_val = []
        
        FIELDNAMES = ['p_value', 'cutoff', 'group1_counts', 'group2_counts', 'group1_med_surv', 'group2_med_surv', 'significant']
        with open(self.paths[f'mpv_{num_col_name}'], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        
        for cutoff in iterator:
            group_1 = surv_df[surv_df[num_col_name] <= cutoff]
            group_2 = surv_df[surv_df[num_col_name] > cutoff]
        
            minimum_p_val_dict = {}
            minimum_p_val_dict['cutoff']= cutoff
            minimum_p_val_dict['group1_counts']= group_1.shape[0]
            minimum_p_val_dict['group2_counts']= group_2.shape[0]
            
            if group_1.empty or group_2.empty:
                pass
                
            else:
                results = logrank_test(
                    group_1[surv_dict["time"]], group_2[surv_dict["time"]],
                    event_observed_A=group_1[surv_dict["event"]],
                    event_observed_B=group_2[surv_dict["event"]]
                )
                kmf1 = KaplanMeierFitter()
                kmf2 = KaplanMeierFitter()
        
                kmf1.fit(group_1[surv_dict["time"]], event_observed=group_1[surv_dict["event"]])
                kmf2.fit(group_2[surv_dict["time"]], event_observed=group_2[surv_dict["event"]])
        
                median_survival_group_1 = kmf1.median_survival_time_
                median_survival_group_2 = kmf2.median_survival_time_
        
                minimum_p_val_dict['p_value']= results.p_value
                minimum_p_val_dict['group1_med_surv']= median_survival_group_1
                minimum_p_val_dict['group2_med_surv']= median_survival_group_2
                minimum_p_val_dict['significant'] = results.p_value<0.05
        
        
            with open(self.paths[f'mpv_{num_col_name}'], "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(minimum_p_val_dict)
                
        self.plot_mpv(num_col_name=num_col_name, continuous_range=continuous_range, plot=False, save=True)

    def get_cutoff_near_median(self, num_col_name):
        mpv_df = pd.read_csv(self.paths[f'mpv_{num_col_name}'])
        median = self.df[[num_col_name]].median().item()
        filtered_cutoffs = mpv_df[mpv_df['significant']]['cutoff']
        distances = abs(filtered_cutoffs-median)
        min_distance_index = distances.idxmin()
        closest_value = filtered_cutoffs.loc[min_distance_index]
    
        return closest_value

    def plot_mpv_with_counts(self,num_col_name, show_lines = True, plot = True, color='purple', title = None, save=False, continuous_range=False):
        
        mpv_df = pd.read_csv(self.paths[f'mpv_{num_col_name}'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        ax1.plot(mpv_df['cutoff'], mpv_df['group1_med_surv'], label='Group 1 Median Survival', color='red', linestyle='-', marker='_')
        ax1.plot(mpv_df['cutoff'], mpv_df['group2_med_surv'], label='Group 2 Median Survival', color='green', linestyle='-', marker='_')
        
        filtered_data = mpv_df[mpv_df['significant']]
        points, = ax1.plot(filtered_data['cutoff'], [0.05]*len(filtered_data), 'o', label='Significant P-Values (p < 0.05)', markersize=2, color= 'black')
        # ax1_twin = ax1.twinx()
        # points, = ax1_twin.plot(mpv_df['cutoff'], mpv_df['p_value'], 'o', label='P-Values', markersize=2,color= 'black')
        # ax1_twin.axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='p = 0.05')
        #ax1_twin.legend()
        
        if show_lines:
            for cutoff in filtered_data['cutoff']:
                ax1.axvline(x=cutoff, color=color, linestyle='--', linewidth=0.2)
        
        ax1.set_ylabel('Median Survival')
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax1_twin.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        ax2.plot(mpv_df['cutoff'], mpv_df['group1_counts'], label='Group 1 Counts', color='orange', linestyle='-', marker='o',markersize=8, alpha = 0.5)
        ax2.plot(mpv_df['cutoff'], mpv_df['group2_counts'], label='Group 2 Counts', color='blue', linestyle='-', marker='o',markersize=8, alpha = 0.5)
        ax2.set_xlabel('Cutoff')
        ax2.set_ylabel('Counts')
        ax2.legend()
        ax2.grid(True)
        
        if title is None:
            title = f"Median Survival and Counts vs. Cutoff ({num_col_name.replace('_', ' ').capitalize()})"
        
        plt.suptitle(title)
        
        if save:
            plt.savefig(self.paths[f'mpv_{num_col_name}'].parent/ f"{self.paths[f'mpv_{num_col_name}'].stem} (Counts).jpg")
        
        if plot:
            plt.show()
        else:
            plt.close()
    
    def plot_mpv(self,num_col_name, show_lines = True, plot = True, color='purple', title = None, save=False, continuous_range=False):
        mpv_df = pd.read_csv(self.paths[f'mpv_{num_col_name}'])
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot for group1_med_surv and group2_med_surv on primary y-axis
        line1, = ax1.plot(mpv_df['cutoff'], mpv_df['group1_med_surv'], label='Group 1 Median Survival', color='red', linestyle='-', marker='_')
        line2, = ax1.plot(mpv_df['cutoff'], mpv_df['group2_med_surv'], label='Group 2 Median Survival', color='green', linestyle='-', marker='_')
        
        ax2 = ax1.twinx()
        filtered_data = mpv_df[mpv_df['significant']]
        
        points, = ax2.plot(filtered_data['cutoff'], [0.05]*len(filtered_data), 'o', label='Significant P-Values (p < 0.05)', markersize=2,color= 'black')
        # points, = ax2.plot(mpv_df['cutoff'], mpv_df['p_value'], 'o', label='P-Values', markersize=2,color= 'black')

        ax2.axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='p = 0.05')
        
        if show_lines:
            # Plot vertical lines for each significant p-value
            for cutoff in filtered_data['cutoff']:
                ax1.axvline(x=cutoff, color=color, linestyle='--', linewidth=0.2)
        # Combine legends from both axes
        handles, labels = [], []
        for ax in [ax1, ax2]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        ax2.set_ylim(0, max(mpv_df['p_value'].max(), 0.05))  # Ensure range covers significant p-values
        ax2.set_ylabel('P-Values')  # No label needed for secondary y-axis
        ax1.set_xlabel('Cutoff')
        ax1.set_ylabel('Median Survival')
        ax1.legend(handles=handles, labels=labels, loc='upper right')
        ax1.grid(True)
        if title is None:
                title = f"Median Survival and Counts vs. Cutoff ({num_col_name.replace('_', ' ').capitalize()})"
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(self.paths[f'mpv_{num_col_name}'].parent/ f"{self.paths[f'mpv_{num_col_name}'].stem}.jpg")
        if plot:
            plt.show()
        else:
            plt.close()
