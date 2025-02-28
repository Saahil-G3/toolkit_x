import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

from .single_class import SingleClass

from toolkit.analytics.descriptives import get_p_value_label, interpret_hazard_ratio

class BinaryClass:
    def __init__(
        self,
        df0,
        df1,
        surv_dict,
        group_labels: dict,
        alpha=0.05,
    ):
        self.df0 = df0
        self.df1 = df1
        self.surv_dict = surv_dict
        self.group_labels = group_labels
        self.alpha = alpha

    def run(self, baseline_group=0, censoring="right"):
        self._set_comparison_dict()
        self.comparison_dict['hazard_dict'] = self.fit_cph(baseline_group=baseline_group, censoring=censoring)

    def _set_comparison_dict(self):
        self.km0 = SingleClass(df=self.df0, surv_dict=self.surv_dict)
        self.km0.run()
        
        self.km1 = SingleClass(df=self.df1, surv_dict=self.surv_dict)
        self.km1.run()

        results = logrank_test(
            self.df0[self.surv_dict["time"]],
            self.df1[self.surv_dict["time"]],
            event_observed_A=self.df0[self.surv_dict["event"]],
            event_observed_B=self.df1[self.surv_dict["event"]],
        )

        comparison_dict = {}
        comparison_dict["p_value"] = results.p_value
        comparison_dict["group0_counts"] = self.df0.shape[0]
        comparison_dict["group1_counts"] = self.df1.shape[0]
        comparison_dict["group0_median_survival"] = self.km0.descriptives[
            "median_survival"
        ]
        comparison_dict["group1_median_survival"] = self.km1.descriptives[
            "median_survival"
        ]
        comparison_dict["group0_median_follow_up"] = self.km0.descriptives[
            "median_follow_up"
        ]
        comparison_dict["group1_median_follow_up"] = self.km1.descriptives[
            "median_follow_up"
        ]


        self.comparison_dict = comparison_dict

    def fit_cph(self, baseline_group=0, censoring="right", get_hazard_dict=True):
        self.baseline_group = baseline_group

        surv_df0 = self.df0[[self.surv_dict["time"], self.surv_dict["event"]]].copy()
        surv_df0["group"] = self.group_labels[0]

        surv_df1 = self.df1[[self.surv_dict["time"], self.surv_dict["event"]]].copy()
        surv_df1["group"] = self.group_labels[1]

        surv_df = pd.concat([surv_df0, surv_df1], ignore_index=True)

        surv_df["group"] = surv_df["group"].astype("category")
        surv_df = pd.get_dummies(surv_df, columns=["group"], prefix_sep="", prefix="")

        surv_df = surv_df.drop(columns=[self.group_labels[self.baseline_group]])

        self.cph = CoxPHFitter(alpha=self.alpha)

        if censoring == "right":
            self.cph.fit_right_censoring(
                surv_df,
                duration_col=self.surv_dict["time"],
                event_col=self.surv_dict["event"],
            )
        else:
            raise NotImplemented(f"Censoring type {censoring} not implemented.")

        if get_hazard_dict:
            return self._get_hazard_dict()

    def _get_hazard_dict(self):
        if not hasattr(self, "cph"):
            raise ValueError("Please run fit_cph first")

        summary = self.cph.summary
        row_name = summary.index.tolist()
        assert len(row_name) == 1, "Multiple variables found in binary surv"
        row_name = row_name[0]
        row = summary.loc[row_name]
        hr = row["exp(coef)"]
        hr_ci = [row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]]

        hazard_dict = {}
        hazard_dict["baseline_group"] = self.group_labels[self.baseline_group]
        hazard_dict["hr"] = hr
        hazard_dict["hr_ci"] = hr_ci
        hazard_dict["p_value"] = row["p"]
        hazard_dict["label"] = (
            f"Hazard ratio, {hazard_dict['hr']:.2f} (95% CI, {hazard_dict['hr_ci'][0]:.2f} - {hazard_dict['hr_ci'][1]:.2f})"
        )
        hazard_dict["p_value_label"] = get_p_value_label(hazard_dict["p_value"])
        hazard_dict["interpretation"] = interpret_hazard_ratio(
            hazard_ratio=hazard_dict["hr"],
            baseline_group_name=self.group_labels[self.baseline_group],
            other_group_name=self.group_labels[1 - self.baseline_group],
        )

        return hazard_dict

    def plot_km_curves(
        self,
        hazard_dict=None,
        print_hazard_stats=True,
        plot=True,
        title=None,
        savepath=None,
        plot_grid=True,
        x_axis_range=None,
        add_risk_table=True,
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        self.km0.kmf.plot_survival_function(
            ax=ax, label=f"{self.group_labels[0]}", ci_show=False
        )
        self.km1.kmf.plot_survival_function(
            ax=ax, label=f"{self.group_labels[1]}", ci_show=False
        )

        legend = ax.legend()  # Get the generated legend object
        legend.set_title("Survival Groups")  # Set a title for the legend
        plt.xlabel("Time (months)")
        plt.ylabel("Survival Probability")
        if title is not None:
            plt.title(title)
        ax.set_ylim(0, 1)

        if x_axis_range is None:
            max_time = max(
                self.km0.kmf.event_table.index[-1], self.km1.kmf.event_table.index[-1]
            )
            x_axis_range = range(0, int(max_time) + 1, 12)

        plt.xticks(x_axis_range)
        plt.axhline(y=0.5, color="red", linestyle="--")

        if add_risk_table:
            add_at_risk_counts(
                self.km0.kmf,
                self.km1.kmf,
                ax=ax,
                labels=[self.group_labels[0], self.group_labels[1]],
            )
            plt.subplots_adjust(
                left=0.2, bottom=0.3
            )  # Adjust plot to make room for at-risk table

        if print_hazard_stats:
            if hazard_dict is None:
                hazard_dict = self.comparison_dict['hazard_dict']
                
            plt.text(
                x=0.05,
                y=0.15,
                s=hazard_dict["label"],
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
            plt.text(
                x=0.05,
                y=0.10,
                s=hazard_dict["p_value_label"],
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
                
        if plot_grid:
            plt.grid(True)
            
        if savepath is not None:
            plt.savefig(savepath)

        if plot:
            plt.show()
        else:
            plt.close()
