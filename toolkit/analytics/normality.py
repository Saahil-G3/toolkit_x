import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors

from scipy.stats import probplot
from scipy.stats import shapiro, kstest, normaltest, jarque_bera, anderson, norm


class Normality:
    
    def __init__(self):
        self._implemented_tests = [
            "shapiro",
            "ks",
            "anderson",
            "dagostino",
            "jb",
            "lilliefors",
        ]

    def set_data(self, data, alpha=0.05, ddof=1):
        assert (
            len(data.shape) == 1
        ), "Input data must be a 1-dimensional array (single row)."

        self._alpha = alpha
        self._ddof = ddof  # Set 0 for polulation

        self._data = data.copy().dropna()
        self._mean = np.mean(self._data)
        self._std = np.std(self._data, ddof=self._ddof)

    def get_normality_report_full(self):
        self._normality_report = []

        self._normality_report.append(self._shapiro())
        self._normality_report.append(self._ks())
        self._normality_report.append(self._anderson())
        self._normality_report.append(self._dagostino())
        self._normality_report.append(self._jb())
        self._normality_report.append(self._lilliefors())

        return self._normality_report

    def get_normality_report_default(self):
        """
        Generates a normality report for the dataset using either the Shapiro-Wilk test or the Kolmogorov-Smirnov test,
        depending on the size of the dataset.
    
        This method automatically selects the appropriate normality test based on the sample size:
        - If the dataset has 50 or fewer observations, the Shapiro-Wilk test is used.
        - If the dataset has more than 50 observations, the Kolmogorov-Smirnov test is used.
    
        Returns:
            dict: A dictionary containing the results of the
        """
        if len(self._data) <=50:
            return self._shapiro()
        else:
            return self._ks()

    def get_normality_report(self, test_type):
        """
        Generates a normality report for the data using the specified test type.

        Args:
            test_type (str): The type of normality test to perform. Must be one of the following:
                - 'shapiro': Shapiro-Wilk test
                - 'ks': Kolmogorov-Smirnov test
                - 'anderson': Anderson-Darling test
                - 'dagostino': D'Agostino's K-squared test
                - 'jb': Jarque-Bera test
                - 'lilliefors': Lilliefors test

        Raises:
            ValueError: If the specified `test_type` is not one of the implemented tests.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the selected normality test.

        Example:
            >>> report = instance.get_normality_report('shapiro')
            >>> print(report)
        """

        if test_type not in self._implemented_tests:
            raise ValueError(
                f"Test not implemented, must be one of {self._implemented_tests}"
            )

        if test_type == "shapiro":
            return self._shapiro()
        elif test_type == "ks":
            return self._ks()
        elif test_type == "anderson":
            return self._anderson()
        elif test_type == "dagostino":
            return self._dagostino()
        elif test_type == "jb":
            return self._jb()
        elif test_type == "lilliefors":
            return self._lilliefors()

    def _shapiro(self):

        shapiro_stat, shapiro_p = shapiro(self._data)
        shapiro_normal = shapiro_p > self._alpha

        temp_dict = {}
        temp_dict["test_type"] = "Shapiro-Wilk"
        temp_dict["stat"] = shapiro_stat
        temp_dict["p"] = shapiro_p
        temp_dict["normal"] = shapiro_normal
        temp_dict["recommended_sample_size"] = "<=50"

        return temp_dict

    def _ks(self):

        ks_stat, ks_p = kstest(self._data, "norm", args=(self._mean, self._std))
        ks_normal = ks_p > self._alpha

        temp_dict = {}
        temp_dict["test_type"] = "Kolmogorov-Smirnov"
        temp_dict["stat"] = ks_stat
        temp_dict["p"] = ks_p
        temp_dict["normal"] = ks_normal
        temp_dict["recommended_sample_size"] = ">50"

        return temp_dict

    def _anderson(self):
        anderson_result = anderson(self._data, dist="norm")
        anderson_stat = anderson_result.statistic
        significance_index = next(
            i
            for i, val in enumerate(anderson_result.significance_level)
            if val >= self._alpha
        )
        anderson_critical = anderson_result.critical_values[significance_index]
        anderson_normal = anderson_stat < anderson_critical

        temp_dict = {}
        temp_dict["test_type"] = "Anderson-Darling"
        temp_dict["stat"] = anderson_stat
        temp_dict["p"] = "Based On Anderson Critical Value"
        temp_dict["normal"] = anderson_normal
        temp_dict["recommended_sample_size"] = ">20"

        return temp_dict

    def _dagostino(self):
        dagostino_stat, dagostino_p = normaltest(self._data)
        dagostino_normal = dagostino_p > self._alpha

        temp_dict = {}
        temp_dict["test_type"] = "D'Agostino-Pearson"
        temp_dict["stat"] = dagostino_stat
        temp_dict["p"] = dagostino_p
        temp_dict["normal"] = dagostino_normal
        temp_dict["recommended_sample_size"] = ">50"

        return temp_dict

    def _jb(self):
        jb_stat, jb_p = jarque_bera(self._data)
        jb_normal = jb_p > self._alpha

        temp_dict = {}
        temp_dict = {}
        temp_dict["test_type"] = "Jarque-Bera"
        temp_dict["stat"] = jb_stat
        temp_dict["p"] = jb_p
        temp_dict["normal"] = jb_normal
        temp_dict["recommended_sample_size"] = "--"

        return temp_dict

    def _lilliefors(self):
        lilliefors_stat, lilliefors_p = lilliefors(self._data)
        lilliefors_normal = lilliefors_p > self._alpha

        temp_dict = {}
        temp_dict = {}
        temp_dict["test_type"] = "Lilliefors"
        temp_dict["stat"] = lilliefors_stat
        temp_dict["p"] = lilliefors_p
        temp_dict["normal"] = lilliefors_normal
        temp_dict["recommended_sample_size"] = "--"

        return temp_dict

    @staticmethod
    def qq_plot(data, save_path=None, plot=True, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        probplot(data, dist="norm", plot=plt)

        if save_path is not None:
            plt.savefig(save_path)

        if plot:
            plt.show()
        else:
            plt.close()
