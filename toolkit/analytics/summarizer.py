import numpy as np
import pandas as pd
from pathlib import Path

from .cleaner import Cleaner
from .normality import Normality

class Summarizer(Cleaner):
    def __init__(self, run_id=None):
        super().__init__(run_id=run_id)
        self._dirs["summarizer_root"] = Path(f"analytics/summarizer")


    def configure_summarizer_run(self, branch_name=None, df_name=None, make_df_dir=True):
        
        self._set_common_configuration(branch_name=branch_name, df_name=df_name, make_df_dir=make_df_dir)
        self._set_common_df()

        self._col_report = pd.ExcelFile(self._paths["col_report_clean"])
        self._overview = self._col_report.parse("overview", index_col=None)
        self.df = pd.read_csv(self._paths["df_clean"])

        if self._overview["cat_col_names"].dropna().empty:
            self.cat_col_names = None
        else:
            self.cat_col_names = self._overview["cat_col_names"].dropna().to_list()

        if self._overview["num_col_names"].dropna().empty:
            self.num_col_names = None
        else:
            self.num_col_names = self._overview["num_col_names"].dropna().to_list()

        self.identifiers = self._overview["identifiers"].dropna().to_list()
        
        self._initialize_summarizer_paths(make_df_dir=make_df_dir)


    def _initialize_summarizer_paths(self, make_df_dir=True):
        
        self._dirs["summarizer_results"] = self._dirs["summarizer_root"]/f"{self.run_id}"
        self._dirs["missing_value_report"] = self._dirs["summarizer_results"]/f"missing_value_report"
        self._dirs["summary_report"] = self._dirs["summarizer_results"]/f"summary_report"
        self._dirs["normality_report"] = self._dirs["summarizer_results"]/f"normality_report"

        if make_df_dir:
            self._dirs["summarizer_results"].mkdir(exist_ok=True, parents=True)
            self._dirs["missing_value_report"].mkdir(exist_ok=True, parents=True)
            self._dirs["summary_report"].mkdir(exist_ok=True, parents=True)
            self._dirs["normality_report"].mkdir(exist_ok=True, parents=True)

        self._paths["missing_value_report"] = (
            self._dirs["missing_value_report"] / f"missing_value_report_{self.df_name}.xlsx"
        )

        self._paths["excel_summary_report"] = (
            self._dirs["summary_report"] / f"summary_report_{self.df_name}.xlsx"
        )

        self._paths["normality_report"] = (
            self._dirs["normality_report"] / f"normality_report_{self.df_name}.csv"
        )

    def create_missing_report(self):
        
        self._set_missing_value_report()
        self._set_missing_value_identifier_report()

        with pd.ExcelWriter(self._paths["missing_value_report"]) as writer:
            pd.DataFrame(self._missing_value_report).to_excel(writer, sheet_name="overview", index=False)
            pd.DataFrame(self._missing_identifier_report).to_excel(writer, sheet_name="missing_identifiers", index=False)

    def create_excel_summary_report(self, normality_test_type=None):

        with pd.ExcelWriter(self._paths["excel_summary_report"]) as writer:
            if self.num_col_names is not None:
                self._set_num_col_summary(normality_test_type=normality_test_type)
                pd.DataFrame(self.num_col_summary).to_excel(writer, sheet_name="numerical", index=False)
                
            if self.cat_col_names is not None:
                self._set_cat_col_summary()
                pd.DataFrame(self.cat_col_summary).to_excel(writer, sheet_name="categorical", index=False)
                
    def _set_missing_value_report(self):
        self._missing_value_report = []
        
        if self.cat_col_names is not None:
            for col_name in self.cat_col_names:
                col = self.df[col_name]
    
                temp_dict = {}
                temp_dict["col_name"] = col_name
                temp_dict["col_dtype"] = "categorical"
                temp_dict["missing_values"] = col.isnull().sum()
                temp_dict["total_observations"] = len(col)
    
                self._missing_value_report.append(temp_dict)
                
        if self.num_col_names is not None:
            for col_name in self.num_col_names:
                col = self.df[col_name]
                temp_dict = {}
                temp_dict["col_name"] = col_name
                temp_dict["col_dtype"] = "numerical"
                temp_dict["missing_values"] = col.isnull().sum()
                temp_dict["total_observations"] = len(col)
    
                self._missing_value_report.append(temp_dict)

    def _set_missing_value_identifier_report(self):
        self._missing_identifier_report = []
        for missing_value_report in self._missing_value_report:
            if missing_value_report["missing_values"] > 0:
                col = self.df[missing_value_report["col_name"]]
                for identifier in self.identifiers:
                    missing_identifiers = self.df.loc[col.isnull(), identifier].tolist()

                    for missing_identifier in missing_identifiers:
                        temp_dict = {}
                        temp_dict["identifier"] = identifier
                        temp_dict["missing_identifier"] = missing_identifier
                        temp_dict["col_name"] = missing_value_report["col_name"]
                        self._missing_identifier_report.append(temp_dict)
                        
    def _set_num_col_summary(self, normality_test_type = None):
        self.num_col_summary = []
        for col_name in self.num_col_names:
            col = self.df[col_name].dropna().to_numpy()
                        
            temp_dict = {}
            temp_dict["variable_name"] = col_name
            temp_dict["mean"] = np.mean(col).item()
            temp_dict["std"] = np.std(col).item()
            temp_dict["min"] = np.min(col)
            temp_dict["q1"] = q1 = np.percentile(col, q=25).item()
            temp_dict["median"] = q2 = np.percentile(col, q=50).item()
            temp_dict["q3"] = np.percentile(col, q=75).item()
            temp_dict["max"] = np.max(col)
            
            self.num_col_summary.append(temp_dict)
            
        if normality_test_type is not None:
            normality = Normality()
            for temp_dict in self.num_col_summary:
                col = self.df[temp_dict['variable_name']].dropna().to_numpy()
                normality.set_data(data=col)
                normality_report = normality.get_normality_report(test_type=normality_test_type)
                temp_dict['normality_test_type'] = normality_report['test_type']
                temp_dict['normal'] = normality_report['normal']
                temp_dict['normality_p_value'] = normality_report['p']
            
    def _set_cat_col_summary(self):
        self.cat_col_summary = []
        for col_name in self.cat_col_names:
            col = self.df[col_name].dropna()
            categories = col.value_counts()
            total = categories.sum()
            column_summary_list=[]
            for category, category_count in categories.items():
                percentage = (category_count / total) * 100
                temp_dict = {}
                temp_dict["variable_name"]=col_name
                temp_dict["category"]=category
                temp_dict["counts"] = category_count
                temp_dict["%"]=percentage
                self.cat_col_summary.append(temp_dict) 