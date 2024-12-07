import copy
import pandas as pd
from pathlib import Path
from datetime import datetime

from toolkit.system.storage.data_io_tools import load_pickle, save_pickle
from toolkit.system.logging_tools import Logger

logger = Logger(name="cleaner").get_logger()

pd.set_option("future.no_silent_downcasting", True)


def sanitize_sheet_name(sheet_name):
    invalid_chars = [":", "/", "\\", "?", "*", "[", "]"]
    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")
    return sheet_name


def get_datetime_run_id():
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    return current_datetime


class Cleaner:
    def __init__(self, run_id=None):

        if not run_id:
            run_id = get_datetime_run_id()

        self.run_id = run_id
        self.results_dir = Path(f"analytics/{self.run_id}/cleaner")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.numerical_column_names = None
        self.categorical_column_names = None
        self.identifiers = []
        self._metadata = None

    def _set_paths(self, df_name):
        if df_name:
            self.df_dir = self.results_dir / df_name
            self.df_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.df_dir = self.results_dir

        self._df_path = Path(f"{self.df_dir}/df.csv")

        self._rectified_df_path = self.df_dir / "df_rectified.csv"

        self._column_type_counter_path = self.df_dir / "column_type_counter.pkl"

        self._column_reports_dir = self.df_dir / "column_reports"
        self._column_reports_dir.mkdir(exist_ok=True, parents=True)

        self._rectified_column_reports_dir = self.df_dir / "column_reports_rectified"
        self._rectified_column_reports_dir.mkdir(exist_ok=True, parents=True)

        self._numerical_column_names_path = (
            self._column_reports_dir / "column_names_numerical.pkl"
        )
        self._categorical_column_names_path = (
            self._column_reports_dir / "column_names_categorical.pkl"
        )

        self._rectified_categorical_columns_report_path = (
            self._rectified_column_reports_dir / "column_report_categorical.xlsx"
        )
        self._rectified_numerical_columns_report_path = (
            self._rectified_column_reports_dir / "column_report_numerical.csv"
        )

        self._rectified_numerical_column_names_path = (
            self._rectified_column_reports_dir / "column_names_numerical.pkl"
        )
        self._rectified_categorical_column_names_path = (
            self._rectified_column_reports_dir / "column_names_categorical.pkl"
        )

        self._identifiers_path = self._rectified_column_reports_dir / "identifiers.pkl"

        self._metadata_path = self.df_dir / "metadata.pkl"

    def set_df(self, df, df_name=None):
        self._set_paths(df_name=df_name)

        if self._metadata_path.exists():
            logger.info("Metadata exists")
            self._metadata = load_pickle(self._metadata_path)

        if self._rectified_df_path.exists():
            self.df = pd.read_csv(self._rectified_df_path)
            logger.info(
                f"Rectified df Exists at {self._rectified_df_path}, No Need for processing."
            )
            return

        elif self._df_path.exists():
            self.df = pd.read_csv(self._df_path)
            logger.info(
                f"df saved before with identical configuration at {self._df_path}."
            )
        else:
            self.df = copy.deepcopy(df)
            self.df.to_csv(self._df_path, index=False)

        if self._metadata is not None:
            logger.info("Metadata exists")

            # Numerical Trigger
            if self._metadata["numerical_column_counts"] > 0:
                self._set_numerical_columns()

                if self._rectified_numerical_columns_report_path.exists():
                    if self._rectified_numerical_column_names_path.exists():
                        self.numerical_column_names = load_pickle(
                            self._rectified_numerical_column_names_path
                        )
                        logger.info(f"Rectified numerical column data exists.")
                        self._metadata[
                            "processed_rectified_numerical_column_reports"
                        ] = False
                    else:
                        self._process_rectified_numerical_column_reports()
                        self._metadata[
                            "processed_rectified_numerical_column_reports"
                        ] = True
                else:
                    logger.info(
                        f"Numerical column data exists, please add rectified data for further processing."
                    )
                    self._metadata["processed_rectified_numerical_column_reports"] = (
                        False
                    )
            else:
                self._metadata["processed_rectified_numerical_column_reports"] = True

            # Categorical Trigger
            if self._metadata["categorical_column_counts"] > 0:
                self._set_categorical_columns()

                if self._rectified_categorical_columns_report_path.exists():
                    if self._rectified_categorical_column_names_path.exists():
                        self.categorical_column_names = load_pickle(
                            self._rectified_categorical_column_names_path
                        )
                        logger.info(f"Rectified categorical column data exists.")
                        self._metadata[
                            "processed_rectified_categorical_column_reports"
                        ] = False
                    else:
                        self._process_rectified_categorical_column_reports()
                        self._metadata[
                            "processed_rectified_categorical_column_reports"
                        ] = True
                else:
                    logger.info(
                        f"Categorical column data exists, please add rectified data for further processing."
                    )
                    self._metadata["processed_rectified_categorical_column_reports"] = (
                        False
                    )
            else:
                self._metadata["processed_rectified_categorical_column_reports"] = True

        else:
            self._metadata = {}
            self._set_identifiers()

            self._set_numerical_columns()
            self._metadata["processed_rectified_numerical_column_reports"] = False

            self._set_categorical_columns()
            self._metadata["processed_rectified_categorical_column_reports"] = False
            save_pickle(self._metadata, self._metadata_path)

        if (
            self._metadata["processed_rectified_numerical_column_reports"]
            and self._metadata["processed_rectified_categorical_column_reports"]
        ):
            self.df.to_csv(self._rectified_df_path, index=False)
            save_pickle(self.identifiers, self._identifiers_path)

    # def process_rectified_reports(self):
    #     if self._metadata is None:
    #         self._metadata = load_pickle(self._metadata_path)

    #     if self._metadata["numerical_column_counts"]>0:
    #         self._process_rectified_numerical_column_reports()

    #     if self._metadata["categorical_column_counts"]>0:
    #         self._process_rectified_categorical_column_reports()

    def _set_identifiers(self):
        if self._identifiers_path.exists():
            self.identifiers = load_pickle(self._identifiers_path)

    def _set_numerical_columns(self):
        if self._numerical_column_names_path.exists():
            self.numerical_column_names = load_pickle(self._numerical_column_names_path)

        else:
            self.numerical_column_names = self.df.select_dtypes(
                include=["number"]
            ).columns.tolist()

            if len(self.numerical_column_names) == 0:
                logger.info("No numerical columns found in the dataframe.")
            else:
                self._create_numerical_column_report()
                save_pickle(
                    self.numerical_column_names, self._numerical_column_names_path
                )
            self._metadata["numerical_column_counts"] = len(self.numerical_column_names)

    def _set_categorical_columns(self):

        if self._categorical_column_names_path.exists():
            self.categorical_column_names = load_pickle(
                self._categorical_column_names_path
            )

        else:
            self.categorical_column_names = self.df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            if len(self.categorical_column_names) == 0:
                logger.info("No categorical columns found in the dataframe.")
            else:
                self._create_categorical_column_report()
                save_pickle(
                    self.categorical_column_names, self._categorical_column_names_path
                )

            self._metadata["categorical_column_counts"] = len(
                self.categorical_column_names
            )

    def _change_categorical_labels(self, column_name, label_change_dict):
        self.df[column_name] = self.df[column_name].replace(label_change_dict)
        logger.info(f"Labels changed for Categorical Column {column_name}")

    def _create_numerical_column_report(self):
        if self.numerical_column_names is None:
            raise ValueError("No numerical columns found, please check.")

        numerical_columns = []
        for idx, column_name in enumerate(self.numerical_column_names):
            temp_dict = {}
            temp_dict["column_name"] = column_name
            temp_dict["numerical"] = True
            temp_dict["change_column_name_to"] = pd.NA
            temp_dict["add_to_identifiers"] = pd.NA
            temp_dict["add_to_categorical"] = pd.NA
            temp_dict["remove_from_analysis"] = pd.NA
            numerical_columns.append(temp_dict)
        pd.DataFrame(numerical_columns).to_csv(
            f"{self._column_reports_dir}/column_report_numerical.csv", index=False
        )

    def _create_categorical_column_report(self):
        if self.categorical_column_names is None:
            raise ValueError("No categorical columns found, please check.")
        with pd.ExcelWriter(
            f"{self._column_reports_dir}/column_report_categorical.xlsx"
        ) as writer:
            categorical_columns = []
            for idx, column_name in enumerate(self.categorical_column_names):

                temp_dict = {}
                temp_dict["column_name"] = column_name
                temp_dict["categorical"] = True
                temp_dict["add_to_identifiers"] = pd.NA
                temp_dict["add_to_numerical"] = pd.NA
                temp_dict["remove_from_analysis"] = pd.NA
                temp_dict["change_column_name_to"] = pd.NA

                categorical_columns.append(temp_dict)

            temp_df = pd.DataFrame(categorical_columns)
            temp_df.to_excel(writer, sheet_name="categorical_columns", index=False)

            for column_name in self.categorical_column_names:

                categorical_columns = []
                column = self.df[column_name]
                categories = column.value_counts()

                for category_label, value_counts in categories.items():

                    temp_dict = {}
                    temp_dict["category_label"] = category_label
                    temp_dict["counts"] = value_counts
                    temp_dict["change_label_to"] = pd.NA

                    categorical_columns.append(temp_dict)

                # print(column_name)
                temp_df = pd.DataFrame(categorical_columns)
                temp_df.to_excel(
                    writer, sheet_name=sanitize_sheet_name(column_name), index=False
                )

    def _process_rectified_categorical_column_reports(self):

        rectified_categorical_columns_excel = pd.ExcelFile(
            self._rectified_categorical_columns_report_path
        )
        sheet_names = rectified_categorical_columns_excel.sheet_names
        sheet_names.remove("categorical_columns")

        rectified_categorical_columns_df = rectified_categorical_columns_excel.parse(
            "categorical_columns"
        )
        rectified_categorical_columns_df["add_to_identifiers"] = (
            rectified_categorical_columns_df["add_to_identifiers"]
            .fillna(False)
            .astype(bool)
        )
        rectified_categorical_columns_df["add_to_numerical"] = (
            rectified_categorical_columns_df["add_to_numerical"]
            .fillna(False)
            .astype(bool)
        )
        rectified_categorical_columns_df["remove_from_analysis"] = (
            rectified_categorical_columns_df["remove_from_analysis"]
            .fillna(False)
            .astype(bool)
        )
        rectified_categorical_columns_df["change_column_name_to"] = (
            rectified_categorical_columns_df["change_column_name_to"].fillna(False)
        )

        self._process_categorical_column_names(rectified_categorical_columns_df)

        label_change_dicts = {}
        for sheet_name in sheet_names:
            temp_df = rectified_categorical_columns_excel.parse(
                sheet_name, index_col=None
            )
            temp_df["change_label_to"] = temp_df["change_label_to"].fillna(False)

            temp_dict = {}
            for idx, row in temp_df.iterrows():
                if row["change_label_to"]:
                    temp_dict[row["category_label"]] = row["change_label_to"]
            if temp_dict:
                label_change_dicts[sheet_name] = temp_dict

        
        for column_name, label_change_dict in label_change_dicts.items():
            if column_name in self.df.columns:
                self._change_categorical_labels(column_name, label_change_dict)

        save_pickle(
            self.categorical_column_names, self._rectified_categorical_column_names_path
        )

    def _process_rectified_numerical_column_reports(self):

        rectified_numerical_columns_df = pd.read_csv(
            self._rectified_numerical_columns_report_path
        )

        rectified_numerical_columns_df["add_to_identifiers"] = (
            rectified_numerical_columns_df["add_to_identifiers"]
            .fillna(False)
            .astype(bool)
        )
        rectified_numerical_columns_df["add_to_categorical"] = (
            rectified_numerical_columns_df["add_to_categorical"]
            .fillna(False)
            .astype(bool)
        )
        rectified_numerical_columns_df["remove_from_analysis"] = (
            rectified_numerical_columns_df["remove_from_analysis"]
            .fillna(False)
            .astype(bool)
        )
        rectified_numerical_columns_df["change_column_name_to"] = (
            rectified_numerical_columns_df["change_column_name_to"].fillna(False)
        )
        self._process_numerical_column_names(rectified_numerical_columns_df)

        save_pickle(
            self.numerical_column_names, self._rectified_numerical_column_names_path
        )

    def _process_categorical_column_names(self, rectified_df):

        for idx, row in rectified_df.iterrows():
            column_name = row["column_name"]
            add_to_identifiers = row["add_to_identifiers"]
            add_to_numerical = row["add_to_numerical"]
            remove_from_analysis = row["remove_from_analysis"]
            change_column_name_to = row["change_column_name_to"]

            if change_column_name_to:
                self.df.rename(
                    columns={column_name: change_column_name_to}, inplace=True
                )

                self.categorical_column_names.append(change_column_name_to)
                logger.info(
                    f"Successfully renamed {column_name} to {change_column_name_to}"
                )

                if column_name in self.categorical_column_names:
                    self.categorical_column_names.remove(column_name)
                    column_name = change_column_name_to
                else:
                    logger.info(
                        f"Can't remove {column_name} after renaming, not in the list."
                    )

            if remove_from_analysis:
                if column_name in self.categorical_column_names:
                    self.categorical_column_names.remove(column_name)
                    self.df = self.df.drop(columns=[column_name])
                else:
                    print(f"Can't remove {column_name}, not in the list.")

            elif add_to_identifiers:
                self.identifiers.append(column_name)
                if column_name in self.categorical_column_names:
                    self.categorical_column_names.remove(column_name)
                else:
                    print(
                        f"Can't remove {column_name} after adding to identifier, not in the list."
                    )

            elif add_to_numerical:
                self.numerical_column_names.append(column_name)
                if column_name in self.categorical_column_names:
                    self.categorical_column_names.remove(column_name)
                else:
                    print(
                        f"Can't remove {column_name} after adding to numerical, not in the list."
                    )

    def _process_numerical_column_names(self, rectified_df):
        for idx, row in rectified_df.iterrows():
            column_name = row["column_name"]
            add_to_identifiers = row["add_to_identifiers"]
            add_to_categorical = row["add_to_categorical"]
            remove_from_analysis = row["remove_from_analysis"]
            change_column_name_to = row["change_column_name_to"]

            if change_column_name_to:
                self.df.rename(
                    columns={column_name: change_column_name_to}, inplace=True
                )

                self.numerical_column_names.append(change_column_name_to)

                logger.info(
                    f"Successfully renamed {column_name} to {change_column_name_to}"
                )

                if column_name in self.numerical_column_names:
                    self.numerical_column_names.remove(column_name)
                    column_name = change_column_name_to
                else:
                    logger.info(
                        f"Can't remove {column_name} after renaming, not in the list."
                    )

            if remove_from_analysis:
                if column_name in self.numerical_column_names:
                    self.numerical_column_names.remove(column_name)
                    self.df = self.df.drop(columns=[column_name])
                else:
                    logger.info(f"Can't remove {column_name}, not in the list.")

            elif add_to_identifiers:
                self.identifiers.append(column_name)
                if column_name in self.numerical_column_names:
                    self.numerical_column_names.remove(column_name)
                else:
                    logger.info(
                        f"Can't remove {column_name} after adding to identifier, not in the list."
                    )

            elif add_to_categorical:
                self.categorical_column_names.append(column_name)
                if column_name in self.categorical_column_names:
                    self.numerical_column_names.remove(column_name)

                else:
                    logger.info(
                        f"Can't remove {column_name} after adding to categorical, not in the list."
                    )
