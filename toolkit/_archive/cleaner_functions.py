    def set_df(self, df, df_name=None):
        
        if df_name:
            self.df_dir = self.results_dir / df_name
            self.df_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.df_dir = self.results_dir

        self._column_type_counter_path = self.df_dir/"column_type_counter.pkl"
        
        self._column_reports_dir = self.df_dir / "column_reports"
        self._column_reports_dir.mkdir(exist_ok=True, parents=True)

        self._rectified_column_reports_dir = self.df_dir / "column_reports_rectified"
        self._rectified_column_reports_dir.mkdir(exist_ok=True, parents=True)

        self._df_path = Path(f"{self.df_dir}/df.csv")
        self._rectified_df_path = self.df_dir / "df_rectified.csv"
        
        self._set_paths(df_name=df_name)
        
        if self._rectified_df_path.exists():
            self.df = pd.read_csv(self._rectified_df_path)
            logger.info(
                f"Rectified df Exists at {self._rectified_df_path}, No Need for processing."
            )
        elif self._df_path.exists():
            self.df = pd.read_csv(self._df_path)
            logger.info(
                f"df saved before with identical configuration at {self._df_path}."
            )
        else:
            self.df = copy.deepcopy(df)
            self.df.to_csv(self._df_path, index=False)

        self.numerical_column_names_path = (self._column_reports_dir / "column_names_numerical.pkl")
        self.categorical_column_names_path = (self._column_reports_dir / "column_names_categorical.pkl")

        self.rectified_numerical_column_names_path = (self._rectified_column_reports_dir / "column_names_numerical.pkl")
        self.rectified_categorical_column_names_path = (self._rectified_column_reports_dir / "column_names_categorical.pkl")

        self.identifiers_path = self._rectified_column_reports_dir / "identifiers.pkl"

        if self.identifiers_path.exists():
            self.identifiers = load_pickle(self.identifiers_path)

        Process Numerical Columns
        if self.rectified_numerical_column_names_path.exists():
            self.numerical_column_names = load_pickle(self.rectified_numerical_column_names_path)
            logger.info(f"Rectified numerical column data exists.")
        elif self.numerical_column_names_path.exists():
            self.numerical_column_names = load_pickle(self.numerical_column_names_path)
            logger.info(f"Numerical column data exists.")
        else:
            self.numerical_column_names = df.select_dtypes(include=["number"]).columns.tolist()
            
            if len(self.numerical_column_names) == 0:
                logger.info("No numerical columns found in the dataframe.")
            else:
                self._create_numerical_column_report()
                save_pickle(self.numerical_column_names, self.numerical_column_names_path)

            self._numerical_column_counts = len(self.numerical_column_names)
                
        Process Categorical Columns
        if self.rectified_categorical_column_names_path.exists():
            self.categorical_column_names = load_pickle(self.rectified_categorical_column_names_path)
            logger.info(f"Rectified categorical column data exists.")
        elif self.categorical_column_names_path.exists():
            self.categorical_column_names = load_pickle(self.categorical_column_names_path)
            logger.info(f"Categorical column data exists.")
        else:
            self.categorical_column_names = df.select_dtypes(include=["object"]).columns.tolist()
            
            if len(self.categorical_column_names) == 0:
                logger.info("No categorical columns found in the dataframe.")
            else:
                self._create_categorical_column_report()
                save_pickle(self.categorical_column_names, self.categorical_column_names_path)

            self._categorical_column_counts = len(self.categorical_column_names)
        
        
        if (
            self.rectified_numerical_column_names_path.exists()
            and self.rectified_categorical_column_names_path.exists()
        ):
            self.numerical_column_names = load_pickle(
                self.rectified_numerical_column_names_path
            )
            self.categorical_column_names = load_pickle(
                self.rectified_categorical_column_names_path
            )
            self.identifiers = load_pickle(self.identifiers_path)

            logger.info(f"Rectified column data exists.")

        elif (
            self.numerical_column_names_path.exists()
            and self.numerical_column_names_path.exists()
        ):
            self.numerical_column_names = load_pickle(self.numerical_column_names_path)
            self.categorical_column_names = load_pickle(
                self.categorical_column_names_path
            )
            logger.info(f"Numerical and categorical columns exists.")

        else:
            self.numerical_column_names = df.select_dtypes(
                include=["number"]
            ).columns.tolist()
            self.categorical_column_names = df.select_dtypes(
                include=["object"]
            ).columns.tolist()
            if len(self.categorical_column_names) == 0:
                logger.info("No categorical columns found in the dataframe.")
            else:
                self._create_categorical_column_report()
                save_pickle(self.categorical_column_names, self.categorical_column_names_path)
            
            if len(self.numerical_column_names) == 0:
                logger.info("No numerical columns found in the dataframe.")
            else:
                self._create_numerical_column_report()
                save_pickle(self.numerical_column_names, self.numerical_column_names_path)
        
        
        if self.rectified_numerical_column_names_path.exists():
            self.numerical_column_names = load_pickle(self.rectified_numerical_column_names_path)
            logger.info(f"Rectified numerical columns exists.")
        elif self.numerical_column_names_path.exists():
            self.numerical_column_names = load_pickle(self.numerical_column_names_path)
            logger.info(f"Numerical columns exists.")
        else:
            self.numerical_column_names = df.select_dtypes(include=['number']).columns.tolist()
            save_pickle(self.numerical_column_names, self.numerical_column_names_path)

        if self.rectified_categorical_column_names_path.exists():
           self.categorical_column_names = load_pickle(self.rectified_categorical_column_names_path)
           logger.info(f"Rectified categorical columns exists.")
        elif self.categorical_column_names_path.exists():
           self.categorical_column_names = load_pickle(self.categorical_column_names_path)
           logger.info(f"Categorical columns exists.")
        else:
           self.categorical_column_names = df.select_dtypes(include=['object']).columns.tolist()
           save_pickle(self.categorical_column_names, self.categorical_column_names_path)