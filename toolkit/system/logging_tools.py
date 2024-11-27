import os
import csv
import time
import logging
import colorlog
import pandas as pd
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


class CSVLogHandler(logging.Handler):
    def __init__(self, csv_filename: Path):
        super().__init__()
        self.csv_filename = csv_filename

    def emit(self, record):
        log_entry = self.format(record)
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        date, time = timestamp.split(" ")
        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([date, time, record.levelname, record.getMessage()])


class Logger:
    def __init__(
        self,
        name: str,
        log_folder: str = "logs",
        log_to_txt: bool = False,
        log_to_csv: bool = False,
        add_timestamp: bool = False,
    ):
        """
        Initializes the Logger.

        Args:
            name: The name of the logger, usually the module name.
            log_folder: The folder where log files will be stored.
            log_to_txt: Whether to save logs to a text file.
            log_to_csv: Whether to save logs to a CSV file.
        """
        if add_timestamp:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            name = f"{name}_{timestamp.replace(' ', '_')}"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        self.logger.propagate = False  # Prevent duplicate logs

        self.log_folder = Path(log_folder)

        # Handlers for logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

        if log_to_txt:
            self.log_folder.mkdir(parents=True, exist_ok=True)
            txt_log_file = os.path.join(log_folder, f"{name}.log")
            txt_file_handler = RotatingFileHandler(
                txt_log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            txt_file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(txt_file_handler)

        if log_to_csv:
            self.log_folder.mkdir(parents=True, exist_ok=True)
            csv_log_file = os.path.join(log_folder, f"{name}.csv")
            self._ensure_csv_header(csv_log_file)
            csv_handler = CSVLogHandler(Path(csv_log_file))
            csv_handler.setFormatter(self._get_csv_formatter())
            self.logger.addHandler(csv_handler)

    @staticmethod
    def _get_formatter():
        """
        Returns the standard log formatter.
        """
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s (%(asctime)s)%(reset)s\n%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # formatter = logging.Formatter(
        #    "%(message)s - %(asctime)s", datefmt="%d-%m-%Y %H:%M:%S"
        # )
        # formatter = logging.Formatter(
        #    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        #    datefmt="%d-%m-%Y %H:%M:%S",
        # )

        return formatter

    @staticmethod
    def _get_csv_formatter():
        """
        Returns a CSV-compatible formatter.
        """
        return logging.Formatter(
            "%(asctime)s,%(levelname)s,%(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )

    def _ensure_csv_header(self, csv_log_file):
        """
        Ensures that the CSV file has the correct header: Date, Time, Level, Message.
        """
        if not os.path.exists(csv_log_file) or os.stat(csv_log_file).st_size == 0:
            with open(csv_log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Level", "Message"])

    def get_logger(self):
        """
        Returns the logger instance.

        Returns:
            A logger instance configured with the specified handlers and level.
        """
        return self.logger


class Timer:
    def __init__(
        self, print_time=False, timer_name=None, logs_folder=None, save_logs=False
    ):
        self._start_time = None
        self._end_time = None
        self.print_time = print_time
        self._timer_run_counter = 0
        self._timer_logs = []
        self._custom_metrics = {}
        
        if timer_name:
            self._timer_name = timer_name
        else:
            self._timer_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        self.logs_folder = Path(logs_folder) if logs_folder else None

        self.timer_logs_path = Path(
            f"{self.logs_folder}/timer_logs_{self._timer_name}.csv"
        )
        self.save_logs = save_logs

    def _save_timer_logs(self):
        df = pd.DataFrame(self._timer_logs)

        try:
            with open(self.timer_logs_path, "x") as f:
                df.to_csv(self.timer_logs_path, index=False, mode="w", header=True)
        except FileExistsError:
            df.to_csv(self.timer_logs_path, index=False, mode="a", header=False)

        self._timer_logs = []

    def set_custom_timer_metrics(self, custom_metrics: dict):
        self._custom_metrics = custom_metrics

    def start(self):
        """Starts the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None  # Reset end time

    def stop(self):
        """
        Stops the timer and returns the elapsed time.

        Returns:
            float: Elapsed time in seconds if less than a minute.
            str: Elapsed time in minutes (as a formatted string) if more than a minute.
        Raises:
            RuntimeError: If the timer has not been started before calling stop.
        """
        self._timer_run_counter += 1
        self._temp_timer_dict = {}
        self._temp_timer_dict.update(self._custom_metrics)
        self._temp_timer_dict["run"] = self._timer_run_counter
        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started. Call `start()` before `stop()`."
            )

        self._end_time = time.perf_counter()
        elapsed_time = self._end_time - self._start_time

        if elapsed_time < 60:
            elapsed_time = round(elapsed_time, 2)
            self._temp_timer_dict["time_taken"] = elapsed_time
            self._temp_timer_dict["unit"] = "seconds"
        else:
            elapsed_time = round(elapsed_time / 60, 2)
            self._temp_timer_dict["time_taken"] = elapsed_time
            self._temp_timer_dict["unit"] = "minutes"

        self._timer_logs.append(self._temp_timer_dict)

        if self.save_logs:
            self._save_timer_logs()

        if self.print_time:
            print(
                f" {self._temp_timer_dict['time_taken']} {self._temp_timer_dict['unit']}"
            )

    def reset(self):
        """Resets the timer."""
        self.start_time = None
        self.end_time = None