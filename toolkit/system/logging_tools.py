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
        log_to_console=True,
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
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)

        if log_to_txt:
            self.log_folder.mkdir(parents=True, exist_ok=True)
            txt_log_file = os.path.join(log_folder, f"{name}.log")
            txt_file_handler = RotatingFileHandler(
                txt_log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            txt_file_handler.setFormatter(self._get_text_formatter())
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
    def _get_text_formatter():
        formatter = logging.Formatter(
            "%(asctime)s,%(levelname)s\n%(message)s", datefmt="%d-%m-%Y %H:%M:%S"
        )
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
        self,
        print_time=False,
        timer_name=None,
        logs_folder: str = "logs",
    ):
        self._start_time = None
        self._end_time = None
        self.print_time = print_time

        self._timer_logs = []
        self._custom_metrics = {}
        self._lap_idx = 0
        self.temp_timer_dict = {}

        if timer_name:
            self._timer_name = timer_name
        else:
            self._timer_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        self._save_name = self._timer_name

        self.logs_folder = Path(logs_folder)

        self.timer_logs_path = Path(
            f"{self.logs_folder}/timer_logs_{self._timer_name}.csv"
        )

    def set_custom_timer_metrics(self, custom_metrics: dict):
        self._custom_metrics = custom_metrics

    def start(self):
        """Starts the timer."""
        self._start_time = time.perf_counter()
        self.temp_timer_dict = {}
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

        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started. Call `start()` before `stop()`."
            )

        elapsed_time, unit = self._get_elapsed_time()

        self.temp_timer_dict.update(self._custom_metrics)
        self.temp_timer_dict["time_taken"] = f"{elapsed_time} {unit}"

        self._timer_logs.append(self.temp_timer_dict)

        if self.print_time:
            print(self.temp_timer_dict["time_taken"])

    def change_timer_name(self, name):
        self._save_name = f"{self._timer_name}_{name}"
        self.timer_logs_path = Path(
            f"{self.logs_folder}/timer_logs_{self._save_name}.csv"
        )

    def save_timer_logs(self):
        self.temp_timer_dict.update(self._custom_metrics)
        self._timer_logs.append(self.temp_timer_dict)
        self._save_timer_logs()

    def reset(self):
        """Resets the timer."""
        self.start_time = None
        self.end_time = None
        self._start_subtime = None
        self._end_subtime = None
        self.temp_timer_dict = {}
        self._custom_metrics = {}

    def _get_elapsed_time(self):

        self._end_time = time.perf_counter()
        elapsed_time = self._end_time - self._start_time

        if elapsed_time < 60:
            elapsed_time = round(elapsed_time, 2)
            unit = "seconds"
        else:
            elapsed_time = round(elapsed_time / 60, 2)
            unit = "minutes"

        return elapsed_time, unit

    def start_subtimer(self):
        """Starts the timer."""
        self._start_subtime = time.perf_counter()
        self._end_subtime = None  # Reset end time
        if not self._start_time:
            self._start_time = self._start_subtime

    def stop_subtimer(self, process=None, comments=None):
        if self._start_subtime is None:
            raise RuntimeError(
                "Subtimer has not been started. Call `start_subtimer()` before `stop_subtimer()`."
            )
        elapsed_time, unit = self._get_elapsed_subtime()
        self.temp_timer_dict[process] = f"{elapsed_time} {unit}"
        if comments:
            self.temp_timer_dict["comments"] = comments

    def _get_elapsed_subtime(self):

        self._end_subtime = time.perf_counter()
        elapsed_time = self._end_subtime - self._start_subtime

        if elapsed_time < 60:
            elapsed_time = round(elapsed_time, 2)
            unit = "seconds"
        else:
            elapsed_time = round(elapsed_time / 60, 2)
            unit = "minutes"

        return elapsed_time, unit

    def lap(self, process=None, comments=None):
        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started. Call `start()` before `stop()`."
            )
        elapsed_time, unit = self._get_elapsed_time()
        self.temp_timer_dict[f"lap {self._lap_idx}: {process}"] = (
            f"{elapsed_time} {unit}"
        )
        if comments:
            self.temp_timer_dict[f"lap {self._lap_idx}: comments"] = comments
        self._lap_idx += 1

    def _save_timer_logs(self):
        df = pd.DataFrame(self._timer_logs)

        try:
            with open(self.timer_logs_path, "x") as f:
                df.to_csv(self.timer_logs_path, index=False, mode="w", header=True)
        except FileExistsError:
            df.to_csv(self.timer_logs_path, index=False, mode="a", header=False)

        self._timer_logs = []
