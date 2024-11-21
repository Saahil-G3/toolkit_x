import os
import csv
import logging
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
        log_folder: str,
        log_to_txt: bool = True,
        log_to_csv: bool = False,
    ):
        """
        Initializes the Logger.

        Args:
            name: The name of the logger, usually the module name.
            log_folder: The folder where log files will be stored.
            log_to_txt: Whether to save logs to a text file.
            log_to_csv: Whether to save logs to a CSV file.
        """
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        name = f"{name}_{timestamp.replace(' ', '_')}"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        self.logger.propagate = False  # Prevent duplicate logs

        log_folder = Path(log_folder)
        log_folder.mkdir(parents=True, exist_ok=True)

        # Handlers for logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

        if log_to_txt:
            txt_log_file = os.path.join(log_folder, f"{name}.log")
            txt_file_handler = RotatingFileHandler(
                txt_log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            txt_file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(txt_file_handler)

        if log_to_csv:
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
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )

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
