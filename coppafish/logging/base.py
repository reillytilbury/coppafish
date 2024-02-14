import logging
from datetime import datetime
from typing import Union


DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
severity_to_name = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
}


def set_log_config(minimum_print_severity: int, log_file_path: str = None) -> None:
    """
    Set the required information before logging.

    Args:
        minimum_print_severity (int): the minimum severity of message to be printed to the terminal.
        log_file_path (str): the file path to the file to place all messages inside of. Default: do not save.
    """
    global _minimum_print_severity
    _minimum_print_severity = minimum_print_severity
    global _log_file
    _log_file = log_file_path
    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    logging.getLogger("coppafish").setLevel(logging.DEBUG)


def debug(msg: str) -> None:
    log(msg, DEBUG)


def info(msg: str) -> None:
    log(msg, INFO)


def warn(msg: str) -> None:
    log(msg, WARNING)


def error(msg: str) -> None:
    log(msg, ERROR)


def log(msg: Union[str, Exception], severity: int) -> None:
    """
    Log a message to the log file. Also, print message to the terminal if the message is severe enough.

    Args:
        msg (str or str like or Exception): message to log. Either a str or something that can be converted into a str.
        severity (int): severity of message.
        end (str, optional): end of print. Default: new line.
    """
    message = datetime_string()
    message += f":{severity_to_name[severity]}: "
    message += str(msg)
    if _log_file is not None:
        # Append message to log file
        with open(_log_file, "a") as log_file:
            log_file.write(message + "\n")
    if severity >= _minimum_print_severity:
        if severity >= ERROR:
            # Crash on error severity
            if isinstance(msg, Exception):
                raise msg
            raise LogError(message)
        logging.getLogger("coppafish").log(severity, message)


def datetime_string() -> str:
    """
    Get the current date/time in a readable format for the logs.

    Returns:
        str: current date and time as a string with second precision.
    """
    return datetime.now().strftime("%d/%m/%y %H:%M:%S")


class LogError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


set_log_config(INFO)
