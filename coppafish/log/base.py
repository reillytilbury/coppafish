from datetime import datetime
import logging
import smtplib
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Union

DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
EMAIL_ON = ERROR
CRASH_ON = ERROR
severity_to_name = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
}


def error_catch(func: Callable, *args, **kwargs) -> Any:
    """
    Any raised Exceptions that are not Keyboard/System interrupts are captured here and then sent to the logger as an
    error so the errors can be saved to the pipeline's .log file, when given.

    Args:
        func (Callable): function to run and catch errors on. All other parameters are input into func.

    Returns:
        Any: function output.
    """
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        error(e)
        raise RuntimeError("Should not reach here")
    return result


def set_log_config(
    minimum_print_severity: int = INFO,
    log_file_path: str = None,
    email_recipient: str = None,
    email_sender: str = None,
    email_sender_password: str = None,
) -> None:
    """
    Set the required information before logging.

    Args:
        minimum_print_severity (int): the minimum severity of message to be printed to the terminal.
        log_file_path (str, optional): the file path to the file to place all messages inside of. Default: do not save.
    """
    global _start_time
    _start_time = time.time()
    global _minimum_print_severity
    _minimum_print_severity = minimum_print_severity
    global _log_file
    _log_file = log_file_path
    global _email_recipient
    _email_recipient = email_recipient
    global _email_sender
    _email_sender = email_sender
    global _email_sender_password
    _email_sender_password = email_sender_password
    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    logging.getLogger("coppafish").setLevel(logging.DEBUG)


def debug(msg: Union[str, Exception], force_email: bool = False) -> None:
    log(msg, DEBUG, force_email=force_email)


def info(msg: Union[str, Exception], force_email: bool = False) -> None:
    log(msg, INFO, force_email=force_email)


def warn(msg: Union[str, Exception], force_email: bool = False) -> None:
    log(msg, WARNING, force_email=force_email)


def error(msg: Union[str, Exception], force_email: bool = False) -> None:
    log(msg, ERROR, force_email=force_email)


def log(msg: Union[str, Exception], severity: int, force_email: bool = False) -> None:
    """
    Log a message to the log file. The message is printed to the terminal if the message is severe enough.

    Args:
        msg (str or str like or Exception): message to log. Either a str or something that can be converted into a str.
        severity (int): severity of message.
        force_email (bool): force send an email to the recipient, no matter the severity.
    """
    message = datetime_string()
    message += f":{severity_to_name[severity]}: "
    message += str(msg)
    if _log_file is not None:
        # Append message to log file
        append_to_log_file(message)
    if severity >= _minimum_print_severity:
        logging.getLogger("coppafish").log(severity, message)
    if (
        (severity >= EMAIL_ON or force_email)
        and _email_sender is not None
        and _email_sender_password is not None
        and _email_recipient is not None
    ):
        delta_time = (time.time() - _start_time) / 60
        email_message = (
            f"On device {socket.gethostname()}, "
            + f"after {round(delta_time // 60)}hrs and {round(delta_time % 60)}mins:\n\n"
            + message
            + f"\n\nFor errors, please refer to our troubleshoot page "
            + f"(https://reillytilbury.github.io/coppafish/troubleshoot/)"
        )
        send_email(
            f"COPPAFISH: {severity_to_name[severity]}",
            email_message,
            _email_sender,
            _email_recipient,
            _email_sender_password,
        )
    if severity >= CRASH_ON:
        # Crash on high severity
        if isinstance(msg, Exception):
            # Add the traceback to the log file for debugging purposes
            append_to_log_file(traceback.format_exc())
            raise msg
        raise LogError(message)


def datetime_string() -> str:
    """
    Get the current date/time in a readable format for the logs.

    Returns:
        str: current date and time as a string with second precision.
    """
    return datetime.now().strftime("%d/%m/%y %H:%M:%S.%f")


def append_to_log_file(message: str) -> None:
    """
    Appends `message` to the end of the user's log file, if it exists.

    Args:
        message (str): message to append.
    """
    if _log_file is None:
        return
    with open(_log_file, "a") as log_file:
        log_file.write(message + "\n")


def log_package_versions(severity: int = DEBUG) -> None:
    """
    Log the current Python version and the Python package versions.

    Args:
        severity (int, optional): the log severity. Default: debug.
    """
    log(f"Python=={get_python_version()}", severity=severity)
    pip_list = str(subprocess.run(["python", "-m", "pip", "list"], capture_output=True, text=True).stdout)
    pip_list: list[str] = pip_list.split("\n")
    pip_list = pip_list[2:-1]
    names = []
    versions = []
    for package in pip_list:
        package = package.strip()
        separation_index = package.index(" ")
        names.append(package[:separation_index])
        versions.append(package[separation_index:].strip())
    for name, version in zip(names, versions):
        log(f"{name}=={version}", severity=severity)


def get_python_version() -> str:
    """
    Get the running Python version.

    Returns:
        str: python version as a string.
    """
    return sys.version.split()[0]


def send_email(subject: str, body: str, sender: str, recipient: str, password: str) -> None:
    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.starttls()
    s.login(sender, password)
    message = "Subject: {}\n\n{}".format(subject, body)
    s.sendmail(sender, recipient, message)
    s.quit()


class LogError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


set_log_config()
