import numpy as np
import psutil


def get_available_memory() -> float:
    """
    Get system's available memory at the time of calling this function.

    Returns:
        float: available memory in GB.
    """
    return psutil.virtual_memory().available / 1e9


def get_core_count() -> int:
    """
    Get the number of threads available for multiprocessing tasks on the system.

    Returns:
        int: number of available threads.
    """
    n_threads = psutil.cpu_count(logical=True)
    if n_threads is None:
        n_threads = 1
    else:
        n_threads -= 2
    n_threads = np.clip(n_threads, 1, 999, dtype=int)

    return int(n_threads)
