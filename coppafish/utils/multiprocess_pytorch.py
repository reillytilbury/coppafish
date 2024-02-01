from torch import multiprocessing
from typing import Callable, List, Any


def multiprocess_function(fn: Callable, args: List[Any]) -> List[Any]:
    """
    Run function `fn` by spawning `len(args)` number of processes using `pytorch.multiprocessing` module. Process `i`
    will run `fn` with arguments `args[i]`.

    Args:
        fn (Callable): function to multiprocess. The function must have exactly one parameter.
        args (list of tuple[any]]): each item in the list is the argument to pass into function call `fn`.

    Returns:
        output (list of any): `output[i]` is equal to `fn[args[i]]`, i.e. the order out outputs and inputs is 
            maintained.

    Notes:
        - This function will not check that the number of CPU cores is large enough to spawn so many processes, that
            must be thought about before calling.
    """
    with multiprocessing.Pool(len(args)) as p:
        result = p.map(fn, args)
    return result
