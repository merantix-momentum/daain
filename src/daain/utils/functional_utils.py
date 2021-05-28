from functools import reduce
from itertools import groupby
from typing import Any, Callable, Iterable, List, Tuple, TypeVar

T = TypeVar("T")  # Generic Type used to make type inference possible / applicable


def repeat_function(fn: Callable[[T], Any], num_times: int, start_value: T) -> T:
    """Applies the given function `fn` `num_times` in a left fold manner: fn(fn(fn(fn(start_value))))

    Args:
        fn:Callable[[T], Any]
            Function to use
        num_times:int
            How many folds
        start_value:T
            Starting value
    """

    return reduce(lambda acc, _: fn(acc), range(num_times), start_value)


def pipeline(fns: Iterable[Callable[[Any], Any]], init_value: Any) -> Any:
    """Executes given functions in a pipeline:
    E.g.: pipeline((a, b, c), x) == c(b(a(x)))
    """
    return reduce(lambda acc, el: el(acc), fns, init_value)


def group_by(xs: Iterable[Any], key: Callable) -> Iterable[Tuple[Any, List[Any]]]:
    for k, v in groupby(sorted(xs, key=key), key):
        yield k, list(v)
