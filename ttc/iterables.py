import itertools
from typing import Iterable, TypeVar, Optional, List, Tuple, Dict

T = TypeVar("T")


def iter_by_triples(
    iterable: Iterable[T],
) -> Iterable[Tuple[Optional[T], T, Optional[T]]]:
    """
    Yields elements of given iterable with a sliding window of triplets, where the
    first element of the triple is "previous", second is "current", and third is "next".
    So, first is None on the first yield and third is None on the last, respectively.
    """
    iterator = iter(iterable)
    prv = None
    cur = next(iterator)
    try:
        while True:
            nxt = next(iterator)
            yield prv, cur, nxt
            prv, cur = cur, nxt
    except StopIteration:
        yield prv, cur, None


def flatmap(func, *iterable: Iterable[T]) -> Iterable[T]:
    return itertools.chain.from_iterable(map(func, *iterable))


def merge(destination: Dict, source: Dict) -> Dict:
    """
    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(a, b) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            merge(value, destination.setdefault(key, {}))
        else:
            destination[key] = value

    return destination
