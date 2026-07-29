import itertools
from collections import OrderedDict
from collections.abc import Iterable, Iterator


def iter_by_triples[T](
    iterable: Iterable[T],
) -> Iterable[tuple[T | None, T, T | None]]:
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


def flatten[T](iterable: Iterator[Iterable[T]]) -> Iterable[T]:
    return itertools.chain.from_iterable(iterable)


def deduplicate[T](iterable: Iterable[T]) -> list[T]:
    return list(OrderedDict.fromkeys(iterable))


def merge(destination: dict, source: dict) -> dict:
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
