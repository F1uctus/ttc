from typing import Iterable, TypeVar, Optional, List, Tuple

T = TypeVar("T")


def iter_by_abs_idx(seq: List[T], start: int) -> Iterable[Tuple[T, int]]:
    """
    Iterates over a sequence of elements with their indices following the
    canonical enumeration of integers (e.g. 0 1 -1 2 -2 3 -3 ...).

    Parameters
    ----------
    seq
        A sequence to iterate over.
    start
        An index to start from. Must be within given iterable.

    Returns
    -------
    Iterable[tuple]
        An iterable with tuples of (element, element offset from the start)
    """
    if len(seq) == 0 or not (0 <= start < len(seq)):
        return

    yield seq[start], 0
    offset = 0
    while True:
        if offset > 0:
            offset = -offset  # pos -> neg
            if start + offset < 0:
                return
        else:
            offset = -offset + 1  # neg -> pos + 1
            if start + offset >= len(seq):
                return
        yield seq[start + offset], offset


def canonical_int_enumeration() -> Iterable[int]:
    """
    Yields the canonical enumeration of integers (e.g. 0 1 -1 2 -2 3 -3).
    Also known as the A001057 sequence of OEIS.
    """
    yield 0
    offset = 1
    while True:
        yield offset
        offset = -offset
        if offset >= 0:
            offset += 1


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
