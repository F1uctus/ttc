from spacy.lang.char_classes import LIST_HYPHENS

HYPHENS = "".join(LIST_HYPHENS)
OPEN_QUOTES = frozenset({'"', "'", "`", "«", "‘", "‚", "“", "„", "‹", "❮", "''", "``"})
CLOSE_QUOTES = frozenset({'"', "'", "`", "»", "’", "‛", "”", "‟", "›", "❯", "''", "``"})
QUOTES = frozenset(OPEN_QUOTES | CLOSE_QUOTES)
ELLIPSES = {"…", "...", "....", ".....", "......", ".......", "........"}
