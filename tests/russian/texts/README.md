# Annotated corpus

## File format

```
<raw text>
--------------------
<Actor>::<Replica text>            one line per replica, in document order
--------------------               optional third section
<Canonical> = <alias> | <alias>    character aliases; '#' starts a comment
```

`None::` marks a replica whose speaker is genuinely unidentifiable.
Aliases let gold labels (and predictions) use several surface forms of one
character — e.g. `Ясна = принцесса | светлость`. Parser/serializer:
[`ttc/corpus.py`](../../../ttc/corpus.py). Create new files with
`ttc annotate` (machine-prefilled, click to correct).

## tune/ vs heldout/ — the contract

- **Heuristics may only ever be tuned against `tune/`.** Iterate with
  `ttc eval tests/russian/texts/tune --by-file --errors`.
- **`heldout/` is for aggregate numbers only.** Never read its per-replica
  errors while tuning (`ttc eval` refuses `--errors` on it), never write a
  rule to fix a specific held-out mistake. Record aggregates per milestone
  in `docs/eval-log.md`.
- **One source book belongs to exactly one split** — style and character
  names leak. (Grandfathered exception: Sanderson Stormlight excerpts exist
  in both; do not add more cross-split books.)
- Alias sections are identity facts of the text, fixed at annotation time —
  never extend them to make a wrong prediction pass.
- New raw texts for annotation go to `raw/` (git-ignored). Prefer
  public-domain prose; do not grow the copyrighted Sanderson excerpts.
