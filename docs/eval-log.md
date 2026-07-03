# Accuracy log

Aggregate numbers from `ttc eval tests/russian/texts/tune tests/russian/texts/heldout`.
Held-out is consulted **only** at milestones recorded here — see
[tests/russian/texts/README.md](../tests/russian/texts/README.md) for the contract.

Metric: end-to-end attribution accuracy (correctly attributed replicas / all gold replicas).

| Date | Rev | Corpus (tune/heldout pairs) | Model | Tune | Held-out | Notes |
|---|---|---|---|---|---|---|
| 2026-07-04 | a1f1117+eval-infra | 289 / 92 | lg | 100.0% | 69.6% | Baseline. Alias sections + 2 gold dash fixes applied to heldout (pre-alias heldout was 64.1%). Heldout extraction P/R 97.8% (2 replicas truncated at mid-replica ", –" constructs). |
| 2026-07-04 | a1f1117+eval-infra | 289 / 92 | md | 97.6% | 67.4% | |
| 2026-07-04 | a1f1117+eval-infra | 289 / 92 | sm | 97.2% | 67.4% | lg−sm on held-out = 2.2pp at n=92 — model gate (switch default to sm) to be decided after corpus expansion. |
