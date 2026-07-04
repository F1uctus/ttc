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

## Corpus data layer (Phase 1) — adapter conversion counts

Real-download verification of `ttc/corpora/` adapters (2026-07-04, converted
outside the repo under `~/corpora/`; data never committed):

| Source | Docs | Replicas | Notes |
|---|---|---|---|
| rusdracor (ru, drama) | 212 | 119,333 | DraCor API; server gzips TEI, adapter decompresses |
| pdnc (en, prose) | 28 | 48,810 | byte==char offsets (ASCII texts); multi-span quotes → 1 replica/span |
| jy_quoteplus (zh, prose) | 8,144 | 8,144 | per-quote context JSON; one doc per quote item |
| droc (de, prose) | 90 | 4,490 | UIMA XMI (not TEI); speech 4,377 + thought 113 |

DROC notes: real release is UIMA CAS XMI, not TEI — adapter rewritten
accordingly. `DirectSpeech.Speaker` → `NamedEntity.xmi:id` → coref cluster.
`Replica.mode` carries `speech`/`thought` (thoughts kept for TTS voicing).
Validation flags ~227 replica-overlap issues: genuine nested/enclosing
utterances in the source (e.g. a thought inside a long epistolary speech),
not parser bugs — left for downstream filtering, source not mutated.
