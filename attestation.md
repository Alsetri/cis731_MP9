# MP 9 — Annotation Attestation

**Annotator:** Abdulla Mansoor (`abdulla03`), CIS 731
**Instrument:** `annotation/label_tool.html` (browser, offline)
**Corpus:** team RateYourMusic artist metadata, `master_ontology_mapped.csv`

## Timestamps

These are recorded automatically by the labelling tool at the first keypress
and at completion. They are **not** self-reported after the fact.

| | |
|---|---|
| Started (UTC) | `2026-11-19T03:36:00.624000+00:00` |
| Ended (UTC) | `2026-11-19T03:52:09.473000+00:00` |
| Elapsed | **968 s = 16 min 8 s** |
| Objects presented | 50 |
| Mean time per object | **19.4 s** |

## Effort ceiling (INV-21)

| Constraint | Limit | Actual | Status |
|---|---|---|---|
| Objects annotated | ≤ 50 | 50 | PASS |
| Wall-clock effort | ≤ 2 h | 0.27 h | PASS |

Both bounds are enforced by `validate_annotation_submission.py`, which fails
the submission at more than 50 rows and additionally reports an advisory
effort-plausibility check (floor 8 s/object).

## Outcome

| | count | share |
|---|---|---|
| Labelled | 40 | 80% |
| `UNSURE` | 10 | 20% |

Label distribution across the adjudicated set:

- Trap: 13
- Jazz Rap: 7
- Memphis: 4
- Boom Bap: 3
- Bounce: 3
- Crunk: 3
- G-Funk: 3
- Drill: 2
- Hyphy: 2

## Disclosure of a first, discarded pass

An earlier pass over the same 50 objects completed in **106 s (2.1 s/object)**
and was discarded before any analysis was reported. It failed the advisory
effort check, and its labels showed agreement with the silver labels of 0.10
(κ = −0.02) — chance level for a ten-class problem. The pass recorded above
is a full re-annotation, not an edit of the first.

This is disclosed rather than omitted because the discarded file exists in the
session history and because the contrast is itself evidence that the effort
check does what it is for.

## Declaration

I assigned every label in `gold_raw.csv` personally, from the evidence shown
on each card, applying the decision rules in `codebook.md`. No label was
assigned, suggested per-item, or altered by a generative model (INV-17). The
assistant drafted the codebook wording and built the labelling instrument;
it did not label.

Signed: ______________________________  Date: ______________
Abdulla Mansoor — CIS 731
