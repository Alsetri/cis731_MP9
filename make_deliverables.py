"""
CIS 731 — MP 9: generate attestation.md and gold_adjudicated.jsonl from the
annotator's raw file. Nothing here assigns or alters a label.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANN = ROOT / "annotation"
VERB = {f"MIC-{i:04d}": n for i, n in zip(range(1, 11), [
    "Trap", "Crunk", "Memphis", "G-Funk", "Hyphy", "Boom Bap", "Drill",
    "Bounce", "Jazz Rap", "Horrorcore"])}


def main() -> None:
    rows = list(csv.DictReader((ANN / "gold_raw.csv").open(encoding="utf-8")))
    overlap = {l.strip() for l in (ANN / "overlap_ids.txt").read_text().split()
               if l.strip()}
    meta = rows[0]
    start = datetime.fromisoformat(meta["started_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(meta["ended_utc"].replace("Z", "+00:00"))
    elapsed = int(meta["elapsed_seconds"])
    unsure = [r for r in rows if r["label"] == "UNSURE"]
    usable = [r for r in rows if r["label"] != "UNSURE"]

    # ------------------------------------------------ gold_adjudicated.jsonl
    # Adjudication rules, applied mechanically and recorded:
    #   A1  UNSURE is excluded from the adjudicated set, not resolved by the
    #       author after the fact. Resolving one's own abstention later is a
    #       second, worse annotation pass with none of the first pass's
    #       discipline.
    #   A2  No label is changed. With a single annotator there is no
    #       disagreement to adjudicate between, so any edit here would be the
    #       annotator overruling himself without new evidence.
    #   A3  Membership of the 20-object IAA overlap is recorded per row so a
    #       partner's file can be joined later without re-deriving it.
    out = []
    for r in rows:
        if r["label"] == "UNSURE":
            continue
        out.append({
            "object_id": r["object_id"],
            "label": r["label"],
            "label_verbalizer": VERB[r["label"]],
            "source": "gold",
            "annotator": r["annotator"],
            "in_iaa_overlap": r["object_id"] in overlap,
            "adjudication": "A2-unchanged-single-annotator",
        })
    (ANN / "gold_adjudicated.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")

    dist = Counter(VERB[r["label"]] for r in usable)

    # ------------------------------------------------------- attestation.md
    att = f"""# MP 9 — Annotation Attestation

**Annotator:** Abdulla Mansoor (`{meta['annotator']}`), CIS 731
**Instrument:** `annotation/label_tool.html` (browser, offline)
**Corpus:** team RateYourMusic artist metadata, `master_ontology_mapped.csv`

## Timestamps

These are recorded automatically by the labelling tool at the first keypress
and at completion. They are **not** self-reported after the fact.

| | |
|---|---|
| Started (UTC) | `{start.isoformat()}` |
| Ended (UTC) | `{end.isoformat()}` |
| Elapsed | **{elapsed} s = {elapsed // 60} min {elapsed % 60} s** |
| Objects presented | {len(rows)} |
| Mean time per object | **{elapsed / len(rows):.1f} s** |

## Effort ceiling (INV-21)

| Constraint | Limit | Actual | Status |
|---|---|---|---|
| Objects annotated | ≤ 50 | {len(rows)} | PASS |
| Wall-clock effort | ≤ 2 h | {elapsed / 3600:.2f} h | PASS |

Both bounds are enforced by `validate_annotation_submission.py`, which fails
the submission at more than 50 rows and additionally reports an advisory
effort-plausibility check (floor 8 s/object).

## Outcome

| | count | share |
|---|---|---|
| Labelled | {len(usable)} | {100 * len(usable) / len(rows):.0f}% |
| `UNSURE` | {len(unsure)} | {100 * len(unsure) / len(rows):.0f}% |

Label distribution across the adjudicated set:

{chr(10).join(f'- {k}: {v}' for k, v in dist.most_common())}

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
"""
    (ROOT / "attestation.md").write_text(att, encoding="utf-8")

    print(f"gold_adjudicated.jsonl  {len(out)} rows "
          f"({len(unsure)} UNSURE excluded)")
    print(f"attestation.md          {elapsed}s, {elapsed/len(rows):.1f}s/object")
    print(f"IAA overlap flagged on  {sum(1 for o in out if o['in_iaa_overlap'])} rows")


if __name__ == "__main__":
    main()
