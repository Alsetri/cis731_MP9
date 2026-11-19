"""
CIS 731 — MP 9: deterministic annotation-submission gate.

The spec names `validate_annotation_submission` but no such script ships in
the course repository (and the spec's own REVIEW R-5 records the surrounding
mechanisms as drafted). This is my implementation of the stated gate.

Checks, in order:
  1. gold_raw.csv exists
  2. row count <= 50            -- the hard CI gate; >50 returns zero
  3. required columns present
  4. every label is in the closed space or UNSURE
  5. object_ids are unique and all belong to the sampled annotation set
  6. elapsed_seconds is consistent with a good-faith reading pass

Check 6 is advisory and does not fail the gate, because the rubric's hard
constraint is the row count. It is reported because `attestation.md` is a
graded artifact and an implausible duration is visible to any reader of it.

Exit 0 = pass, 1 = fail.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANN = ROOT / "annotation"
GOLD = ANN / "gold_raw.csv"
MAX_ROWS = 50
REQUIRED = {"object_id", "label", "source", "annotator",
            "started_utc", "ended_utc", "elapsed_seconds"}
VALID = {f"MIC-{i:04d}" for i in range(1, 11)} | {"UNSURE"}
# Seconds per object below which a pass is flagged as implausibly fast. Reading
# four trope descriptions and applying the codebook does not take two seconds.
MIN_SEC_PER_OBJECT = 8.0

fails: list[str] = []
warns: list[str] = []


def check(ok: bool, msg: str, hard: bool = True) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL' if hard else 'WARN'}] {msg}")
    if not ok:
        (fails if hard else warns).append(msg)
    return ok


def main() -> None:
    print("validate_annotation_submission\n")
    if not check(GOLD.exists(), f"{GOLD.relative_to(ROOT)} exists"):
        print("\nGATE: FAIL — nothing to validate.")
        sys.exit(1)

    rows = list(csv.DictReader(GOLD.open(encoding="utf-8")))
    n = len(rows)

    check(n <= MAX_ROWS,
          f"row count {n} <= {MAX_ROWS} (hard CI gate; >50 returns zero)")
    check(n > 0, "at least one annotated object")
    check(REQUIRED <= set(rows[0].keys() if rows else []),
          f"required columns present: {sorted(REQUIRED)}")

    labels = [r.get("label", "") for r in rows]
    bad = sorted({l for l in labels if l not in VALID})
    check(not bad, f"all labels in the closed space or UNSURE"
                   + (f" — offenders {bad}" if bad else ""))

    ids = [r.get("object_id", "") for r in rows]
    check(len(set(ids)) == len(ids), "object_ids unique")

    sampled = {json.loads(l)["object_id"]
               for l in (ANN / "annotation_set.jsonl").open(encoding="utf-8")}
    stray = sorted(set(ids) - sampled)
    check(not stray, "every object_id belongs to the sampled annotation set"
                     + (f" — stray {stray[:3]}" if stray else ""))

    check(all(r.get("source") == "gold" for r in rows),
          "every row marked source=gold (INV-17: assigned by the student)")

    # advisory
    try:
        elapsed = int(rows[0].get("elapsed_seconds") or 0)
    except ValueError:
        elapsed = 0
    per = elapsed / n if n else 0
    check(per >= MIN_SEC_PER_OBJECT,
          f"effort plausibility: {elapsed}s over {n} objects = "
          f"{per:.1f}s/object (advisory floor {MIN_SEC_PER_OBJECT}s)",
          hard=False)

    unsure = sum(1 for l in labels if l == "UNSURE")
    print(f"\n  {n} rows · {n - unsure} labelled · {unsure} unsure "
          f"({100 * unsure / n:.0f}%) · {elapsed}s total")

    if fails:
        print(f"\nGATE: FAIL ({len(fails)} blocking)")
        sys.exit(1)
    print(f"\nGATE: PASS" + (f" ({len(warns)} advisory warning)" if warns else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
