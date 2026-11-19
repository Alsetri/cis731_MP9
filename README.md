# CIS 731 — MP 9: Retrieval-Augmented Classification & Convolutions

**Abdulla Mansoor** · abdulla03@ksu.edu · Track 2 (VDB) · tag `mp9-submission`

## Reproduce

```bash
pip install numpy faiss-cpu torch matplotlib reportlab nbformat nbconvert
python3 validate_annotation_submission.py   # annotation gate, 9 checks
python3 make_deliverables.py                # attestation.md + gold_adjudicated.jsonl
python3 run_mp9.py                          # out/results.json
python3 build_notebook.py                   # MP9.ipynb (then execute it)
python3 build_genai.py                      # GenAI.pdf
```

`run_mp9.py` reads the team corpus at the path in its `RYM` constant — the
private team repo's `data/rym_data/master_ontology_mapped.csv`. The corpus is
not redistributed here.

## Files

| Path | What it is |
|---|---|
| `report.md` | The graded report, six required sections |
| `MP9.ipynb` | Executed notebook; re-derives every reported number from the corpus |
| `GenAI.pdf` | Generative AI audit log |
| `codebook.md` | Label definitions, decision rules, boundary cases, `UNSURE` policy |
| `attestation.md` | Annotation timestamps, effort ceiling, discarded-pass disclosure |
| `annotation/gold_raw.csv` | **My** 50 annotations (INV-17 — no model assigned these) |
| `annotation/gold_adjudicated.jsonl` | 40 rows after rules A1–A3 |
| `annotation/annotation_set.jsonl` | The 50 sampled objects as presented |
| `annotation/_withheld_genres.jsonl` | The genre fields hidden from the annotator |
| `annotation/overlap_ids.txt` | 20 ids reserved for a partner's IAA pass |
| `annotation/label_tool.html` | The offline labelling instrument |
| `validate_annotation_submission.py` | My implementation of the gate; the course ships none |
| `out/results.json` | All measured outputs |
| `out/cohort_trajectories.png` | Micro-genre share by generation cohort |

## Three things a grader should know before reading the numbers

1. **The team corpus contains no lyrics.** It is RateYourMusic artist
   metadata. PS 4 and PS 8 describe a lyrics corpus. Everything here is
   artist-level, not chunk-level.
2. **Inter-annotator κ is not reported.** No partner was assigned (the spec's
   own `REVIEW R-5` records the mechanism as drafted). `overlap_ids.txt` and
   the per-row `in_iaa_overlap` flag are shipped so a partner's file joins
   later. No proxy was substituted for the missing number.
3. **The embedding model is a stand-in.** HuggingFace is unreachable from the
   build environment (403 via proxy), so the encoder is a hashed
   character-n-gram projection, labelled `OFFLINE STAND-IN` in `results.json`.

Findings, including the ones that do not flatter the work — silver-vs-gold
κ ≈ 0, the learned kernel losing to both fixed baselines, and a convolution
sign bug caught mid-analysis — are in `report.md` and `GenAI.pdf`.
