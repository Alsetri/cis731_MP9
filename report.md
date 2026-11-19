# MP 9 — Retrieval-Augmented Classification & Convolutions

**Abdulla Mansoor · CIS 731 · abdulla03@ksu.edu · Track 2 (VDB) · Tag `mp9-submission`**

Reproduce: `python3 run_mp9.py` then `python3 make_deliverables.py`. Gate: `python3 validate_annotation_submission.py`.

**Corpus change since PS 8.** This milestone is the first to run on the team's real data rather than a synthetic fixture. That data turned out not to be what PS 4 and PS 8 described: it is **RateYourMusic artist metadata — 1,443 rows of genres, albums, years active and trope prose, 1,433 of them usable here — and contains no lyrics at all.** Every analysis below is therefore artist-level, not chunk-level. The substitution is named here rather than absorbed silently, because it changes what the pipeline can claim.

**Two course artifacts the spec names do not exist**, consistent with its own `REVIEW R-5` ("drafted mechanisms"): there is no `validate_annotation_submission` script and no annotation-partner assignment in the course repository. I implemented the former; the latter is addressed in §2.

## Codebook and Attestation

`codebook.md` extends the PS 6 three-point relevance protocol into a label-assignment codebook: a closed 10-label space from ontology v0.2.0, a three-step decision rule (regional origin → era → trope cues), five named boundary cases, and an explicit policy that `UNSURE` is a recorded outcome rather than a failure.

**The central design decision is what the annotator was *not* shown.** RYM's `Primary_Genres` / `Secondary_Genres` fields literally contain the label names, and those same fields drive the weak-supervision labelling functions in §3. Had the annotator seen them, gold would have been a transcription of silver, their agreement would have been ≈1.0 by construction, and §3 would have measured nothing. Hiding them makes gold an independent human judgement from trope prose, which turns the agreement question into a real one.

| Attestation | |
|---|---|
| Objects presented | 50 (ceiling 50, INV-21) |
| Labelled / `UNSURE` | 40 / 10 (20%) |
| Elapsed | 968 s = 16 min 8 s |
| Mean per object | **19.4 s** |
| Gate | **PASS** — 9/9 checks |

Timestamps are captured by the labelling instrument at first keypress and at completion; they are not self-reported.

**A first pass is disclosed and discarded.** An earlier pass over the same 50 objects completed in **106 s (2.1 s/object)**. It passed the hard row-count gate but failed the advisory effort check, and its labels agreed with silver at 0.10 (κ = −0.02) — exactly chance for ten classes. It was discarded and the objects fully re-annotated. This is reported rather than omitted because the contrast is the clearest available evidence that the effort check is load-bearing: the hard gate alone would have accepted a pass that carried no signal.

## Inter-Annotator Agreement and Adjudication

**Inter-annotator κ could not be computed, and I will not substitute a number that looks like one.** The spec requires 20 objects labelled independently by an assigned partner; no partner was assigned, and the spec's own metadata records the mechanism as drafted. `annotation/overlap_ids.txt` (20 ids) exists and `gold_adjudicated.jsonl` carries an `in_iaa_overlap` flag on each row, so a partner's file joins without re-deriving anything.

What is reported instead, clearly labelled as a *different* quantity: **human-vs-function agreement** (§3). That is not a substitute for human-vs-human reliability — it measures whether one annotator agrees with an automated heuristic, which is a weaker and different claim.

**Adjudication rules**, applied mechanically:

- **A1** — `UNSURE` rows are excluded, not resolved later. Resolving one's own abstention after the fact is a second annotation pass without the first's discipline.
- **A2** — no label was changed. With one annotator there is no disagreement to adjudicate; any edit would be the annotator overruling himself with no new evidence. All 40 rows carry `A2-unchanged-single-annotator`.
- **A3** — overlap membership is recorded per row for later joining.

## Weak-Supervision Agreement Analysis

Four labelling functions over the withheld fields, aggregated by weighted vote. Weights are set a priori by evidence strength, not tuned: an explicit primary-genre tag is stronger evidence than a keyword in trope prose.

| Labelling function | Signal | Weight | Coverage |
|---|---|---|---|
| `primary_genre` | regex over `Primary_Genres` | 3.0 | 859 (59.9%) |
| `secondary_genre` | regex over `Secondary_Genres` | 1.5 | 808 (56.4%) |
| `trope_keyword` | regex over the four trope fields | 0.75 | 1,151 (80.3%) |
| `region_era` | city/region regex, mass spread over in-region labels | 0.4 | 682 (47.6%) |

**Silver pool: 1,309 artists** (requirement ≥ 1,000). Distribution is heavily skewed — Trap 494, Boom Bap 397, versus Bounce 9 and Hyphy 11.

**Silver vs gold, on the 36 adjudicated objects present in the pool: agreement 0.194 (95% CI [0.083, 0.333]), Cohen's κ = −0.016.** That is chance.

**The failure modes, which are the finding.** The near-zero κ is not noise in the annotation; it is a structural mismatch between two label spaces:

1. **34% of artists carry two or more of the ten labels** in their RYM tags ("Cloud Rap; Horrorcore; Trap"). The vote has to pick one and the tie is close to arbitrary.
2. **18% carry none of them**, and **7 of the 10 most common individual RYM primary tags are absent from ontology v0.2.0 entirely** — Gangsta Rap (350 artists), Hardcore Hip Hop (292), Pop Rap (236), plain Hip Hop (220), Conscious Hip Hop (208), Abstract Hip Hop (203), Experimental Hip Hop (169). Only Trap, Boom Bap and Jazz Rap are shared vocabulary.
3. The resulting silver distribution tracks *tag frequency*; the gold distribution, judged from trope prose, is far more even. The top confusions are symmetric (Jazz Rap↔Trap, 3 each), the signature of two annotators answering different questions rather than one being wrong.

The actionable consequence for MP 10 and HW 12: the ontology must either be extended to cover those seven uncovered genres, or be explicitly scoped as a *sub*-taxonomy that does not attempt to partition the corpus. Reporting a flattering agreement number would have hidden a decision the project has to make.

## Retrieval-Augmented Classification Results

Exemplars are retrieved from a FAISS `IndexFlatIP` index over trope-text embeddings (1,309 vectors, d = 384), the query object excluded, and the silver labels of the top-k neighbours vote.

**The embedding model is a stand-in and is labelled as one in the code.** `all-MiniLM-L6-v2` is unreachable from this environment — HuggingFace returns 403 through the proxy — so the encoder is a hashed character-n-gram projection to 384 dimensions, identified in `results.json` as `hashed-char-ngram-svd-384 (OFFLINE STAND-IN)`. It is lexical, not semantic, so it will underperform a sentence encoder on paraphrase. That weakens this section, but it does not explain the result below, whose ceiling is set by §3 and would bind for any encoder.

| k | Accuracy | 95% CI | P@k | 95% CI |
|---|---|---|---|---|
| 3 | 0.194 | [0.083, 0.333] | 0.176 | [0.065, 0.296] |
| 5 | 0.139 | [0.028, 0.250] | 0.167 | [0.061, 0.283] |
| 10 | 0.194 | [0.083, 0.333] | 0.183 | [0.072, 0.303] |

**The MP 7 fixed-demonstration baseline is not a fair comparator here and is not claimed as one.** MP 7 ran on a synthetic chunk corpus with a stub scorer; this runs on real artist metadata. Comparing the two numbers would be comparing two different datasets, so the MP 7 figure is reported as context only, never as a contrast.

All three settings sit at the level implied by §3: retrieval is evaluated *against the gold labels*, but votes come from *silver* labels, so the ceiling on this table is the silver-gold agreement itself (0.194). The retrieval is not failing — it faithfully propagates labels whose relationship to gold is already chance. Intervals overlap completely across k, so no k is distinguishable from another at n = 36. The honest reading is that **this evaluation is uninformative until the label-space mismatch in §3 is resolved**, and the fix is upstream of retrieval.

## Cohort Convolution Analysis

**Axis:** `generation_cohort := start decade of Years_Active`, giving five real cohorts (1980–2020), populated for 100% of artists. Per-cohort trope-frequency vectors are the normalised silver-label distribution within each cohort.

**Kernels:** smoothing `[0.25, 0.5, 0.25]`; first difference `[-1, 1]`; drift CIs by bootstrap over artists within cohort, B = 2000.

**A sign bug worth recording.** `np.convolve` reverses its kernel, so `convolve(x, [-1, 1])` computes `x[t] − x[t+1]` — the negative of cohort-to-cohort drift. The first run reported every drift with the wrong sign, e.g. Trap 1990→2000 as −0.161 against its own bootstrap CI of [+0.106, +0.219]. The disagreement between point estimate and interval is what exposed it; `np.correlate` applies the kernel as written. After the fix, **all 40 point estimates fall inside their own intervals.**

**17 of 40 cohort transitions are significant** (CI excludes zero). The recovered trajectories are historically coherent, which is the main validity check available without external ground truth:

| Label | 1980 | 1990 | 2000 | 2010 | 2020 |
|---|---|---|---|---|---|
| Trap | 0.071 | 0.026 | 0.187 | 0.563 | 0.636 |
| Boom Bap | 0.762 | 0.423 | 0.500 | 0.168 | 0.179 |

Trap rises from near-absent to dominant with its largest jump at 2000→2010 (+0.376, CI [+0.307, +0.444]); Boom Bap declines by the mirror image. Crunk peaks in the 2000s and falls (2000→2010: −0.056, CI [−0.091, −0.026]); Memphis peaks in the 1990s and falls. None of this was supplied to the pipeline — it emerges from silver labels over `Years_Active`.

**Learned kernel (CIS 731, PyTorch 2.15.0, CPU).** A 1-D `Conv1d(1,1,k=3)` trained on sliding windows over the cohort series, 70/30 split.

| Predictor | Held-out MSE |
|---|---|
| Learned kernel `[−0.389, 0.002, 1.088]` | 0.00225 |
| Fixed smoothing `[0.25, 0.5, 0.25]` | 0.00104 |
| Persistence (`x[t+1] = x[t]`) | **0.00029** |

**The learned kernel loses to both fixed baselines, and persistence wins outright.** This is the expected result and is reported as such: five cohorts × ten label series yields only ~20 training windows for a 4-parameter model, so the learned kernel overfits. Its weights are also interpretable — it has essentially rediscovered persistence with a spurious negative lag-2 term. The correct conclusion is that **a learned kernel is not justified at this cohort resolution**; it would need either finer bins (release-year rather than decade) or many more series.

## GenAI Audit

Yes — the Generative AI Kit was used. **`GenAI.pdf` is included.** Gold labels were assigned by me personally and by no model (INV-17); the assistant drafted the codebook wording and built the labelling instrument, and explicitly declined, on request, to assign labels. Defects found by running the work rather than reading it — the convolution sign inversion, the discarded 106-second pass, and the label-space mismatch behind the near-zero κ — are documented above and in `GenAI.pdf`.
