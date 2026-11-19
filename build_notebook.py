"""Build MP9.ipynb, then execute it in place. The notebook re-derives every
number in report.md from the team corpus; it does not read out/results.json."""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r"""# MP 9 — Retrieval-Augmented Classification & Convolutions

**Abdulla Mansoor · CIS 731 · Track 2 (VDB)**

Every figure in `report.md` is re-derived below from the team's RateYourMusic
corpus. Nothing is read back from `out/results.json`.

**One substitution to state up front.** The team corpus is artist-level
metadata — genres, albums, years active, trope prose — and contains **no
lyrics**, contrary to the PS 4 / PS 8 problem statements. All analysis here is
therefore artist-level rather than chunk-level."""))

cells.append(code(r"""import sys, json, csv, warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import faiss
import run_mp9 as m
from mp7.corpus import LABEL_IDS, HashedNgramEncoder

print("seed       ", m.SEED)
print("bootstrap B", m.B_BOOT)
print("numpy      ", np.__version__)
print("corpus     ", m.RYM.name)"""))

# ----------------------------------------------------------------- 1. corpus
cells.append(md(r"""## 1 · Corpus

`load()` keeps artists that have both trope prose and a parseable start year,
since the cohort axis in §6 is undefined without the latter."""))

cells.append(code(r"""header = next(csv.reader(m.RYM.open(encoding="utf-8")))
print("columns in the team corpus:")
for h in header:
    print("   ", h)
print("\nany lyrics column?",
      any("lyric" in h.lower() and "trope" not in h.lower() for h in header))"""))

cells.append(code(r"""rows = m.load()
dec = Counter(r["decade"] for r in rows)
print(f"artists usable: {len(rows)}\n")
for d in sorted(dec):
    print(f"  {d}s cohort: {dec[d]:5d}")"""))

# ------------------------------------------------------- 2. weak supervision
cells.append(md(r"""## 2 · Weak supervision

Four labelling functions over the fields withheld from the annotator. Weights
are set a priori by evidence strength and are **not** tuned: an explicit
primary-genre tag is stronger evidence than a keyword in trope prose."""))

cells.append(code(r"""W = {"primary_genre": 3.0, "secondary_genre": 1.5,
     "trope_keyword": 0.75, "region_era": 0.4}
cov = {name: sum(1 for r in rows if fn(r)) for name, fn in m.LFS}

print(f"{'labelling function':20s} {'weight':>7s} {'coverage':>9s} {'%':>8s}")
for name, _ in m.LFS:
    print(f"{name:20s} {W[name]:7.2f} {cov[name]:9d} "
          f"{100*cov[name]/len(rows):7.1f}%")"""))

cells.append(code(r"""pool = []
for r in rows:
    lab, conf, fired = m.silver_label(r)
    if lab:
        pool.append({**r, "silver": lab, "silver_conf": conf})

print(f"silver pool: {len(pool)}   (requirement >= 1000: "
      f"{'PASS' if len(pool) >= 1000 else 'FAIL'})")
print(f"mean vote confidence: {np.mean([p['silver_conf'] for p in pool]):.4f}\n")
for k, v in Counter(m.VERB[p["silver"]] for p in pool).most_common():
    print(f"  {k:12s} {v:5d}  {100*v/len(pool):5.1f}%")"""))

cells.append(md(r"""The distribution is severely skewed — Trap and Boom Bap take
roughly two thirds of the pool between them, while Bounce and Hyphy together
take under 2%. That skew is the first half of the diagnosis in §4."""))

# ------------------------------------------------------------- 3. annotation
cells.append(md(r"""## 3 · Gold annotation

50 artists, stratified 10 per decade. The annotator saw artist, years active,
regional origins, albums and the four trope descriptors — and **not**
`Primary_Genres` / `Secondary_Genres`, which contain the label names verbatim
and drive the labelling functions above. Showing them would have made gold a
transcription of silver and §4 vacuous."""))

cells.append(code(r"""gold_rows = list(csv.DictReader((m.ANN / "gold_raw.csv").open(encoding="utf-8")))
el = int(gold_rows[0]["elapsed_seconds"])
unsure = [g for g in gold_rows if g["label"] == "UNSURE"]
usable_rows = [g for g in gold_rows if g["label"] != "UNSURE"]

print(f"presented   {len(gold_rows)}   (ceiling 50, INV-21)")
print(f"labelled    {len(usable_rows)}")
print(f"UNSURE      {len(unsure)}  ({100*len(unsure)/len(gold_rows):.0f}%)")
print(f"elapsed     {el} s = {el//60} min {el%60} s")
print(f"per object  {el/len(gold_rows):.1f} s")
print(f"source=gold {all(g['source']=='gold' for g in gold_rows)}  (INV-17: "
      "assigned by the student, not a model)\n")
for k, v in Counter(m.VERB[g["label"]] for g in usable_rows).most_common():
    print(f"  {k:12s} {v}")"""))

cells.append(md(r"""**A discarded first pass is disclosed.** An earlier pass over
these same 50 objects finished in 106 s (2.1 s/object). It cleared the hard
row-count gate but failed the advisory effort check, and agreed with silver at
0.10 (κ = −0.02) — chance for ten classes. It was discarded and the set fully
re-annotated. The contrast is the clearest evidence available that the effort
check is load-bearing: the hard gate alone would have accepted a pass carrying
no signal."""))

# -------------------------------------------------------- 4. silver vs gold
cells.append(md(r"""## 4 · Silver-vs-gold agreement

Inter-**annotator** κ is not computable: the spec requires a partner's
independent pass on 20 overlap objects, and no partner was assigned (the spec's
own `REVIEW R-5` records the mechanism as drafted). `overlap_ids.txt` and the
per-row `in_iaa_overlap` flag exist so a partner's file joins later without
re-deriving anything. What follows is human-vs-**function** agreement — a
weaker and different quantity, labelled as such."""))

cells.append(code(r"""adj = [json.loads(l) for l in
       (m.ANN / "gold_adjudicated.jsonl").open(encoding="utf-8")]
silver_by_id = {p["object_id"]: p["silver"] for p in pool}

pairs = [(a["label"], silver_by_id[a["object_id"]]) for a in adj
         if a["object_id"] in silver_by_id]
g, s = zip(*pairs)
agree = float(np.mean([x == y for x, y in zip(g, s)]))
lo, hi = m.boot(lambda a, b: float(np.mean(a == b)), np.array(g), np.array(s))
kappa = m.cohen_kappa(g, s, LABEL_IDS)

print(f"adjudicated rows      {len(adj)}   (UNSURE excluded by rule A1)")
print(f"n paired with pool    {len(pairs)}")
print(f"raw agreement         {agree:.4f}   95% CI [{lo:.3f}, {hi:.3f}]")
print(f"Cohen's kappa         {kappa:+.4f}   <- chance level")
print(f"IAA overlap flagged   {sum(1 for a in adj if a['in_iaa_overlap'])} rows")
print("\ntop confusions (gold -> silver):")
for (a, b), c in Counter((m.VERB[x], m.VERB[y])
                         for x, y in pairs if x != y).most_common(5):
    print(f"  {a:12s} -> {b:12s} {c}")"""))

cells.append(md(r"""### Why κ ≈ 0, and why it is not annotator error

Two measurements below locate the cause in the **label space** rather than in
the annotation."""))

cells.append(code(r"""import re
n_multi = n_none = 0
for r in rows:
    tags = f"{r['primary']} {r['secondary']}".lower()
    hits = sum(1 for pat in m.GENRE_PAT.values() if re.search(pat, tags))
    n_multi += hits >= 2
    n_none += hits == 0

print(f"carry 2+ of the 10 labels : {n_multi:5d}  {100*n_multi/len(rows):3.0f}%"
      "   <- the vote picks one, near-arbitrarily")
print(f"carry none of the 10      : {n_none:5d}  {100*n_none/len(rows):3.0f}%"
      "   <- outside the ontology entirely")

raw = Counter()
for r in rows:
    for t in re.split(r"[|;]", r["primary"]):
        if t.strip():
            raw[t.strip()] += 1

space = [v.lower() for v in m.VERB.values()]
print("\nten most common individual RYM primary tags, vs ontology v0.2.0:")
absent = 0
for t, c in raw.most_common(10):
    inside = any(w in t.lower() for w in space)
    absent += not inside
    print(f"  {t:24s} {c:5d}  {'in ontology' if inside else 'ABSENT'}")
print(f"\noutside the ontology, in the top ten: {absent} of 10")"""))

cells.append(md(r"""Most of RYM's commonest primary tags are absent from the
ontology. The silver distribution therefore tracks *tag frequency*, while gold
— judged from trope prose — is far more even, and the leading confusions are
**symmetric** (Jazz Rap↔Trap). Symmetry is the signature of two annotators
answering different questions, not of one being wrong.

Consequence for MP 10 / HW 12: the ontology must either be extended to cover
the uncovered genres, or be explicitly scoped as a *sub*-taxonomy that does not
claim to partition the corpus."""))

# ------------------------------------------------------------------- 5. RAG
cells.append(md(r"""## 5 · Retrieval-augmented classification

`IndexFlatIP` over hashed-n-gram embeddings of trope text; the query artist is
excluded from its own neighbourhood; the silver labels of the top-k neighbours
vote."""))

cells.append(code(r"""texts = [p["trope_text"] for p in pool]
enc = HashedNgramEncoder().fit(texts)
V = enc.encode(texts)
index = faiss.IndexFlatIP(V.shape[1])
index.add(V)
print(f"index: {index.ntotal} vectors, d={V.shape[1]}, "
      f"exact inner product, encoder={enc.name}")"""))

cells.append(code(r"""ids = [p["object_id"] for p in pool]
text_by_id = {p["object_id"]: p["trope_text"] for p in pool}
gold_by_id = {a["object_id"]: a["label"] for a in adj}
q_ids = [k for k in gold_by_id if k in text_by_id]
Q = enc.encode([text_by_id[k] for k in q_ids])

print(f"queries: {len(q_ids)} adjudicated objects present in the pool\n")
print(f"{'k':>3s} {'accuracy':>9s} {'95% CI':>18s} {'P@k':>8s} {'95% CI':>18s}")
for k in (3, 5, 10):
    _, idx = index.search(Q, k + 1)
    hits, patk = [], []
    for qi, qid in enumerate(q_ids):
        nb = [ids[j] for j in idx[qi] if ids[j] != qid][:k]
        votes = Counter(silver_by_id[n] for n in nb)
        hits.append(votes.most_common(1)[0][0] == gold_by_id[qid])
        patk.append(float(np.mean([silver_by_id[n] == gold_by_id[qid]
                                   for n in nb])))
    a_lo, a_hi = m.boot(lambda v: float(np.mean(v)), np.array(hits))
    p_lo, p_hi = m.boot(lambda v: float(np.mean(v)), np.array(patk))
    print(f"{k:3d} {np.mean(hits):9.4f} [{a_lo:.3f}, {a_hi:.3f}]"
          f" {np.mean(patk):8.4f} [{p_lo:.3f}, {p_hi:.3f}]")"""))

cells.append(md(r"""The votes come from **silver** labels but are scored against
**gold**, so the ceiling on this table *is* the silver-gold agreement from §4.
Retrieval is not failing — it faithfully propagates labels whose relation to
gold is already chance. All intervals overlap completely, so no k is
distinguishable from another at this n, and the fix is upstream of retrieval.

The MP 7 fixed-demonstration figure is **not** a fair comparator: MP 7 ran on a
synthetic chunk corpus with a stub scorer. It is context, never a contrast."""))

# ------------------------------------------------------------ 6. convolution
cells.append(md(r"""## 6 · Cohort convolution

Axis: `generation_cohort := start decade of Years_Active`, giving five real
cohorts (1980–2020) populated for 100% of artists. Each cohort's vector is the
normalised silver-label distribution within it."""))

cells.append(code(r"""decades = sorted({p["decade"] for p in pool})
by_dec = {d: [p for p in pool if p["decade"] == d] for d in decades}
M = np.zeros((len(LABEL_IDS), len(decades)))
for j, d in enumerate(decades):
    c = Counter(p["silver"] for p in by_dec[d])
    tot = sum(c.values()) or 1
    for i, lid in enumerate(LABEL_IDS):
        M[i, j] = c[lid] / tot

print("trope-frequency matrix (labels x cohorts)")
print(f"{'':12s}" + "".join(f"{d:>9d}" for d in decades))
for i, lid in enumerate(LABEL_IDS):
    print(f"{m.VERB[lid]:12s}" + "".join(f"{M[i,j]:9.4f}"
                                         for j in range(len(decades))))"""))

cells.append(code(r"""SMOOTH, DIFF = np.array([0.25, 0.5, 0.25]), np.array([-1.0, 1.0])

# np.convolve REVERSES its kernel, so convolve(x, [-1, 1]) computes
# x[t] - x[t+1] -- the NEGATIVE of cohort-to-cohort drift. The first version of
# this analysis used convolve, and every drift came out sign-flipped relative to
# its own bootstrap CI (Trap 1990->2000 reported -0.161 against a CI of
# [+0.106, +0.219]); that point-vs-interval disagreement is what exposed it.
# np.correlate applies the kernel as written: -x[t] + x[t+1].
sm = np.vstack([np.convolve(M[i], SMOOTH, mode="same")
                for i in range(M.shape[0])])
df = np.vstack([np.correlate(M[i], DIFF, mode="valid")
                for i in range(M.shape[0])])

print("smoothing kernel is symmetric, so convolve == correlate there:",
      np.allclose(sm, np.vstack([np.correlate(M[i], SMOOTH, mode="same")
                                 for i in range(M.shape[0])])))
print("first-difference the two ways disagree by a sign:",
      np.allclose(np.convolve(M[0], DIFF, mode="valid"),
                  -np.correlate(M[0], DIFF, mode="valid")))
print("drift matrix:", df.shape, "= 10 labels x 4 transitions")"""))

cells.append(code(r"""drifts, inconsistent = [], 0
for i, lid in enumerate(LABEL_IDS):
    for j in range(df.shape[1]):
        a = np.array([p["silver"] == lid for p in by_dec[decades[j]]], float)
        b = np.array([p["silver"] == lid for p in by_dec[decades[j+1]]], float)
        if len(a) < 3 or len(b) < 3:
            continue
        # Reseeded per transition, matching run_mp9.py exactly, so the notebook
        # and results.json report the same count rather than two defensible but
        # different ones.
        rng = np.random.default_rng(m.SEED)
        samp = [b[rng.integers(0, len(b), len(b))].mean()
                - a[rng.integers(0, len(a), len(a))].mean()
                for _ in range(m.B_BOOT)]
        lo, hi = float(np.percentile(samp, 2.5)), float(np.percentile(samp, 97.5))
        d = float(df[i, j])
        inconsistent += not (lo <= d <= hi)
        drifts.append((m.VERB[lid], decades[j], decades[j+1], d, lo, hi,
                       lo > 0 or hi < 0))

print(f"point estimate outside its own CI : {inconsistent} of {len(drifts)}")
sig = [d for d in drifts if d[6]]
print(f"significant transitions (CI excl 0): {len(sig)} of {len(drifts)}\n")
for v, d0, d1, d, lo, hi, _ in sorted(sig, key=lambda t: -abs(t[3]))[:10]:
    print(f"  {v:12s} {d0}->{d1}  {d:+.4f}  CI[{lo:+.4f},{hi:+.4f}]")"""))

cells.append(code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
for i, lid in enumerate(LABEL_IDS):
    if M[i].max() >= 0.08:
        ax.plot(decades, M[i], marker="o", lw=2, label=m.VERB[lid])
ax.set_xlabel("generation cohort (start decade of Years_Active)")
ax.set_ylabel("share of cohort")
ax.set_title("Micro-genre share by artist generation cohort (silver labels)")
ax.set_xticks(decades)
ax.grid(alpha=.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("out/cohort_trajectories.png")
plt.show()"""))

cells.append(md(r"""Trap climbs from near-absent to dominant with its largest
jump at 2000→2010; Boom Bap falls as the mirror image; Crunk peaks in the 2000s
and Memphis in the 1990s, both then declining. None of this chronology was
supplied to the pipeline — it emerges from silver labels crossed with
`Years_Active`, and its agreement with the actual history of the genre is the
main validity check available without external ground truth."""))

# --------------------------------------------------------- 7. learned kernel
cells.append(md(r"""## 7 · Learned kernel vs fixed kernels

CIS 731 requirement: a trained `Conv1d` compared against the fixed kernels.
CPU only (INV-14/15)."""))

cells.append(code(r"""import torch, torch.nn as nn
torch.manual_seed(m.SEED)
print("torch", torch.__version__, "| device cpu")

win = []
for i in range(M.shape[0]):
    for j in range(len(decades) - 3):
        win.append((M[i, j:j+3], M[i, j+3]))
Xw = torch.tensor(np.array([a for a, _ in win]), dtype=torch.float32).unsqueeze(1)
yw = torch.tensor(np.array([b for _, b in win]), dtype=torch.float32).view(-1, 1)
n_tr = int(0.7 * len(win))
print(f"training windows {n_tr}, held out {len(win)-n_tr}, "
      f"free parameters 4")

conv = nn.Conv1d(1, 1, kernel_size=3)
opt = torch.optim.Adam(conv.parameters(), lr=0.05)
for _ in range(600):
    opt.zero_grad()
    ((conv(Xw[:n_tr]).view(-1, 1) - yw[:n_tr]) ** 2).mean().backward()
    opt.step()

with torch.no_grad():
    kern = conv.weight.view(-1).numpy()
    bias = float(conv.bias.item())
    mse_learned = float(((conv(Xw[n_tr:]).view(-1, 1) - yw[n_tr:]) ** 2).mean())

Xh, yh = Xw[n_tr:].squeeze(1).numpy(), yw[n_tr:].view(-1).numpy()
mse_smooth = float(np.mean(((Xh * SMOOTH).sum(1) - yh) ** 2))
mse_persist = float(np.mean((Xh[:, -1] - yh) ** 2))

print(f"\nlearned kernel [{kern[0]:+.4f}, {kern[1]:+.4f}, {kern[2]:+.4f}]"
      f"  bias {bias:+.4f}\n")
print(f"{'predictor':34s} {'held-out MSE':>13s}")
cand = [("learned Conv1d(1,1,k=3)", mse_learned),
        ("fixed smoothing [.25,.5,.25]", mse_smooth),
        ("persistence  x[t+1] = x[t]", mse_persist)]
for nm, v in cand:
    print(f"{nm:34s} {v:13.5f}")
print(f"\nbest: {min(cand, key=lambda t: t[1])[0]}")"""))

cells.append(md(r"""**The learned kernel loses to both fixed baselines and
persistence wins outright.** This is reported as the result, not worked around.
Five cohorts × ten label series gives only a couple of dozen training windows
for a 4-parameter model, so it overfits; its weights are interpretable as
persistence plus a spurious negative lag-2 term. The honest conclusion is that
a learned kernel is **not justified at this cohort resolution** — it would need
release-year bins or many more series."""))

# ------------------------------------------------------------------ closing
cells.append(md(r"""## Findings

1. **No lyrics.** The team corpus is artist metadata; PS 4 / PS 8 describe a
   lyrics corpus. Everything here is artist-level by necessity.
2. **Inter-annotator κ is unobtainable** — no partner was assigned. The join
   path is prepared rather than a number invented.
3. **Silver-vs-gold κ ≈ 0** is an ontology/vocabulary mismatch, not annotator
   error: a third of artists carry 2+ of the ten labels, a fifth carry none,
   and most of RYM's commonest tags are outside the ontology.
4. **RAG accuracy is capped by that mismatch**, not by retrieval; no k is
   distinguishable at this n.
5. **The convolution sign bug** (`np.convolve` reverses its kernel) inverted
   every drift estimate until the point-vs-interval disagreement exposed it.
   After the fix, 0 of 40 estimates fall outside their own CI, and the recovered
   genre chronology matches the real history.
6. **The learned kernel loses** to both fixed baselines.
7. **A 106-second first annotation pass was discarded** and is disclosed in
   `attestation.md`.
8. `validate_annotation_submission.py` is my implementation; the course ships
   none."""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "MP9.ipynb")
print(f"wrote MP9.ipynb with {len(cells)} cells")
