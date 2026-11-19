"""
CIS 731 — MP 9: weak supervision, retrieval-augmented classification, and
cohort convolution over the team's real RateYourMusic corpus.

Everything here runs without gold labels. The three gold-dependent analyses
(IAA, silver-vs-gold agreement, RAG evaluation) are computed only if
annotation/gold_raw.csv exists; otherwise they are recorded as PENDING with
the reason, and running this script again once the file lands completes them
with no other change. INV-17 forbids this script from inventing those labels
and it does not.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mp7.corpus import LABELS, LABEL_IDS, HashedNgramEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parent
RYM = Path("/mnt/user-data/uploads/HWS/teamrepo/data/rym_data/"
           "master_ontology_mapped.csv")
ANN = ROOT / "annotation"
OUT = ROOT / "out"
SEED = 20261116
B_BOOT = 2000
TROPE_FIELDS = ("Production_Tropes", "Lyrical_Tropes", "Vocal_Tropes",
                "General_Tropes")

VERB = {l.id: l.verbalizer for l in LABELS}
REGION = {l.id: l.region for l in LABELS}


# --------------------------------------------------------------------- data
def start_decade(y: str) -> int | None:
    m = re.match(r"(\d{4})", (y or "").strip())
    return (int(m.group(1)) // 10) * 10 if m else None


def trope_text(r: dict) -> str:
    return " ".join((r.get(f) or "").replace("|", " ") for f in TROPE_FIELDS)


def load() -> list[dict]:
    rows = list(csv.DictReader(RYM.open(encoding="utf-8")))
    out = []
    for i, r in enumerate(rows):
        t = trope_text(r).strip()
        d = start_decade(r.get("Years_Active", ""))
        if not t or d is None:
            continue
        out.append({
            "object_id": f"rym-{i:05d}",
            "artist": (r.get("Target_Artist") or "").strip(),
            "decade": d,
            "trope_text": t,
            "primary": (r.get("Primary_Genres") or "").strip(),
            "secondary": (r.get("Secondary_Genres") or "").strip(),
            "region_txt": (r.get("Regional_Origins") or "").strip(),
        })
    return out


# ------------------------------------------------- weak supervision (silver)
# Each labelling function returns {label_id: weight} or {}. Weights encode how
# much the signal is trusted, set a priori rather than tuned: an explicit
# primary-genre tag is stronger evidence than a trope-prose keyword.
GENRE_PAT = {
    "MIC-0001": r"\btrap\b", "MIC-0002": r"\bcrunk\b",
    "MIC-0003": r"memphis", "MIC-0004": r"g-?funk",
    "MIC-0005": r"\bhyphy\b", "MIC-0006": r"boom ?bap",
    "MIC-0007": r"\bdrill\b", "MIC-0008": r"\bbounce\b",
    "MIC-0009": r"jazz rap", "MIC-0010": r"horrorcore",
}
TROPE_PAT = {
    "MIC-0001": r"hi-?hat|808", "MIC-0002": r"chant|shout",
    "MIC-0003": r"lo-?fi|murky|chopped", "MIC-0004": r"synth|smooth|funk",
    "MIC-0005": r"hyphy|thizz", "MIC-0006": r"sampl|breakbeat|boom",
    "MIC-0007": r"grim|slid", "MIC-0008": r"call.and.response|bounce",
    "MIC-0009": r"jazz|horn|upright", "MIC-0010": r"macabre|morbid|horror",
}
CITY_REGION = {
    "South": r"atlanta|memphis|new orleans|houston|miami|south|texas",
    "West Coast": r"los angeles|compton|long beach|bay area|oakland|"
                  r"vallejo|california|west coast",
    "East Coast": r"new york|brooklyn|bronx|queens|harlem|new jersey|"
                  r"philadelphia|east coast",
    "Midwest": r"chicago|detroit|midwest|st\.? louis|cleveland",
}


def lf_primary(r: dict) -> dict:
    s = r["primary"].lower()
    return {k: 3.0 for k, p in GENRE_PAT.items() if re.search(p, s)}


def lf_secondary(r: dict) -> dict:
    s = r["secondary"].lower()
    return {k: 1.5 for k, p in GENRE_PAT.items() if re.search(p, s)}


def lf_tropes(r: dict) -> dict:
    s = r["trope_text"].lower()
    return {k: 0.75 for k, p in TROPE_PAT.items() if re.search(p, s)}


def lf_region_era(r: dict) -> dict:
    s = (r["region_txt"] + " " + r["artist"]).lower()
    reg = next((g for g, p in CITY_REGION.items() if re.search(p, s)), None)
    if reg is None:
        return {}
    # Weak on its own: region narrows to 2-4 labels, so spread the mass and
    # let the genre LFs break the tie rather than guessing inside the region.
    cand = [k for k in LABEL_IDS if REGION[k] == reg]
    return {k: 0.4 for k in cand}


LFS = [("primary_genre", lf_primary), ("secondary_genre", lf_secondary),
       ("trope_keyword", lf_tropes), ("region_era", lf_region_era)]


def silver_label(r: dict) -> tuple[str | None, float, dict]:
    votes: Counter = Counter()
    fired = {}
    for name, fn in LFS:
        v = fn(r)
        if v:
            fired[name] = sorted(v)
        for k, w in v.items():
            votes[k] += w
    if not votes:
        return None, 0.0, fired
    top, score = votes.most_common(1)[0]
    total = sum(votes.values())
    return top, score / total, fired


# -------------------------------------------------------------- bootstrap CI
def boot(fn, *arrs, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(arrs[0])
    if n < 3:
        return (float("nan"), float("nan"))
    s = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        s.append(fn(*[np.asarray(a)[i] for a in arrs]))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def cohen_kappa(a, b_, labels):
    a, b_ = list(a), list(b_)
    n = len(a)
    po = sum(1 for x, y in zip(a, b_) if x == y) / n
    ca, cb = Counter(a), Counter(b_)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = load()
    res: dict = {"seed": SEED, "corpus": {
        "source": "team repo data/rym_data/master_ontology_mapped.csv "
                  "(RateYourMusic artist metadata — NOT lyrics)",
        "artists_with_tropes_and_years": len(rows),
    }}
    print(f"corpus: {len(rows)} artists")

    # ------------------------------------------------- 1. weak supervision
    pool = []
    lf_cov: Counter = Counter()
    for r in rows:
        lab, conf, fired = silver_label(r)
        for name in fired:
            lf_cov[name] += 1
        if lab:
            pool.append({**r, "silver": lab, "silver_conf": round(conf, 4)})
    res["weak_supervision"] = {
        "pool_size": len(pool),
        "meets_1000_minimum": len(pool) >= 1000,
        "labeling_functions": [
            {"name": n, "coverage": lf_cov[n],
             "coverage_pct": round(100 * lf_cov[n] / len(rows), 1)}
            for n, _ in LFS],
        "aggregation": "weighted vote; weights 3.0 primary-genre, 1.5 "
                       "secondary-genre, 0.75 trope-keyword, 0.4 region-era, "
                       "set a priori by evidence strength, not tuned",
        "silver_distribution": {VERB[k]: v for k, v in
                                Counter(p["silver"] for p in pool).most_common()},
        "mean_confidence": round(float(np.mean([p["silver_conf"]
                                                for p in pool])), 4),
    }
    print(f"silver pool: {len(pool)} (>=1000: {len(pool) >= 1000})")

    # ------------------------------------------------------ 2. FAISS index
    enc = HashedNgramEncoder().fit([p["trope_text"] for p in pool])
    V = enc.encode([p["trope_text"] for p in pool])
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(V)
    res["index"] = {"type": "IndexFlatIP", "n": int(index.ntotal),
                    "dim": int(V.shape[1]), "encoder": enc.name}

    # --------------------------------------------- 3. cohort convolution
    decades = sorted({r["decade"] for r in pool})
    M = np.zeros((len(LABEL_IDS), len(decades)))
    for j, d in enumerate(decades):
        c = Counter(p["silver"] for p in pool if p["decade"] == d)
        tot = sum(c.values()) or 1
        for i, lid in enumerate(LABEL_IDS):
            M[i, j] = c[lid] / tot
    SMOOTH, DIFF = np.array([0.25, 0.5, 0.25]), np.array([-1.0, 1.0])
    # SMOOTHING: symmetric, so convolution and correlation agree; either is fine.
    sm = np.vstack([np.convolve(M[i], SMOOTH, mode="same")
                    for i in range(M.shape[0])])
    # FIRST DIFFERENCE: np.convolve REVERSES the kernel, so np.convolve(x,
    # [-1, 1]) computes x[t] - x[t+1] -- the negative of the cohort-to-cohort
    # drift this analysis is supposed to estimate. The first version used
    # convolve and every drift estimate came out sign-flipped relative to its
    # own bootstrap CI (e.g. Trap 1990->2000 reported delta -0.16 against a CI
    # of [+0.11, +0.22]). The disagreement between point estimate and interval
    # is what exposed it. np.correlate applies the kernel as written, giving
    # -x[t] + x[t+1] = the forward difference.
    df = np.vstack([np.correlate(M[i], DIFF, mode="valid")
                    for i in range(M.shape[0])])

    drift = []
    for i, lid in enumerate(LABEL_IDS):
        for j in range(df.shape[1]):
            d0, d1 = decades[j], decades[j + 1]
            a = [1 if p["silver"] == lid else 0
                 for p in pool if p["decade"] == d0]
            b_ = [1 if p["silver"] == lid else 0
                  for p in pool if p["decade"] == d1]
            if len(a) < 3 or len(b_) < 3:
                continue
            rngl = np.random.default_rng(SEED)
            s = [float(np.mean(np.asarray(b_)[rngl.integers(0, len(b_), len(b_))])
                       - np.mean(np.asarray(a)[rngl.integers(0, len(a), len(a))]))
                 for _ in range(B_BOOT)]
            drift.append({
                "label": VERB[lid], "from": d0, "to": d1,
                "delta": round(float(df[i, j]), 5),
                "ci95": [round(float(np.percentile(s, 2.5)), 5),
                         round(float(np.percentile(s, 97.5)), 5)],
                "significant": bool(np.percentile(s, 2.5) > 0
                                    or np.percentile(s, 97.5) < 0),
            })
    res["cohort_convolution"] = {
        "axis": "generation_cohort := start decade of Years_Active",
        "cohorts": decades,
        "frequency_matrix": {VERB[LABEL_IDS[i]]: [round(x, 4) for x in M[i]]
                             for i in range(len(LABEL_IDS))},
        "smoothing_kernel": SMOOTH.tolist(),
        "difference_kernel": DIFF.tolist(),
        "smoothed": {VERB[LABEL_IDS[i]]: [round(x, 4) for x in sm[i]]
                     for i in range(len(LABEL_IDS))},
        "drift": drift,
        "n_significant": sum(1 for d in drift if d["significant"]),
    }
    print(f"cohorts: {decades} | drift estimates: {len(drift)} "
          f"({res['cohort_convolution']['n_significant']} significant)")

    # ------------------------------- 4. learned 1-D kernel (CIS 731, PyTorch)
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    X, Y = [], []
    for i in range(M.shape[0]):
        for j in range(M.shape[1] - 3):
            X.append(M[i, j:j + 3])
            Y.append(M[i, j + 3])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
    ntr = int(0.7 * len(X))
    conv = nn.Conv1d(1, 1, kernel_size=3, bias=True)
    opt = torch.optim.Adam(conv.parameters(), lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = ((conv(X[:ntr]).squeeze(-1) - Y[:ntr]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse_learned = float(((conv(X[ntr:]).squeeze(-1) - Y[ntr:]) ** 2).mean())
        k = conv.weight.detach().numpy().ravel()
    Xn = X.squeeze(1).numpy()
    Yn = Y.squeeze(1).numpy().ravel()
    mse_smooth = float(np.mean((Xn[ntr:] @ SMOOTH - Yn[ntr:]) ** 2))
    mse_persist = float(np.mean((Xn[ntr:, -1] - Yn[ntr:]) ** 2))
    res["learned_kernel"] = {
        "framework": f"PyTorch {torch.__version__} (CPU)",
        "kernel": [round(float(x), 4) for x in k],
        "bias": round(float(conv.bias.detach().numpy()[0]), 5),
        "n_windows": len(X), "n_train": ntr, "n_heldout": len(X) - ntr,
        "heldout_mse_learned": round(mse_learned, 6),
        "heldout_mse_fixed_smoothing": round(mse_smooth, 6),
        "heldout_mse_persistence": round(mse_persist, 6),
        "beats_fixed_smoothing": bool(mse_learned < mse_smooth),
        "beats_persistence": bool(mse_learned < mse_persist),
    }
    print(f"learned kernel {res['learned_kernel']['kernel']} | heldout MSE "
          f"learned {mse_learned:.5f} vs smoothing {mse_smooth:.5f} "
          f"vs persistence {mse_persist:.5f}")

    # ------------------------------------------- 5. gold-dependent analyses
    gold_path = next((p for p in (ANN / "gold_raw.csv", ANN / "gold_raw.jsonl")
                      if p.exists()), None)
    if gold_path is None:
        res["gold_dependent"] = {
            "status": "PENDING",
            "reason": "annotation/gold_raw.csv not present. INV-17 requires "
                      "gold labels to be assigned by the student; this script "
                      "will not generate them. Re-run unchanged once the file "
                      "exists and IAA, silver-vs-gold agreement and the "
                      "retrieval evaluation complete automatically.",
            "blocks": ["IAA and adjudication", "weak-supervision agreement",
                       "retrieval-augmented classification evaluation"],
        }
        print("\ngold_raw.csv ABSENT -> gold-dependent sections PENDING")
    else:
        res["gold_dependent"] = gold_analyses(gold_path, pool, enc, index, rng)

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'results.json'}")


def gold_analyses(gold_path: Path, pool, enc, index, rng) -> dict:
    rows = list(csv.DictReader(gold_path.open(encoding="utf-8")))
    gold = {r["object_id"]: r["label"] for r in rows}
    usable = {k: v for k, v in gold.items() if v != "UNSURE"}
    withheld = {json.loads(l)["object_id"]: json.loads(l)
                for l in (ANN / "_withheld_genres.jsonl").open()}
    silver_by_id = {p["object_id"]: p["silver"] for p in pool}

    out: dict = {
        "status": "COMPLETE",
        "n_annotated": len(gold), "n_usable": len(usable),
        "unsure_rate": round(1 - len(usable) / len(gold), 4) if gold else None,
        "row_count_gate": {"rows": len(rows), "ceiling": 50,
                           "passes": len(rows) <= 50},
        "elapsed_seconds": int(rows[0].get("elapsed_seconds", 0)) if rows else 0,
    }

    # silver vs gold
    both = [(usable[k], silver_by_id[k]) for k in usable if k in silver_by_id]
    if both:
        g, s = zip(*both)
        agree = float(np.mean([x == y for x, y in zip(g, s)]))
        lo, hi = boot(lambda a, b_: float(np.mean(a == b_)),
                      np.array(g), np.array(s))
        out["silver_vs_gold"] = {
            "n": len(both), "agreement": round(agree, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "cohen_kappa": round(cohen_kappa(g, s, LABEL_IDS), 4),
            "confusion_top": Counter(f"{VERB[x]}->{VERB[y]}"
                                     for x, y in both if x != y).most_common(6),
        }

    # retrieval-augmented classification
    rag = []
    ids = [p["object_id"] for p in pool]
    texts = {p["object_id"]: p["trope_text"] for p in pool}
    q_ids = [k for k in usable if k in texts]
    if q_ids:
        Q = enc.encode([texts[k] for k in q_ids])
        for k in (3, 5, 10):
            sims, idx = index.search(Q, k + 1)
            preds, patk = [], []
            for qi, qid in enumerate(q_ids):
                nb = [ids[j] for j in idx[qi] if ids[j] != qid][:k]
                votes = Counter(silver_by_id[n] for n in nb)
                preds.append(votes.most_common(1)[0][0] if votes else None)
                patk.append(np.mean([silver_by_id[n] == usable[qid]
                                     for n in nb]) if nb else 0.0)
            acc = float(np.mean([p == usable[q]
                                 for p, q in zip(preds, q_ids)]))
            lo, hi = boot(lambda a: float(np.mean(a)),
                          np.array([p == usable[q]
                                    for p, q in zip(preds, q_ids)]))
            plo, phi = boot(lambda a: float(np.mean(a)), np.array(patk))
            rag.append({"k": k, "accuracy": round(acc, 4),
                        "acc_ci95": [round(lo, 4), round(hi, 4)],
                        "p_at_k": round(float(np.mean(patk)), 4),
                        "p_at_k_ci95": [round(plo, 4), round(phi, 4)],
                        "n": len(q_ids)})
    out["retrieval_augmented"] = rag
    return out


if __name__ == "__main__":
    main()
