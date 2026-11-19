"""
CIS 731 — MP 7: evaluation source separation (INV-12) and bootstrap CIs.

Two functions that are deliberately NOT interchangeable.

`compute_graded_metric` reads only the instructor-provided evaluation set and
raises if it is absent. It has no fallback path. That is the whole point: a
metric function that quietly degrades to a dev split when the gold file is
missing is exactly the failure INV-12 exists to prevent, and it would be
indistinguishable from the real thing in a report.

`compute_dev_metric` stamps grading_eligible=False on everything it returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

EVAL_PATH = Path("data/eval_mp7.jsonl")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261103


class GradedEvalUnavailable(FileNotFoundError):
    """Raised when the instructor evaluation set is not present."""


def macro_f1(y_true: list[str], y_pred: list[str],
             labels: tuple[str, ...]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels),
                          average="macro", zero_division=0))


def bootstrap_ci(y_true: list[str], y_pred: list[str],
                 labels: tuple[str, ...],
                 b: int = BOOTSTRAP_B,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile bootstrap over EVALUATION ITEMS (protocol.md §4).

    Resampling items, not seeds: the interval answers "how much would this
    score move on a different sample of the same size from this population",
    which is the question a CI on a held-out score should answer. Resampling
    seeds would instead measure run-to-run variance, which is already
    reported separately as the across-seed std.
    """
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(b, dtype="float64")
    for i in range(b):
        s = rng.integers(0, n, n)
        stats[i] = f1_score(yt[s], yp[s], labels=list(labels),
                            average="macro", zero_division=0)
    return (float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def per_class_pr(y_true, y_pred, labels):
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    return [{"label": l, "precision": round(float(p[i]), 4),
             "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
             "support": int(sup[i])} for i, l in enumerate(labels)]


def _score(y_true, y_pred, labels, eval_source, grading_eligible):
    lo, hi = bootstrap_ci(y_true, y_pred, labels)
    return {
        "macro_f1": round(macro_f1(y_true, y_pred, labels), 4),
        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
        "n": len(y_true),
        "eval_source": eval_source,
        "grading_eligible": grading_eligible,
        "bootstrap": {"method": "percentile", "B": BOOTSTRAP_B,
                      "resampled": "evaluation items",
                      "seed": BOOTSTRAP_SEED},
    }


def compute_graded_metric(y_true, y_pred, labels,
                          eval_path: Path = EVAL_PATH) -> dict:
    """INV-12 graded path. Instructor evaluation set only. No fallback."""
    if not eval_path.exists():
        raise GradedEvalUnavailable(
            f"{eval_path} not found. INV-12 permits graded metrics only "
            "against the instructor-provided evaluation set. Refusing to "
            "score; use compute_dev_metric and report as advisory."
        )
    return _score(y_true, y_pred, labels, str(eval_path), True)


def compute_dev_metric(y_true, y_pred, labels,
                       eval_source: str = "self_dev_split") -> dict:
    """Advisory path. Always grading_eligible=False, no exceptions."""
    return _score(y_true, y_pred, labels, eval_source, False)


def load_graded_eval(eval_path: Path = EVAL_PATH):
    if not eval_path.exists():
        raise GradedEvalUnavailable(f"{eval_path} not found")
    rows = [json.loads(l) for l in eval_path.open(encoding="utf-8")]
    return ([r["chunk_id"] for r in rows], [r["text"] for r in rows],
            [r["label"] for r in rows])
