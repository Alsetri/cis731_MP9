"""
CIS 731 — MP 7: the three classifiers behind one interface, with deal contracts.

The contracts are post-conditions on predict_proba, as protocol.md §5 locked:
non-negative, rows sum to 1.0 within 1e-6, exact output shape, and every
predicted label strictly inside the declared space. They are attached to the
BASE-CLASS wrapper, so all three implementations are bound by them and a new
classifier cannot be added without inheriting them.
"""

from __future__ import annotations

from typing import Sequence

import deal
import faiss
import numpy as np

from .corpus import VERBALIZERS, Encoder
from .lm import parse_label, render_prompt, scan_label_leakage

TOL = 1e-6


def _rows_sum_to_one(p: np.ndarray) -> bool:
    return bool(np.all(np.abs(p.sum(axis=1) - 1.0) <= TOL))


class Classifier:
    """Unified interface. Subclasses implement `_proba`, never `predict_proba`."""

    name: str

    def __init__(self, labels: tuple[str, ...]):
        self.labels = tuple(labels)

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "Classifier":
        return self

    def _proba(self, X: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # The contract lives here so every subclass inherits it.
    @deal.ensure(
        lambda self, X, result: result.shape == (len(X), len(self.labels)),
        message="predict_proba must return (n_samples, n_labels)")
    @deal.ensure(
        lambda self, X, result: bool(np.all(result >= 0.0)),
        message="probabilities must be non-negative")
    @deal.ensure(
        lambda self, X, result: _rows_sum_to_one(result),
        message="each row of predict_proba must sum to 1.0 within 1e-6")
    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return np.asarray(self._proba(X), dtype="float64")

    # TEMPLATE METHOD, AND THE REASON FOR IT.
    #
    # A deal post-condition is attached to the function object, so a subclass
    # that overrides `predict` silently replaces the contract along with the
    # body. An earlier version of this file decorated `predict` and let
    # subclasses override it; the notebook's deliberate out-of-space
    # classifier was then ACCEPTED, and FewShotLLMClassifier -- which does
    # override predict -- was running with no label-space guard at all.
    #
    # `predict` is now final and subclasses override `_predict`, so the
    # closed-set contract is enforced at a point no subclass can step around.
    @deal.ensure(
        lambda self, X, result: all(l in self.labels for l in result),
        message="every predicted label must belong to the declared space")
    @deal.ensure(
        lambda self, X, result: len(result) == len(X),
        message="one prediction per input")
    def predict(self, X: Sequence[str]) -> list[str]:
        return self._predict(X)

    def _predict(self, X: Sequence[str]) -> list[str]:
        idx = self.predict_proba(X).argmax(axis=1)
        return [self.labels[i] for i in idx]


def _softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype="float64") / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# (i) Embedding prototype
# ---------------------------------------------------------------------------

class PrototypeClassifier(Classifier):
    """Mean exemplar embedding per class; cosine similarity to each prototype.

    Prototypes are re-normalized after averaging: the mean of unit vectors is
    not a unit vector, and without renormalizing, a class whose exemplars are
    tightly clustered would get a systematically larger inner product than a
    diffuse class purely because its mean is longer. That would be a
    popularity artefact, not similarity.
    """

    name = "prototype"

    def __init__(self, labels, encoder: Encoder, temp: float = 0.05):
        super().__init__(labels)
        self.enc, self.temp = encoder, temp
        self.P: np.ndarray | None = None

    def fit(self, X, y):
        V = self.enc.encode(list(X))
        P = np.zeros((len(self.labels), V.shape[1]), dtype="float32")
        for i, lab in enumerate(self.labels):
            m = np.array([yy == lab for yy in y])
            if m.any():
                P[i] = V[m].mean(axis=0)
        n = np.linalg.norm(P, axis=1, keepdims=True)
        self.P = P / np.where(n == 0, 1.0, n)
        return self

    def _proba(self, X):
        return _softmax(self.enc.encode(list(X)) @ self.P.T, self.temp)


# ---------------------------------------------------------------------------
# (ii) kNN over FAISS IndexFlatIP
# ---------------------------------------------------------------------------

class KNNClassifier(Classifier):
    """k-nearest-neighbour vote over the exemplar index, similarity-weighted.

    IndexFlatIP is exact (PS 6 §2), so there is no recall parameter confounding
    the classifier comparison: any difference from the prototype classifier is
    the decision rule, not an approximate index.
    """

    name = "knn"

    def __init__(self, labels, encoder: Encoder, k: int = 15):
        super().__init__(labels)
        self.enc, self.k = encoder, k
        self.index: faiss.Index | None = None
        self._y: list[str] = []

    def fit(self, X, y):
        V = self.enc.encode(list(X))
        self.index = faiss.IndexFlatIP(V.shape[1])
        self.index.add(V)
        self._y = list(y)
        return self

    def _proba(self, X):
        Q = self.enc.encode(list(X))
        k = min(self.k, self.index.ntotal)
        sims, idx = self.index.search(Q, k)
        out = np.zeros((len(X), len(self.labels)), dtype="float64")
        pos = {l: i for i, l in enumerate(self.labels)}
        for r in range(len(X)):
            for s, j in zip(sims[r], idx[r]):
                if j < 0:
                    continue
                # Clip to 0: a negative inner product is evidence against, and
                # a negative vote weight would let a dissimilar neighbour
                # subtract mass and break non-negativity.
                out[r, pos[self._y[j]]] += max(float(s), 0.0)
        tot = out.sum(axis=1, keepdims=True)
        # A row with no positive neighbour falls back to uniform rather than
        # dividing by zero -- an honest "no information", still a valid
        # distribution.
        return np.where(tot > 0, out / np.where(tot == 0, 1, tot),
                        1.0 / len(self.labels))


# ---------------------------------------------------------------------------
# (iii) Fixed-demonstration few-shot LLM
# ---------------------------------------------------------------------------

class FewShotLLMClassifier(Classifier):
    """Class-balanced fixed demonstrations, permuted per seed.

    Implements the three pre-registered controls from protocol.md §8:
      * majority-label bias  -> class-balanced demos + null-prompt calibration
      * position bias        -> demonstration order permuted by the seed
      * label leakage        -> exemplar/eval disjoint, verbalizer scan on demos
    """

    name = "fewshot_llm"

    def __init__(self, labels, backend, k: int, seed: int,
                 calibrate: bool = True):
        super().__init__(labels)
        self.backend, self.k, self.seed = backend, k, seed
        self.calibrate = calibrate
        self.verbs = [VERBALIZERS[l] for l in self.labels]
        self.demos: list[tuple[str, str]] = []
        self.leakage: list[str] = []
        self._null: np.ndarray | None = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X, y = list(X), list(y)
        chosen: list[tuple[str, str]] = []
        if self.k > 0:
            # Class-balanced: round-robin over labels until k*n_labels demos.
            by_lab = {l: [x for x, yy in zip(X, y) if yy == l]
                      for l in self.labels}
            for _ in range(self.k):
                for l in self.labels:
                    pool = by_lab[l]
                    if pool:
                        chosen.append((pool[rng.integers(len(pool))],
                                       VERBALIZERS[l]))
            # POSITION-BIAS CONTROL: the seed decides the order, so the five
            # seeds give five distinct orderings of the same demo budget.
            order = rng.permutation(len(chosen))
            chosen = [chosen[i] for i in order]
        self.demos = chosen
        # LABEL-LEAKAGE CONTROL: verified, not assumed.
        self.leakage = scan_label_leakage(chosen)
        if self.leakage:
            raise ValueError(
                f"label leakage: {len(self.leakage)} demonstration(s) contain "
                f"a verbalizer in their input text")
        if self.calibrate:
            # MAJORITY-LABEL-BIAS CONTROL: score a content-free query to
            # measure the harness's prior over verbalizers, then subtract it.
            null = render_prompt("N/A", self.demos, self.verbs)
            self._null = self.backend.score_labels(null, self.verbs)
        return self

    def _proba(self, X):
        rows = []
        for q in X:
            s = self.backend.score_labels(
                render_prompt(q, self.demos, self.verbs), self.verbs)
            if self._null is not None:
                s = s - self._null
            rows.append(s)
        return _softmax(np.vstack(rows), temp=0.05)

    def _predict(self, X):
        # Overrides the template hook, NOT `predict`, so the inherited
        # closed-set post-condition still runs on whatever this returns.
        # Routed through the parser so membership is proved on the verbalizer
        # string a model would actually emit, not merely on an array index.
        idx = self.predict_proba(X).argmax(axis=1)
        return [parse_label(self.verbs[i], self.labels) for i in idx]
