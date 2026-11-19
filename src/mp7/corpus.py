"""
CIS 731 — MP 7: label space, encoders, and the exemplar/evaluation corpus.

HONESTY NOTE — SYNTHETIC CORPUS
-------------------------------
As of 3 Nov 2026 the team has not obtained the lyric corpus (Lucas, team
channel, 19 Oct: "we're still working on obtaining the dataset"), and PS 6
flagged that the planned Genius/MusicBrainz/Spotify sources are Tier 2 and
would be refused by my own INV-9 licence gate. This module therefore builds a
SEEDED SYNTHETIC corpus with label-conditional vocabulary, exactly as the PS 5
replay fixture did and with the same disclosure.

What that means for the results: the classifiers, the index, the contracts,
the sweep and the statistics are all real and all execute. The ABSOLUTE
macro-F1 values are properties of this generator, not of regional hip-hop.
Only the relative comparison between the three classifiers, under an
identical corpus, carries any methodological weight -- and even that is
advisory under INV-12.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# 1. Label space (locked in protocol.md §1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Label:
    id: str
    verbalizer: str
    region: str
    era: str


LABELS: tuple[Label, ...] = (
    Label("MIC-0001", "Trap",       "South",      "2000s-2020s"),
    Label("MIC-0002", "Crunk",      "South",      "1990s-2000s"),
    Label("MIC-0003", "Memphis",    "South",      "1990s-2000s"),
    Label("MIC-0004", "G-Funk",     "West Coast", "1990s-2000s"),
    Label("MIC-0005", "Hyphy",      "West Coast", "2000s-2010s"),
    Label("MIC-0006", "Boom Bap",   "East Coast", "1990s-2000s"),
    Label("MIC-0007", "Drill",      "Midwest",    "2010s-2020s"),
    Label("MIC-0008", "Bounce",     "South",      "1990s-2010s"),
    Label("MIC-0009", "Jazz Rap",   "East Coast", "1990s-2000s"),
    Label("MIC-0010", "Horrorcore", "Midwest",    "1990s-2000s"),
)

LABEL_IDS: tuple[str, ...] = tuple(l.id for l in LABELS)
VERBALIZERS: dict[str, str] = {l.id: l.verbalizer for l in LABELS}
BY_VERBALIZER: dict[str, str] = {l.verbalizer.lower(): l.id for l in LABELS}

# Vocabulary is layered so the task is HARD and, specifically, hard in the way
# the real task is hard: confusable WITHIN a region.
#
# Two earlier versions were wrong and both were caught by running the sweep.
#   (1) Distinctive per-label vocabularies gave prototype and kNN macro-F1 of
#       exactly 1.0000. A benchmark every classifier saturates compares
#       nothing, so the comparison would have been vacuous.
#   (2) The per-label terms included the verbalizers themselves ("memphis",
#       "crunk", "drill", "hyphy", "bounce"), so a demonstration contained its
#       own answer. The pre-registered label-leakage control (protocol.md §8)
#       refused to fit and reported five offending demonstrations.
#
# No term below is a verbalizer or a substring-token of one.
REGION_TERMS: dict[str, list[str]] = {
    "South":      ["south", "atlanta", "dirty", "candy", "trill"],
    "West Coast": ["west", "coast", "cali", "lowrider", "palm"],
    "East Coast": ["east", "brooklyn", "bronx", "subway", "winter"],
    "Midwest":    ["midwest", "chicago", "detroit", "cold", "steel"],
}

SIGNAL: dict[str, list[str]] = {
    "MIC-0001": ["808", "slide", "triplet", "hihat"],
    "MIC-0002": ["chant", "shout", "stomp", "club"],
    "MIC-0003": ["lowfi", "tape", "murky", "chopped"],
    "MIC-0004": ["synth", "whine", "smooth", "ride"],
    "MIC-0005": ["ghostride", "thizz", "slap", "sideshow"],
    "MIC-0006": ["sample", "snare", "crate", "loop"],
    "MIC-0007": ["sliding", "pattern", "grim", "opp"],
    "MIC-0008": ["triggerman", "callresponse", "fast", "parade"],
    "MIC-0009": ["upright", "horn", "swing", "mellow"],
    "MIC-0010": ["macabre", "wicked", "crypt", "clown"],
}

REGION_OF: dict[str, str] = {l.id: l.region for l in LABELS}

# Shared filler present in every class: without it, mere presence of any token
# would identify the label and the task would be lookup, not classification.
FILLER = ["flow", "bars", "verse", "hook", "beat", "track", "crew", "block",
          "night", "street", "city", "tape", "mic", "studio", "record"]

# Deliberately imbalanced, so macro-F1 and micro-F1 differ and the choice of
# macro-F1 in protocol.md §4 is doing real work.
CLASS_WEIGHTS = np.array([0.20, 0.14, 0.12, 0.10, 0.09,
                          0.09, 0.08, 0.07, 0.06, 0.05])


def _chunk_id(split: str, i: int) -> str:
    return hashlib.sha256(f"mp7|{split}|{i}".encode()).hexdigest()


@dataclass(frozen=True)
class Corpus:
    chunk_ids: list[str]
    texts: list[str]
    labels: list[str]
    split: str


def _make_split(split: str, n: int, seed: int, signal_rate: float) -> Corpus:
    rng = np.random.default_rng(seed)
    ids, texts, ys = [], [], []
    for i in range(n):
        lab = LABEL_IDS[rng.choice(len(LABEL_IDS), p=CLASS_WEIGHTS)]
        n_tok = int(rng.integers(14, 32))
        n_sig = max(1, int(round(n_tok * signal_rate)))
        # Most of the "signal" is REGIONAL, shared with sibling labels, and
        # only a minority is label-distinctive. That is what makes
        # within-region pairs (Trap/Crunk/Memphis/Bounce) genuinely confusable
        # instead of separable by a single keyword.
        n_lab = max(1, int(round(n_sig * 0.40)))
        n_reg = n_sig - n_lab
        reg = REGION_TERMS[REGION_OF[lab]]
        toks = [SIGNAL[lab][rng.integers(len(SIGNAL[lab]))] for _ in range(n_lab)]
        toks += [reg[rng.integers(len(reg))] for _ in range(n_reg)]
        toks += [FILLER[rng.integers(len(FILLER))] for _ in range(n_tok - n_sig)]
        rng.shuffle(toks)
        ids.append(_chunk_id(split, i))
        texts.append(" ".join(toks))
        ys.append(lab)
    return Corpus(ids, texts, ys, split)


def build_corpus(seed: int = 20261103) -> tuple[Corpus, Corpus]:
    """Exemplar pool and development evaluation split.

    Disjointness is by construction (different `split` string feeds the
    chunk_id digest) and is asserted by the caller -- protocol.md §8 names
    exemplar/eval overlap as the label-leakage failure mode, so it is checked
    rather than assumed.
    """
    exemplars = _make_split("exemplar", 1200, seed, signal_rate=0.22)
    dev_eval = _make_split("dev_eval", 400, seed + 1, signal_rate=0.22)
    return exemplars, dev_eval


# ---------------------------------------------------------------------------
# 2. Encoders
# ---------------------------------------------------------------------------

class Encoder:
    """Protocol: text -> unit-normalized float32 matrix of shape (n, dim)."""
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SentenceTransformerEncoder(Encoder):
    """The encoder PS 6 pinned. Preferred whenever it can be obtained."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self._m = SentenceTransformer(model_id)
        self.name = model_id
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._m.encode(texts, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        return v.astype("float32")


class HashedNgramEncoder(Encoder):
    """Deterministic offline stand-in: char n-gram hashing + SVD to 384 dims.

    Used ONLY when the pinned sentence-transformer cannot be downloaded. It is
    not a sentence embedder and makes no semantic claims; it is a reproducible
    dense featurizer of the same dimension, so that the index, the classifiers
    and the contracts all exercise the real code path. results.json records
    which encoder actually ran, and the report never attributes a number
    produced here to all-MiniLM-L6-v2.
    """

    def __init__(self, dim: int = 384, seed: int = 20261103):
        self.name = f"hashed-char-ngram-svd-{dim} (OFFLINE STAND-IN)"
        self.dim = dim
        self._pipe = make_pipeline(
            HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              n_features=2 ** 14, alternate_sign=False,
                              norm=None),
            TruncatedSVD(n_components=dim, random_state=seed),
            Normalizer(copy=False),
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "HashedNgramEncoder":
        # Fitted on the EXEMPLAR pool only. Fitting on the evaluation split
        # would leak its distribution into the representation.
        self._pipe.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("HashedNgramEncoder.fit must run on the "
                               "exemplar pool before encoding")
        return np.asarray(self._pipe.transform(texts), dtype="float32")


def select_encoder(exemplar_texts: list[str]) -> tuple[Encoder, dict]:
    """Prefer the pinned model; fall back loudly, never silently."""
    try:
        enc = SentenceTransformerEncoder()
        return enc, {"encoder": enc.name, "encoder_is_pinned_model": True,
                     "encoder_fallback_reason": None}
    except Exception as e:  # noqa: BLE001
        enc = HashedNgramEncoder().fit(exemplar_texts)
        return enc, {
            "encoder": enc.name,
            "encoder_is_pinned_model": False,
            "encoder_fallback_reason":
                f"sentence-transformers/all-MiniLM-L6-v2 unavailable: "
                f"{type(e).__name__}: {str(e)[:160]}",
        }
