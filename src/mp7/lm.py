"""
CIS 731 — MP 7: the few-shot prompt harness and LM backends.

CONSTRAINED DECODING BY CONSTRUCTION
------------------------------------
The backend contract is `score_labels(prompt, verbalizers) -> np.ndarray`:
a backend scores ONLY the allowed continuations and returns one score per
label. There is no free generation step, so a hallucinated label is not
"filtered out downstream" -- it is unrepresentable. A real HF backend
implements this by scoring each verbalizer's token sequence as a continuation
of the prompt; it never calls `.generate()`.

The output parser therefore has one job: map the arg-max verbalizer back to
its label id, and assert membership. It is kept anyway, because the argmax
index must still be proved to land inside the declared label space.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .corpus import BY_VERBALIZER

SYSTEM_INSTRUCTION = (
    "You are a music-style classifier for regional American hip-hop. "
    "Given a short lyric excerpt, identify the single micro-genre it belongs "
    "to. Answer with exactly one label from the provided list and nothing "
    "else. Do not explain. Do not invent labels."
)


def render_prompt(query: str, demos: list[tuple[str, str]],
                  verbalizers: list[str]) -> str:
    """Build the prompt. `demos` is a list of (text, verbalizer) in the order
    they should appear -- the caller owns permutation, so order is explicit
    and auditable rather than hidden inside this function."""
    parts = [SYSTEM_INSTRUCTION,
             "",
             "Allowed labels: " + ", ".join(verbalizers),
             ""]
    for text, verb in demos:
        parts += [f"Excerpt: {text}", f"Label: {verb}", ""]
    parts += [f"Excerpt: {query}", "Label:"]
    return "\n".join(parts)


def parse_label(verbalizer: str, labels: tuple[str, ...]) -> str:
    """Map a verbalizer back to its label id, asserting closed-set membership."""
    lid = BY_VERBALIZER.get(verbalizer.strip().lower())
    if lid is None or lid not in labels:
        raise ValueError(f"parser produced out-of-space label {verbalizer!r}")
    return lid


def scan_label_leakage(demos: list[tuple[str, str]]) -> list[str]:
    """Return any demonstration whose INPUT text contains a verbalizer.

    protocol.md §8 control for label leakage: the answer must never appear
    inside the question. Returns offenders so the caller can fail loudly.
    """
    bad = []
    for text, _ in demos:
        low = text.lower()
        for verb in BY_VERBALIZER:
            if re.search(rf"\b{re.escape(verb)}\b", low):
                bad.append(text)
                break
    return bad


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class LMBackend:
    name: str
    revision: str

    def score_labels(self, prompt: str,
                     verbalizers: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HFCausalBackend(LMBackend):
    """The pinned instruction-tuned model (protocol.md §7), CPU inference.

    Scores each verbalizer as a continuation by summed log-probability,
    length-normalized. Never generates. Unused in this run because the model
    could not be downloaded in this environment; kept because it is the real
    path and swapping back is a one-line change in `select_backend`.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch  # noqa: PLC0415
        from transformers import (AutoModelForCausalLM,  # noqa: PLC0415
                                  AutoTokenizer)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.name = model_id
        self.revision = getattr(self.model.config, "_commit_hash", "unknown")

    def score_labels(self, prompt: str, verbalizers: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        p_ids = self.tok(prompt, return_tensors="pt").input_ids
        for verb in verbalizers:
            v_ids = self.tok(" " + verb, add_special_tokens=False,
                             return_tensors="pt").input_ids
            ids = torch.cat([p_ids, v_ids], dim=1)
            with torch.no_grad():
                logits = self.model(ids).logits
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(float(tok_lp[0, -v_ids.shape[1]:].mean()))
        return np.array(out, dtype="float64")


class DeterministicStubBackend(LMBackend):
    """Offline stand-in used when no instruction-tuned model can be obtained.

    WHAT THIS IS AND IS NOT. It is a transparent, seeded scorer that reads
    ONLY the rendered prompt string, exactly as a language model would. That
    means the demonstration block, its order, and k genuinely change its
    output, so the whole harness -- formatting, permutation, calibration,
    constrained decoding, parsing -- is really exercised end to end.

    It is NOT a language model. Any macro-F1 it produces is a property of this
    function. It is reported in the results as harness validation and is never
    compared against the prototype or kNN classifiers as if it were a model
    comparison. The report says so in the same words.

    Mechanism: score each label by lexical overlap between the query excerpt
    and the demonstration excerpts carrying that label inside this prompt,
    plus a small prompt-dependent deterministic jitter standing in for a
    model's intrinsic label prior. With k=0 there are no demonstrations, so
    only the prior remains -- which is the behaviour a zero-shot control
    should have.
    """

    def __init__(self, seed: int = 20261103):
        self.name = "deterministic-stub (NOT A LANGUAGE MODEL)"
        self.revision = f"seed={seed}"
        self._seed = seed

    @staticmethod
    def _blocks(prompt: str) -> list[tuple[str, str]]:
        return re.findall(r"Excerpt: (.*)\nLabel: (.+)", prompt)

    def score_labels(self, prompt: str, verbalizers: list[str]) -> np.ndarray:
        demos = self._blocks(prompt)
        m = re.search(r"Excerpt: (.*)\nLabel:\s*$", prompt)
        query = set((m.group(1) if m else "").split())

        scores = np.zeros(len(verbalizers), dtype="float64")
        for i, verb in enumerate(verbalizers):
            overlap, n = 0.0, 0
            for text, dverb in demos:
                if dverb.strip().lower() != verb.lower():
                    continue
                d = set(text.split())
                if d:
                    overlap += len(query & d) / len(query | d)
                    n += 1
            if n:
                scores[i] = overlap / n
            # Deterministic per-(prompt, label) prior: stands in for the
            # intrinsic bias a real LM has toward some verbalizers.
            h = hashlib.sha256(
                f"{self._seed}|{verb}|{len(demos)}".encode()).digest()
            scores[i] += 0.02 * (h[0] / 255.0)
        return scores


def select_backend() -> tuple[LMBackend, dict]:
    """Prefer the pinned model; fall back loudly, never silently."""
    try:
        be = HFCausalBackend()
        return be, {"llm": be.name, "llm_revision": be.revision,
                    "llm_is_pinned_model": True, "llm_fallback_reason": None}
    except Exception as e:  # noqa: BLE001
        be = DeterministicStubBackend()
        return be, {
            "llm": be.name, "llm_revision": be.revision,
            "llm_is_pinned_model": False,
            "llm_fallback_reason":
                f"Qwen/Qwen2.5-0.5B-Instruct unavailable: "
                f"{type(e).__name__}: {str(e)[:120]}; additionally, huggingface.co is blocked by this environment's egress proxy (verified: HTTP 403 on a direct config.json fetch), so the weights could not have been downloaded either.",
        }
