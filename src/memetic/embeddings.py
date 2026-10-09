"""Generator redesign: local embedding-based similarity for dedup's
mechanism-text prefilter -- an alternative to dedup.mechanism_text_similarity
(stdlib difflib.SequenceMatcher), tested against real LLM verdicts by
scripts/calibrate_dedup.py rather than assumed to be better.

WHY THIS EXISTS: SequenceMatcher.ratio() scores literal character/word
overlap (Ratcliff-Obershelp longest-common-block matching), not meaning.
That was a defensible default when the dedup prefilter was first built (see
dedup.py's own docstring), but it has a real failure mode specific to this
project: distilled skills come from three independent sources (human,
qwen7b, gpt_oss_20b), each describing the same underlying mechanism in its
own words. Two genuine paraphrases of the same mechanism can share almost no
literal substrings and still score low under SequenceMatcher even though
they mean the same thing -- one real candidate explanation (not yet
confirmed either way, hence the calibration script) for why 88% of "new"
outcomes in a real run never had a single candidate clear the text
threshold at all.

THIS IS NOT a reversal of "no embeddings anywhere in this project" (see
dedup.py's module docstring, and matching.py/judge_skill.py's Stage A/B
design). That decision was scoped to a different task -- applicability
judgment (does a skill's precondition genuinely fit a site) -- where an
embedding's semantic-nearness signal doesn't map onto the actual question
being asked. Dedup's mechanism-text prefilter is a different problem
(paraphrase clustering), and was never a deliberate choice of metric in the
first place -- SequenceMatcher was just what stdlib offered when this stage
was first written. This module lets that one stage's metric be tested and
possibly swapped, without touching Stage A/B's own reasoning at all.

DELIBERATELY NOT embedding manifestations/diffs. skills.py's own invariant
(manifestations/examples must never drive matching/ranking decisions, only
guide generation) exists because one specific edit standing in for the whole
mechanism is a known failure mode (construct-matching's original problem).
Embedding raw diff text here would recreate exactly that under a different
name. Only name+mechanism goes in -- the same text mechanism_text_similarity
already scores -- so swapping the metric changes nothing about what's being
compared, only how.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


class Embedder:
    """Lazy-loaded, cached wrapper around a sentence-transformers model.
    Import of sentence_transformers is deferred to the first .encode()/
    .embed_one() call, so this module stays importable in environments where
    that package isn't installed -- dedup.py and other callers that never
    touch embeddings must not break."""

    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        import os
        # Model resolution order:
        #   1. explicit model_name argument
        #   2. MEMETIC_EMBED_MODEL env var -- a HF repo id OR an absolute path
        #      to a local snapshot directory. Pointing straight at a resolved
        #      snapshot directory sidesteps HF repo-id/cache-location resolution
        #      entirely, which is useful on hosts where the model is cached in a
        #      non-default location or where offline resolution is unreliable.
        #   3. the default HF repo id.
        if model_name is None:
            model_name = os.environ.get("MEMETIC_EMBED_MODEL", self._DEFAULT_MODEL)
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # Force offline mode before constructing the model: by default
            # SentenceTransformer issues a HEAD request to huggingface.co to
            # check for updates even when the model is already cached locally,
            # which hangs/fails on hosts without access to that endpoint. Set
            # MEMETIC_EMBED_ALLOW_ONLINE=1 to opt back into online resolution.
            import os
            if os.environ.get("MEMETIC_EMBED_ALLOW_ONLINE") != "1":
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        # normalize_embeddings=True -> cosine_similarity below reduces to a
        # plain dot product, but it re-normalizes defensively anyway so this
        # stays correct even given vectors from a differently-configured
        # embedder.
        return np.asarray(model.encode(list(texts), normalize_embeddings=True))

    def embed_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MechanismEmbeddingSimilarity:
    """Drop-in replacement for dedup.mechanism_text_similarity -- SAME
    (a_name, a_mechanism, b_name, b_mechanism) -> float call signature, so
    an instance of this class can be passed directly as
    dedup.mechanism_candidates's `similarity_fn` with no other code change;
    only the metric underneath changes.

    Caches embeddings by exact (name, mechanism) text within one instance's
    lifetime (in-memory only, not persisted across runs) -- a skill already
    in the bank gets compared against many new distilled cases over a
    single distillation run, and re-embedding its unchanged text every time
    would be wasted model calls."""

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self._cache: Dict[str, np.ndarray] = {}

    def _vec(self, name: str, mechanism: str) -> np.ndarray:
        key = f"{name} {mechanism}".strip().lower()
        if key not in self._cache:
            self._cache[key] = self.embedder.embed_one(key)
        return self._cache[key]

    def __call__(self, a_name: str, a_mechanism: str,
                 b_name: str, b_mechanism: str) -> float:
        return cosine_similarity(self._vec(a_name, a_mechanism),
                                  self._vec(b_name, b_mechanism))


SimilarityFn = Callable[[str, str, str, str], float]
