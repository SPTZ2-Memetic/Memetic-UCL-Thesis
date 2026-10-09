"""memetic -- generator/fixer self-play with memory over a frozen frontier model.

The public surface is intentionally small: the v2 self-play loop and the two
evolving banks are the entry points (see memetic.loop_v2, memetic.skill_bank,
memetic.fixer_memory). The names re-exported here are the shared primitives
used across modules and by the pipeline scripts.
"""
from memetic.types import Case, Record
from memetic.interfaces import Policy
from memetic.embeddings import Embedder, cosine_similarity
from memetic.loader import load_seed_pool, load_warmstart, load_bugbench_records
from memetic.verifier import FunctionVerifier, VerifyResult, ExecutionOutcome

__all__ = [
    "Case", "Record", "Policy",
    "Embedder", "cosine_similarity",
    "load_seed_pool", "load_warmstart", "load_bugbench_records",
    "FunctionVerifier", "VerifyResult", "ExecutionOutcome",
]
