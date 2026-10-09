"""Fixer-side FAULT fingerprinting -- the dual of the generator's site extraction.

The generator asks "where could a bug land?" and enumerates every mutable site.
The fixer asks the opposite, narrower question: "what single operation is failing,
and how?" -- and keys retrieval on that, because the thing that makes a past repair
reusable for a new bug is NOT surface code similarity (matches on irrelevant parts)
nor the error title (same title, different library, different fix). It is:

    (operation at the fault, failure mode)

- operation : the callee name of a Call (`re.findall`, `df.astype`, `pd.DataFrame`),
              or the operator of a comparison/arithmetic node (`Lt`, `Add`) -- the
              thing that encodes the library + semantics. NEVER the container
              statement label (`Assign`/`Return`): that is what collapsed the old
              bank to a single `assignment/Assign` monoculture (the enclosing
              Assign always won `_fix_fingerprint`'s tie-break because its AST
              type name "Assign" is truthy and its text is longest).
- failure   : a symptom class read off the expected-vs-actual, finer than the old
              3-bucket failure_class (value_mismatch / exception / timeout), which
              was constant within a stratum and so carried no signal.

Also sizes the REPAIR SHAPE from a fix diff (how many sites, which operations,
single-edit vs guard vs multi-coordinated). A single observed symptom can only
ever point at the proximate fault; it cannot know a fix needs three coordinated
changes. A stored skill CAN -- that non-local repair structure is the one thing
memory carries that a local diagnosis cannot reconstruct.

Pure / deterministic. No LLM, no embeddings. Reuses sites.extract_meaningful_sites.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from memetic.sites import extract_meaningful_sites, Site
from memetic.codeutils import error_kind, CRASH_KINDS


# ---------------------------------------------------------- operation token ----
# A site is DISCRIMINATIVE if it names an operation (a callee / a real AST
# operator / a slice). Container statements (assignment, return) and the bare
# structural keywords (If/For/While on a loop or branch header) are NOT: they
# match almost every function and are exactly what saturated the old channel.
_SITE_RANK = {
    "call": 5,            # most discriminative: a named callee (re.findall, astype)
    "collection_op": 4,   # ListComp/DictComp/GeneratorExp
    "comparison": 3,      # Lt / Gt / Eq ...
    "binary_expr": 3,     # Add / Sub / Mod ...
    "boolean_expr": 3,    # And / Or
    "subscript": 2,       # x[...] -- weak token but still localizing
    "loop_header": 1,
    "branch": 1,
    "assignment": 0,
    "return": 0,
}


def _discriminative(s: Site) -> bool:
    if s.site_type == "call":
        return bool(s.operator)           # has a resolvable callee name
    if s.site_type in ("comparison", "binary_expr", "boolean_expr", "collection_op"):
        return bool(s.operator)
    if s.site_type == "subscript":
        return True
    return False


def _rank_key(s: Site):
    # most discriminative first; then by site rank; then the TIGHTER span (shorter
    # text) so the nested call wins over the whole statement that contains it.
    return (_discriminative(s), _SITE_RANK.get(s.site_type, 0), -len(s.source_text))


# ------------------------------------------------------------ span matching ----
_BACKTICK = re.compile(r"`([^`]+)`")


def spans_from_diagnosis(dx: str) -> List[str]:
    """The v2 diagnoser is told to quote the responsible construct; it reliably
    returns it in backticks (`range(len(x) - 2)`, `re.findall(...)`). Those quoted
    spans ARE the read-time localization we lacked before -- extract them."""
    return [m.group(1).strip() for m in _BACKTICK.finditer(dx or "") if m.group(1).strip()]


def _sites_covering(source: str, span: str) -> List[Site]:
    """Sites whose source text overlaps the span (either contains the other)."""
    span = (span or "").strip()
    if not span:
        return []
    out = []
    for s in extract_meaningful_sites(source):
        t = s.source_text.strip()
        if t and (t in span or span in t):
            out.append(s)
    return out


def best_fault_site(source: str, spans: List[str]) -> Optional[Site]:
    """Among the sites covered by any span, the single most discriminative one."""
    cands: List[Site] = []
    for sp in spans:
        cands.extend(_sites_covering(source, sp))
    if not cands:
        return None
    cands.sort(key=_rank_key, reverse=True)
    return cands[0]


def fault_fingerprint(source: str, spans: List[str]) -> Tuple[str, str, Tuple[str, ...]]:
    """(site_type, operation_token, kwarg_names) for the proximate fault, or
    ('', '', ()) if nothing discriminative can be localized (then the structural
    channel simply does not fire and retrieval falls back to semantics)."""
    s = best_fault_site(source, spans)
    if s is None:
        return ("", "", ())
    return (s.site_type, s.operator, tuple(s.keywords))


# ------------------------------------------------------ skill-side (from diff) ----
def _removed_lines(diff: str) -> List[str]:
    return [l[1:].strip() for l in (diff or "").splitlines()
            if l.startswith("-") and not l.startswith("---") and l[1:].strip()]


def _added_lines(diff: str) -> List[str]:
    return [l[1:].strip() for l in (diff or "").splitlines()
            if l.startswith("+") and not l.startswith("+++") and l[1:].strip()]


def fix_fingerprint(buggy_full: str, fixed_diff: str) -> Tuple[str, str, Tuple[str, ...]]:
    """The skill's fix-site fingerprint -- symmetric to fault_fingerprint, but the
    spans are the diff's REMOVED lines (the buggy construct the fix replaced).
    Fixes that only ADD code (missing guard) have no removed construct -> ('','',())
    and just don't fire the structural channel, which is correct."""
    return fault_fingerprint(buggy_full, _removed_lines(fixed_diff))


def fix_shape(buggy_full: str, fixed_diff: str) -> dict:
    """Size the repair: the multiset of operations it touched, a rough site count,
    and a coarse shape tag. This is the multi-site knowledge a skill carries."""
    removed = _removed_lines(fixed_diff)
    added = _added_lines(fixed_diff)
    ops = []
    for sp in removed:
        s = best_fault_site(buggy_full, [sp])
        if s and _discriminative(s):
            ops.append(f"{s.site_type}:{s.operator}" if s.operator else s.site_type)
    added_text = "\n".join(added)
    adds_guard = bool(re.search(r"\b(isinstance|raise|try:|assert)\b|^\s*if\b", added_text, re.M))
    adds_conv = bool(re.search(r"pd\.DataFrame|\.astype|np\.array|\.to_[a-z]+\(|"
                               r"\bfloat\(|\bint\(|\bstr\(|\blist\(", added_text))
    n_sites = max(len(removed), len(added))
    if n_sites <= 1:
        tag = "single-edit"
    elif adds_guard and len(removed) == 0:
        tag = "add-guard"
    elif adds_conv:
        tag = "add-conversion"
    else:
        tag = "multi-coordinated"
    return {"op_multiset": sorted(set(ops)), "n_sites": n_sites, "shape": tag}


# ----------------------------------------------------------- symptom class ----
def symptom_class(error: str) -> str:
    """Finer failure mode than the 3-bucket failure_class. Computed identically on
    the query side (from the test error) and the skill side (from the originating
    RepairCase.failure), so 'same failure mode' is a real, consistent signal.

    NOTE: error_kind returns 'other' whenever error_signature's last exception-shaped
    line is a trailing summary ('FAILED (failures=1)') rather than the AssertionError
    line -- which is common. So we do NOT trust error_kind=='assertion' as the gate;
    we only trust it for the crash kinds (which key on the exception name, reliable)
    and the timeout, and otherwise sub-classify on the RAW text. This is what killed
    the channel before (126/178 skills bucketed to a dead 'other')."""
    s = error or ""
    low = s.lower()
    k = error_kind(s)
    if k == "timeout":
        return "timeout"
    if k in CRASH_KINDS:
        return f"crash:{k}"
    # assertion OR 'other' -> a value-level mismatch; classify from the raw text and
    # NEVER return 'other'. The raw output still carries '!=', 'differ', dtype tokens
    # even when the signature line that fooled error_kind was a summary.
    if (re.search(r"dtype|int64|float64|<class '|object dtype", s)
            or ("type" in low and ("!=" in s or "differ" in low))):
        return "wrong_type"
    if (re.search(r"shape|lists differ|tuples differ|length|elements differ", low)
            or re.search(r"\(\d+,\s*\d*\)", s)):
        return "wrong_shape_len"
    if re.search(r"keyerror|missing|not found|\bnot in\b|counter\(", low):
        return "missing_element"
    return "wrong_value"


# ------------------------------------------------------------ token matching ----
def method_of(op: str) -> str:
    """Trailing method/callee of a dotted op token: 'random_dict.items' -> 'items',
    'np.mean' -> 'mean', 'len' -> 'len'."""
    return op.split(".")[-1] if op else ""


def op_match(q_op: str, s_op: str) -> float:
    """Operation-token match score. 1.0 exact; 0.6 same trailing method but different
    receiver ('random_dict.items' vs 'sales_data.items' -- the SAME operation, which
    the raw token fragmented); 0.0 otherwise. Recovers the receiver-pollution misses
    without needing to classify library vs variable prefixes."""
    if not q_op or not s_op:
        return 0.0
    if q_op == s_op:
        return 1.0
    m = method_of(q_op)
    if m and m == method_of(s_op):
        return 0.6
    return 0.0
