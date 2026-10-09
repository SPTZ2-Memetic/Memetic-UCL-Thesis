"""Canonical decomposition into construct-level chunks for construct-matching.

A bug almost always lives in one statement's expression (sort a list, index a
result, divide by N, compare, filter). So we decompose the canonical into
ast.stmt-level chunks -- the smallest unit that contains a COMPLETE bug-able
expression while staying coherent code. Compound statements (for/if/while)
contribute their header test/iterable as a chunk AND recurse one level into the
body (capped depth). Expression-level would fragment ('reverse', 'True');
block-level would be too coarse (whole loop). Statement-level is the bug-able
grain.
"""

from __future__ import annotations

import ast
import re
from typing import List


def _seg(source: str, node: ast.AST) -> str:
    try:
        s = ast.get_source_segment(source, node)
        return (s or "").strip()
    except Exception:
        return ""


def _is_docstring(seg: str) -> bool:
    seg = seg.lstrip()
    return seg.startswith('"""') or seg.startswith("'''")


def decompose(source: str, max_depth: int = 2) -> List[str]:
    """Return construct-level source snippets from a function's source.

    - each simple statement (Assign/Return/Expr/...) -> one snippet
    - for/if/while -> the test/iterable expression is its own snippet, and body
      statements recurse (up to max_depth)
    Deduplicated; imports and docstrings dropped.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    body = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = n.body
            break
    if body is None:
        body = tree.body

    out: List[str] = []

    def visit(stmts, depth):
        if depth > max_depth:
            return
        for st in stmts:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(st, (ast.For, ast.AsyncFor)):
                seg = _seg(source, st.iter)
                if seg:
                    out.append("iterate: " + seg)
                visit(st.body, depth + 1)
                visit(st.orelse, depth + 1)
            elif isinstance(st, ast.While):
                seg = _seg(source, st.test)
                if seg:
                    out.append("loop while: " + seg)
                visit(st.body, depth + 1)
            elif isinstance(st, ast.If):
                seg = _seg(source, st.test)
                if seg:
                    out.append("condition: " + seg)
                visit(st.body, depth + 1)
                visit(st.orelse, depth + 1)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                visit(st.body, depth + 1)
            else:
                seg = _seg(source, st)
                if seg and not _is_docstring(seg):
                    out.append(seg)

    visit(body, 0)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def decompose_windowed(source: str, max_depth: int = 2, window_lines: int = 2) -> List[str]:
    """Same statement-selection as decompose() (structurally parallel walk,
    so behavior is IDENTICAL -- same statements chosen, same recursion, same
    dedup), but each returned chunk is a WINDOW of `window_lines` lines of
    REAL source before and after the target statement, not the bare
    statement alone.

    Built to fix a confirmed real-data failure: a bare short statement (e.g.
    'return my_dict', 2 words) carries almost no discriminating signal for
    an embedding model -- measured directly, it matched 14 semantically
    unrelated statements (return True, return ax, return None...) purely on
    shallow token shape. A window of surrounding code gives the embedding
    real context about WHAT KIND of code region this is, while staying
    click-comprehensible for a human (few concrete lines, not the whole
    function) -- MORE localized than whole-canonical matching, LESS
    threadbare than a bare statement.

    Uses each statement's own lineno/end_lineno (captured during the SAME
    walk that selects it, not correlated back afterward by text search,
    which would be ambiguous if the same short text occurs twice) to slice
    real lines directly from `source` -- never AST-unparsed/reformatted
    text, so windows are always verbatim, exactly as written.

    window_lines is lines of context on EACH side; the target statement's
    own line range is always included regardless of window_lines. Windows
    are clamped to the source's actual bounds -- never reads past start/end
    of `source`. Compound-statement headers (for/if/while) use the SAME
    "iterate:"/"condition:"/"loop while:" label prefix as decompose(), kept
    OUTSIDE the window (labels the window, isn't part of the real code being
    matched on)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    src_lines = source.splitlines()

    def window_for(lineno: int, end_lineno: int) -> str:
        start = max(0, lineno - 1 - window_lines)
        end = min(len(src_lines), end_lineno + window_lines)
        return "\n".join(src_lines[start:end]).strip()

    body = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = n.body
            break
    if body is None:
        body = tree.body

    out: List[str] = []

    def visit(stmts, depth):
        if depth > max_depth:
            return
        for st in stmts:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(st, (ast.For, ast.AsyncFor)):
                seg = _seg(source, st.iter)
                if seg:
                    win = window_for(st.iter.lineno, getattr(st.iter, "end_lineno", st.iter.lineno))
                    out.append(f"iterate: {seg}\n---\n{win}")
                visit(st.body, depth + 1)
                visit(st.orelse, depth + 1)
            elif isinstance(st, ast.While):
                seg = _seg(source, st.test)
                if seg:
                    win = window_for(st.test.lineno, getattr(st.test, "end_lineno", st.test.lineno))
                    out.append(f"loop while: {seg}\n---\n{win}")
                visit(st.body, depth + 1)
            elif isinstance(st, ast.If):
                seg = _seg(source, st.test)
                if seg:
                    win = window_for(st.test.lineno, getattr(st.test, "end_lineno", st.test.lineno))
                    out.append(f"condition: {seg}\n---\n{win}")
                visit(st.body, depth + 1)
                visit(st.orelse, depth + 1)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                visit(st.body, depth + 1)
            else:
                seg = _seg(source, st)
                if seg and not _is_docstring(seg):
                    win = window_for(st.lineno, getattr(st, "end_lineno", st.lineno))
                    out.append(win)

    visit(body, 0)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ---- changed-construct extraction (skill-side construct-key) --------------

import textwrap as _textwrap

def _normalize_statement(stmt_src: str) -> str:
    """Alpha-rename local identifiers to placeholders (v1, v2, ...) in order of
    first appearance, so a statement that's merely RETYPED (renamed variables,
    no logical change) normalizes IDENTICALLY -- but a statement with a genuine
    change (different literal, operator, call, keyword arg, index) normalizes
    DIFFERENTLY. Pure text similarity cannot make this distinction (both a
    rename and a real edit look like a small textual diff); AST-level alpha-
    renaming can, because renaming preserves structure and a real edit changes
    it."""
    import builtins
    try:
        wrapped = "def _f():\n" + "\n".join(("    " + l) if l.strip() else l
                                             for l in stmt_src.splitlines())
        tree = ast.parse(wrapped)
    except SyntaxError:
        return stmt_src.strip()

    counter = {"n": 0}
    mapping: dict = {}
    KEEP = set(dir(builtins)) | {"self", "cls"}

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in KEEP:
                return node
            if node.id not in mapping:
                counter["n"] += 1
                mapping[node.id] = f"v{counter['n']}"
            node.id = mapping[node.id]
            return node

    try:
        Renamer().visit(tree)
        ast.fix_missing_locations(tree)
        out = ast.unparse(tree)
        lines = out.splitlines()
        body = lines[1:] if lines and lines[0].startswith("def ") else lines
        return _textwrap.dedent("\n".join(body)).strip()
    except Exception:
        return stmt_src.strip()


def _is_low_signal(norm_stmt: str) -> bool:
    """A normalized statement like 'return (v1, v2)' or 'v1 = v2' carries
    almost no discriminating signal -- virtually any statement of the same
    SHAPE normalizes identically regardless of what the identifiers actually
    reference. Confirmed on real data (BigCodeBench/66, human source): a bare
    multi-value return statement ('return analyzed_df, ax' -> 'return
    grouped_df, ax') alone pushed survive_frac to exactly 0.5 for a function
    whose entire computation -- aggregation method, grouping columns, plot
    type -- had been completely rewritten, because the return normalized
    identically despite returning entirely different data. Such statements
    must not count as evidence the function is mostly unchanged; they are
    excluded from the survival count entirely (not counted as survived NOR
    as changed -- they're simply uninformative either way)."""
    stripped = re.sub(r"\bv\d+\b", "", norm_stmt)
    stripped = re.sub(r"[\s(),\[\]:]+", "", stripped)
    stripped = re.sub(r"^(return|pass)$", "", stripped)
    return len(stripped) == 0


def _call_names(stmt_src: str) -> set:
    """The set of function/method names actually invoked in a statement --
    e.g. {'groupby', 'nunique', 'reset_index'} for
    'df.groupby(x)[y].nunique().reset_index()'. This is the signal that
    actually distinguishes a TWEAK (same operations, different args/literals)
    from a REWRITE (different operations entirely) -- raw character-level
    text similarity cannot reliably do this: 'ax = sns.distplot(x)' and
    'ax = sns.histplot(y)' share enough boilerplate characters ('ax = sns.',
    parens) to score sim=0.575 on difflib.SequenceMatcher despite calling a
    completely different function. Confirmed on real data: this let a
    genuine full rewrite (BigCodeBench/66 -- different aggregation method,
    different plot function) pass a 0.4 text-similarity threshold just like
    a real single-argument tweak (BigCodeBench/28) did -- survive_frac also
    could not tell them apart, landing on the identical 1-survived/3-total
    profile for both. Call NAMES, not call TEXT and not survival counts,
    carry the real signal."""
    try:
        wrapped = "def _f():\n" + "\n".join(("    " + l) if l.strip() else l
                                             for l in stmt_src.splitlines())
        tree = ast.parse(wrapped)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _call_overlap(stmt_a: str, stmt_b: str) -> float:
    """Jaccard overlap of call names between two statements. 1.0 if neither
    statement contains any calls (nothing to disagree on -- e.g. plain
    assignments of literals/names); 0.0 if only one of them contains calls.

    CONFIRMED NEEDED on real data (distillation probe, gpt_oss_20b source
    and several BigCodeBench/2-style comprehension->loop rewrites): when
    the SMALLER of the two call-name sets has only ONE element, a shared
    call name is weak evidence on its own -- common utility calls (len,
    randint, str) appear throughout completely unrelated code, so a lone
    match trivially scores 1.0 regardless of the statement's real
    structural role. This let whole-function reimplementations (different
    control flow, different variables, genuinely different logic) pass as
    a "clean local edit" purely because both versions happened to call
    e.g. `len` or `randint` somewhere in the statement. When the smaller
    set has size <=1, require corroborating TEXTUAL similarity as well --
    a real single-call-name tweak (e.g. a literal argument change) still
    looks textually similar; an unrelated rewrite that merely shares one
    call name does not."""
    a, b = _call_names(stmt_a), _call_names(stmt_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    jaccard = len(a & b) / len(a | b)
    if min(len(a), len(b)) <= 1:
        import difflib
        text_sim = difflib.SequenceMatcher(None, stmt_a, stmt_b).ratio()
        if text_sim < 0.35:
            return 0.0
    return jaccard


def changed_construct(canonical_body: str, buggy_body: str, max_parts: int = 3,
                      windowed: bool = False, window_lines: int = 2):
    """Decompose canonical and buggy into statements; return the CANONICAL
    statement(s) that the bug changed -- the construct the skill targets.

    For a local edit, canonical and buggy align statement-for-statement except at
    the bug. We return the canonical statements that are NOT present verbatim in
    the buggy decomposition. Returns (construct_str, aligned: bool):
      aligned=True  -> a clean local edit; construct_str is the targeted code.
      aligned=False -> the two don't align (a genuine rewrite); construct_str is
                       empty. Such cases are NOT local edits and are dropped from
                       construct-matching.

    windowed=True: the RETURNED text is widened to include window_lines lines
    of real surrounding code on each side, via decompose_windowed (index-
    correspondence with the bare decompose() call below -- both walk the
    SAME statements in the SAME order, so canon[i] and its windowed
    counterpart always refer to the same statement; NOT a text search, which
    would be ambiguous if the same short statement occurs twice). The
    ALIGNMENT DETECTION below is entirely UNCHANGED either way -- windowing
    only affects what text gets returned for an already-determined match,
    never whether something counts as aligned. Confirmed needed on real
    data: a bare short statement (e.g. 'return my_dict') carries almost no
    discriminating signal for an embedding model -- measured directly, one
    such construct matched 14 semantically unrelated statements purely on
    shallow token shape (see project notes).
    """
    def _wrap(body):
        # bodies are stored bare (indented, no def) -> wrap so ast can parse
        import textwrap
        b = body if body.lstrip().startswith("def ") else "def _f():\n" + body
        return b
    canon = decompose(_wrap(canonical_body))
    buggy = decompose(_wrap(buggy_body))
    if not canon:
        return "", False

    # Compare NORMALIZED statements (identifiers alpha-renamed to placeholders),
    # not raw text and not fuzzy similarity. A statement "survives" if some
    # buggy statement normalizes to the SAME structure (i.e. it's a rename/
    # retype of it, logically identical). A statement is "changed" if no buggy
    # statement normalizes the same way -- a genuine structural difference
    # (different literal, operator, call, keyword, index), which a rename
    # cannot produce and pure text similarity cannot reliably detect.
    canon_norm = [_normalize_statement(c) for c in canon]
    buggy_norm_set = set(_normalize_statement(b) for b in buggy)

    changed = [c for c, cn in zip(canon, canon_norm) if cn not in buggy_norm_set]
    if not changed:
        return "", False   # every statement normalizes the same -> no real change

    # Low-signal statements (see _is_low_signal) are excluded from the
    # high-signal working set -- they neither count as surviving evidence nor
    # as changed evidence, since their generic shape can coincidentally match
    # (or fail to match) regardless of actual content.
    hi_idx = [i for i, cn in enumerate(canon_norm) if not _is_low_signal(cn)]
    if not hi_idx:
        hi_idx = list(range(len(canon)))

    hi_canon = [canon[i] for i in hi_idx]
    hi_norm = [canon_norm[i] for i in hi_idx]
    hi_changed = [c for c, cn in zip(hi_canon, hi_norm) if cn not in buggy_norm_set]
    survived = len(hi_canon) - len(hi_changed)
    survive_frac = survived / max(1, len(hi_canon))  # kept for logging; not gated on below (see note)

    if len(canon) <= 2:
        # tiny function: survival fraction is unreliable (the only statement may
        # BE the bug). Require the changed statement to still resemble something
        # in buggy textually (a tweak, not an unrelated rewrite).
        import difflib
        def best_sim(stmt, pool):
            return max((difflib.SequenceMatcher(None, stmt, b).ratio() for b in pool), default=0.0)
        aligned = all(best_sim(c, buggy) >= 0.4 for c in changed) and len(buggy) <= len(canon) + 1
    else:
        # Both text-level similarity (difflib) and raw survive_frac were
        # tested against real data and neither can separate a genuine tweak
        # from a genuine rewrite: BigCodeBench/28 (tweak: same base64/post
        # calls, different literal args) and BigCodeBench/66 (rewrite:
        # different aggregation method, different plot function) landed on
        # the IDENTICAL 1-survived/3-total profile, and BOTH cleared a 0.4
        # text-similarity gate (shared boilerplate like 'ax = sns.(...)'
        # inflates character-level similarity regardless of which function
        # is actually called). Call-name overlap is the signal that actually
        # discriminates: a statement counts as a local edit only if it
        # invokes largely the SAME functions/methods as its closest buggy
        # counterpart.
        aligned = (
            len(hi_changed) <= 2
            and all(
                max((_call_overlap(c, b) for b in buggy), default=0.0) >= 0.5
                for c in hi_changed
            )
        )
    if not aligned:
        return "", False

    if not windowed:
        return " ; ".join(changed[:max_parts]), True

    canon_windowed = decompose_windowed(_wrap(canonical_body), window_lines=window_lines)
    if len(canon_windowed) != len(canon):
        # walk mismatch (shouldn't happen -- both walk the SAME source with the
        # SAME logic -- but fall back to bare text rather than risk a wrong
        # index correspondence)
        return " ; ".join(changed[:max_parts]), True
    windows = []
    for c in changed[:max_parts]:
        idx = canon.index(c)
        windows.append(canon_windowed[idx])
    # dedup overlapping windows (adjacent changed statements can share lines)
    seen, uniq_windows = set(), []
    for w in windows:
        if w not in seen:
            seen.add(w)
            uniq_windows.append(w)
    return "\n\n".join(uniq_windows), True


# ---- mechanical edit description (no LLM) ---------------------------------
def _diff_pair(canonical_body: str, buggy_body: str):
    """Return (was, became): the changed canonical statement(s) and their buggy
    counterpart(s), as short code strings. Uses the aligned decomposition."""
    def _wrap(body):
        import textwrap
        return body if body.lstrip().startswith("def ") else "def _f():\n" + body
    canon = decompose(_wrap(canonical_body))
    buggy = decompose(_wrap(buggy_body))
    was = [c for c in canon if c not in set(buggy)]
    became = [b for b in buggy if b not in set(canon)]
    return " ; ".join(was[:3]), " ; ".join(became[:3])


def describe_edit(canonical_body: str, buggy_body: str, error_signature: str = "") -> str:
    """Clean, LLM-readable description of a local-edit skill, built mechanically
    from the diff + error. Format:

        WHEN the code contains:  <was>
        CHANGE it to:            <became>
        TO CAUSE:                <error>

    Returns '' if there is no coherent local change (a rewrite)."""
    construct, aligned = changed_construct(canonical_body, buggy_body)
    if not aligned:
        return ""   # rewrite / no coherent local change -> not a local-edit skill
    was, became = _diff_pair(canonical_body, buggy_body)
    if not was and not became:
        return ""
    lines = []
    if was:
        lines.append(f"WHEN the code contains:  {was}")
    if became:
        lines.append(f"CHANGE it to:            {became}")
    if error_signature:
        lines.append(f"TO CAUSE:                {error_signature.strip()}")
    return "\n".join(lines)
