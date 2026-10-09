"""Generator redesign, Phase 1 (see generator_redesign_note.docx): AST-based
extraction of "meaningful mutation sites" from a canonical function -- the
foundation everything else in the redesign builds on. Pure, deterministic,
NO LLM calls, NO embeddings -- purely mechanical AST analysis.

This directly targets the confirmed root cause behind tonight's false
positives (e.g. BigCodeBench/230's drop_duplicates matching onto unrelated
fillna/groupby code): cosine similarity conflates "shares surface tokens"
with "shares mechanism". A Site here captures STRUCTURAL features (node
type, operator, enclosing construct) that a later matching stage can filter
on directly and cheaply, before any semantic judgment is needed.

Each Site is one of ~10 categories, per the design note:
  comparison, loop_header, call, assignment, return, boolean_expr,
  binary_expr, subscript, branch, collection_op

For each matched node we capture:
  site_type    -- the category (see above)
  operator     -- the specific operator/function for this node (e.g. 'Lt',
                  'Add', a call's function name) -- "" if not applicable
  source_text  -- exact source text (via constructs._seg, the SAME helper
                  used by the validated construct-matching code -- verbatim,
                  never AST-unparsed/reformatted)
  lineno / end_lineno -- for later windowing, reusing decompose_windowed's
                  line-based slicing approach if needed
  parent_type  -- the AST type of the immediately enclosing STATEMENT (not
                  just any ancestor) -- e.g. a Compare inside an If's test
                  has parent_type 'If'; inside an Assign's value, 'Assign'
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional

from memetic.constructs import _seg


@dataclass
class Site:
    site_type: str
    operator: str
    source_text: str
    lineno: int
    end_lineno: int
    parent_type: str
    keywords: tuple = ()   # for site_type=="call" only: the keyword argument
                           # NAMES present (e.g. ("axis",) for df.mean(axis=1))
                           # -- a cheap, deterministic STRUCTURAL fact (no LLM
                           # needed), added specifically because several
                           # CONFIRMED real bug patterns from tonight's own
                           # regressions (fillna/skew axis flips) hinge on
                           # "does this call have an axis kwarg", which the
                           # function name alone can't capture. Empty tuple
                           # for all other site types.
    parent_text: str = ""  # the FULL SOURCE TEXT of the enclosing statement
                           # (not just this site's own narrow expression) --
                           # added because a real judge run against actual
                           # canonicals showed the judge missing a wrapping
                           # math.sqrt() and a guarding `if x else None` that
                           # were directly adjacent to the site but outside
                           # its own narrow text. Equal to source_text when
                           # the site itself already IS the whole statement.

    def __repr__(self):
        op = f" op={self.operator}" if self.operator else ""
        return f"Site({self.site_type}{op} parent={self.parent_type}: {self.source_text!r})"


_COMPARE_OPS = {
    ast.Lt: "Lt", ast.LtE: "LtE", ast.Gt: "Gt", ast.GtE: "GtE",
    ast.Eq: "Eq", ast.NotEq: "NotEq", ast.Is: "Is", ast.IsNot: "IsNot",
    ast.In: "In", ast.NotIn: "NotIn",
}
_BINOP_OPS = {
    ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mult", ast.Div: "Div",
    ast.FloorDiv: "FloorDiv", ast.Mod: "Mod", ast.Pow: "Pow",
    ast.BitAnd: "BitAnd", ast.BitOr: "BitOr", ast.BitXor: "BitXor",
    ast.LShift: "LShift", ast.RShift: "RShift",
}
_BOOLOP_OPS = {ast.And: "And", ast.Or: "Or"}


def _call_func_name(node: ast.Call) -> str:
    """Best-effort function/method name for a Call node -- 'sorted',
    'df.fillna', etc. Falls back to '' for exotic callables (e.g. a call on
    a subscript result) rather than raising."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        base = _call_func_name_of_value(f.value)
        return f"{base}.{f.attr}" if base else f.attr
    return ""


def _call_func_name_of_value(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_func_name_of_value(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def extract_meaningful_sites(source: str, max_sites: Optional[int] = None) -> List[Site]:
    """Parse `source` (a full function, 'def f(...):\\n    ...') once and
    return every meaningful mutation site found, each a Site. Returns []
    on a syntax error (same graceful-degradation contract as
    constructs.decompose -- never raises on malformed input, since a real
    300-step run will throw messy real code at this).

    max_sites: if given, caps the returned list (first N found, in AST
    walk order) -- a safety valve for pathologically large functions;
    None (default) = no cap."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sites: List[Site] = []

    def add(node: ast.AST, site_type: str, operator: str, parent_type: str,
           parent_text: str, keywords: tuple = ()):
        text = _seg(source, node)
        if not text:
            return
        sites.append(Site(
            site_type=site_type, operator=operator, source_text=text,
            lineno=getattr(node, "lineno", 0),
            end_lineno=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            parent_type=parent_type, keywords=keywords,
            parent_text=parent_text or text,
        ))

    def visit_expr(node: ast.AST, parent_type: str, parent_text: str):
        """Look for meaningful sub-expressions WITHIN a statement (a
        statement can contain several: e.g. an Assign's value could itself
        be a Call, a Compare could be nested in a BoolOp, etc.) -- walks
        the full expression subtree, not just the top level, so nested
        sites are found too. parent_text is the FULL enclosing statement's
        text (same for every site found within one statement) -- lets a
        later reader (Stage B's judge) see a wrapping call or guard that
        sits outside the site's own narrow text."""
        for n in ast.walk(node):
            if isinstance(n, ast.Compare) and len(n.ops) == 1:
                op = _COMPARE_OPS.get(type(n.ops[0]), type(n.ops[0]).__name__)
                add(n, "comparison", op, parent_type, parent_text)
            elif isinstance(n, ast.BinOp):
                op = _BINOP_OPS.get(type(n.op), type(n.op).__name__)
                add(n, "binary_expr", op, parent_type, parent_text)
            elif isinstance(n, ast.BoolOp):
                op = _BOOLOP_OPS.get(type(n.op), type(n.op).__name__)
                add(n, "boolean_expr", op, parent_type, parent_text)
            elif isinstance(n, ast.Call):
                kws = tuple(sorted(k.arg for k in n.keywords if k.arg))
                add(n, "call", _call_func_name(n), parent_type, parent_text, keywords=kws)
            elif isinstance(n, ast.Subscript):
                add(n, "subscript", "", parent_type, parent_text)
            elif isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                add(n, "collection_op", type(n).__name__, parent_type, parent_text)

    def visit_stmts(stmts):
        for st in stmts:
            if max_sites is not None and len(sites) >= max_sites:
                return
            stmt_text = _seg(source, st)
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                add(st, "assignment", type(st).__name__, "Assign", stmt_text)
                value = getattr(st, "value", None)
                if value is not None:
                    visit_expr(value, "Assign", stmt_text)
            elif isinstance(st, ast.Return):
                add(st, "return", "", "Return", stmt_text)
                if st.value is not None:
                    visit_expr(st.value, "Return", stmt_text)
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                add(st.iter, "loop_header", "For", "For", stmt_text)
                visit_expr(st.iter, "For", stmt_text)
                visit_stmts(st.body)
                visit_stmts(st.orelse)
            elif isinstance(st, ast.While):
                add(st.test, "loop_header", "While", "While", stmt_text)
                visit_expr(st.test, "While", stmt_text)
                visit_stmts(st.body)
            elif isinstance(st, ast.If):
                add(st.test, "branch", "If", "If", stmt_text)
                visit_expr(st.test, "If", stmt_text)
                visit_stmts(st.body)
                visit_stmts(st.orelse)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                visit_stmts(st.body)
            elif isinstance(st, ast.Expr):
                # a bare expression statement (usually a Call, e.g. print(x))
                visit_expr(st.value, "Expr", stmt_text)
            elif isinstance(st, ast.Try):
                visit_stmts(st.body)
                for h in st.handlers:
                    visit_stmts(h.body)
                visit_stmts(st.orelse)
                visit_stmts(st.finalbody)
            else:
                # any other statement kind: still scan its sub-expressions
                # for calls/comparisons/etc. even if the statement ITSELF
                # isn't one of the named categories
                visit_expr(st, type(st).__name__, stmt_text)

    body = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = n.body
            break
    if body is None:
        body = tree.body

    visit_stmts(body)

    # dedup exact (site_type, source_text, lineno) repeats -- the SAME
    # sub-expression can be reached twice if it's revisited by both a
    # statement-level walk and a nested visit_expr call
    seen = set()
    uniq = []
    for s in sites:
        key = (s.site_type, s.source_text, s.lineno)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq if max_sites is None else uniq[:max_sites]
