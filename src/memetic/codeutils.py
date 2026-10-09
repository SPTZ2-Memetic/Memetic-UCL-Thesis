"""Shared code-parsing utilities.

Used by both loader.py (normalising stored solutions to bodies) and
generator.py (parsing model output). One home, no duplication.

  parse_fenced_block   - pull code out of a ```python block, with fallback
  extract_function_body - AST-based body extraction, 4-space normalised
  is_self_labeled       - detect bugs that announce themselves in a comment
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Optional


_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def strip_fence(code: str) -> str:
    """Return the last fenced block's contents, or the input stripped."""
    if not code:
        return ""
    blocks = _FENCE_RE.findall(code)
    return (blocks[-1] if blocks else code).strip("\n")


def reindent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + ln) if ln.strip() else "" for ln in text.split("\n"))


def parse_fenced_block(model_output: str) -> tuple[str, bool]:
    """Extract code from the first ```python block in model output.

    Returns (code, fell_back). fell_back=True when no proper fenced block was
    found and the whole output was used instead.
    """
    pattern = re.compile(r"^```python\s*$", re.MULTILINE)
    match = pattern.search(model_output)
    if match is None:
        return model_output.strip("\n"), True

    start = match.end() + 1
    close = re.compile(r"^```\s*$", re.MULTILINE).search(model_output, start)
    if close is None:
        return model_output[start:].strip("\n"), True

    code = model_output[start:close.start()]
    lines = code.split("\n")
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines), False


def _all_fenced_python_blocks(model_output: str) -> list[str]:
    """Every ```python ... ``` block in model output, in order (not just the
    first). Each trimmed the same way parse_fenced_block trims its single
    block."""
    out = []
    pos = 0
    open_pat = re.compile(r"^```python\s*$", re.MULTILINE)
    close_pat = re.compile(r"^```\s*$", re.MULTILINE)
    while True:
        m = open_pat.search(model_output, pos)
        if m is None:
            break
        start = m.end() + 1
        c = close_pat.search(model_output, start)
        end = c.start() if c is not None else len(model_output)
        code = model_output[start:end]
        lines = code.split("\n")
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        out.append("\n".join(lines))
        pos = (c.end() if c is not None else len(model_output))
    return out


def parse_and_extract_function(model_output: str, entry_point: str):
    """Find the fenced ```python block that actually contains `def
    entry_point`, and return its extracted body -- regardless of WHICH
    fenced block it is (first, last, or middle).

    CONFIRMED BUG this replaces (see project notes): the model sometimes
    writes a short illustrative snippet in its OWN fenced block before
    giving the real, complete answer in a LATER block -- e.g. when primed by
    diff-style retrieved examples ("the fix should look like: `if not
    x.strip():`") before the full corrected function. The old
    parse_fenced_block() always took the FIRST fenced block unconditionally,
    so it would extract the illustrative fragment (no `def entry_point` in
    it -> parse failure) and discard a perfectly valid answer sitting in the
    next block. Measured: this inflated the with-bank held-out eval's
    "CRASHED" (== unparseable) rate by ~23%% relative to baseline, and every
    such failure is an automatic wrong answer regardless of whether the
    actual code would have passed.

    Returns (body, fell_back): body is the extracted function body or None;
    fell_back mirrors parse_fenced_block's meaning (True if no proper fenced
    block was used, i.e. we fell through to the whole-output fallback)."""
    for block in _all_fenced_python_blocks(model_output):
        body = extract_function_body(block, entry_point)
        if body is not None:
            return body, False
    parsed, fell_back = parse_fenced_block(model_output)
    return extract_function_body(parsed, entry_point), fell_back


def extract_function_body(source: str, entry_point: str) -> Optional[str]:
    """AST-locate the function named entry_point, return its body normalised to
    4-space indent (def line + imports stripped). None if unparseable or the
    named function isn't present. Nested-block relative indent is preserved.

    Strict on the name: a renamed function returns None (a signature change is
    an invalid bug, per the prompt's "do not change the signature").
    """
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            target = node
            break
    if target is None or not target.body:
        return None

    lines = source.splitlines()
    start = target.body[0].lineno - 1     # first body stmt (skips multi-line sig)
    end = getattr(target, "end_lineno", len(lines)) or len(lines)
    raw = "\n".join(lines[start:end])
    dedented = textwrap.dedent(raw)
    normalized = ["    " + ln if ln.strip() else "" for ln in dedented.splitlines()]
    return "\n".join(normalized) + "\n"


_SELF_LABEL_RE = re.compile(r"#.*(bug|buggy|incorrect(ly)?)", re.IGNORECASE)


def is_self_labeled(body: str) -> bool:
    """True if the body announces its bug in a comment ('# Bug: ...'). Small
    models do this ~60-70% of the time despite the prompt forbidding it; this is
    the enforcement the prompt can't provide."""
    return bool(_SELF_LABEL_RE.search(body or ""))


def strip_self_label(body: str) -> str:
    """Remove bug-announcing comments ('# Bug: ...', '# buggy', '# incorrect')
    so a self-labeled generation can be RECOVERED rather than discarded -- the
    giveaway is the comment, not the code (confirmed on real qwen-coder output:
    a valid inverted-division bug rejected purely for a trailing '# Bug:'
    note). Only comments matching the bug pattern are cut; other comments are
    left alone; a line that was ONLY a bug comment is dropped. Line-based, so a
    '#' inside a string on the same line as a bug comment could mis-cut -- but
    then the caller's parse gate rejects it, exactly as a self-labeled body was
    rejected before, so this can only ever recover candidates, never make
    things worse."""
    out = []
    for line in (body or "").splitlines():
        if _SELF_LABEL_RE.search(line):
            hash_idx = line.find("#")
            if hash_idx != -1:
                head = line[:hash_idx].rstrip()
                if head == "":
                    continue          # whole line was the bug comment -> drop
                out.append(head)
                continue
        out.append(line)
    text = "\n".join(out)
    return text + "\n" if body.endswith("\n") else text


# ---- verifier error -> compact signature (for generator memory) -----------
import re as _re

def error_signature(verify_output: str, max_len: int = 160) -> str:
    """Compact one-line signature of a bug's verifier failure, for storing on a
    generator Case and rendering into the prompt. Extracts the exception type +
    its message (the last 'Error/Exception: ...' line), or the assertion diff.
    Falls back to the last non-empty line. Empty string if nothing useful."""
    if not verify_output:
        return ""
    lines = [l.rstrip() for l in verify_output.splitlines() if l.strip()]
    if not lines:
        return ""
    # prefer the last line matching an exception pattern
    for l in reversed(lines):
        m = _re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Failure)):?\s*(.*)$", l.strip())
        if m:
            cls, msg = m.group(1), m.group(2)
            sig = f"{cls}: {msg}".strip().rstrip(":")
            return sig[:max_len]
    # else the last line (often the assertion summary)
    return lines[-1][:max_len]


def error_kind(verify_output: str) -> str:
    """Coarse bucket of the failure, for later tactic-keying. One of:
    'assertion' (wrong value, tests ran), 'index', 'key', 'type', 'value',
    'attribute', 'name', 'timeout', 'other'."""
    sig = error_signature(verify_output).lower()
    if "timeout" in (verify_output or "").lower():
        return "timeout"
    for k in ("indexerror", "keyerror", "typeerror", "valueerror",
              "attributeerror", "nameerror", "zerodivisionerror"):
        if k in sig.replace(" ", ""):
            return k.replace("error", "")
    if "assert" in sig or "differ" in sig:
        return "assertion"
    return "other"


CRASH_KINDS = frozenset({"index", "key", "type", "value", "attribute", "name", "zerodivision"})
# error_kind buckets that correspond to a genuine raised exception with a real
# traceback frame INSIDE the candidate function -- these can be LOCALIZED to
# an exact source line (see extract_traceback_line). 'assertion'/'other'/
# 'timeout' have NO such frame (the code ran to completion, or hung) -- there
# is structurally nothing to localize for these, confirmed by direct
# inspection of real unittest output.


def extract_traceback_line(test_output: str, entry_point: str) -> str:
    """The EXACT source line, verbatim, of the frame matching `in
    {entry_point}` in a real traceback -- i.e. the specific statement that
    raised, inside the buggy function itself (not the test's call site).
    Only meaningful for CRASH_KINDS; returns "" if no such frame is found
    (e.g. an assertion failure, which has no frame inside the candidate).

    Skips Python's "pointer" lines (the '~^^^^^' markers under the faulting
    sub-expression in 3.11+ tracebacks) -- those aren't source code."""
    lines = (test_output or "").splitlines()
    frame_pat = re.compile(rf'in {re.escape(entry_point)}\s*$')
    for i, l in enumerate(lines):
        if frame_pat.search(l) and 'File "' in l:
            for j in range(i + 1, min(i + 3, len(lines))):
                cand = lines[j]
                if cand.strip() and not re.fullmatch(r"[\s~^]*", cand):
                    return cand.strip()
    return ""


def extract_test_docstring(test_output: str) -> str:
    """The FIRST failing test's docstring, as printed by unittest verbosity=2
    (the SAME output already shown to the fixer verbatim -- this extracts,
    it does not add, information). The docstring appears on its own line
    right after the test name line, formatted '{docstring} ... {result}'.
    Returns "" if the first failing test has no docstring (name+result on
    one line instead)."""
    lines = (test_output or "").splitlines()
    for l in lines:
        m = re.match(r"^(.*?)\s+\.\.\.\s+(FAIL|ERROR)\s*$", l)
        if m:
            desc = m.group(1).strip()
            if re.match(r"^test_\w+ \(.*\)$", desc):
                return ""   # name-line itself, no docstring present
            return desc
    return ""


def normalize_diff(diff: str) -> str:
    """Collapse whitespace so near-identical diffs (differing only in
    spacing/blank lines) compare equal -- the shared definition of
    "duplicate" used by BOTH the offline dedup script and the live write-
    gate, so they can never silently drift apart."""
    lines = [re.sub(r"\s+", " ", l).strip() for l in (diff or "").splitlines()]
    return "\n".join(l for l in lines if l)


# NOTE: the old overlap_ratio/scope_of local-edit-vs-reimplementation classifier
# has been REMOVED. Every skill is now a single type (a localized edit); the
# classifier is memetic.constructs.changed_construct's `aligned` flag, which
# diffs canonical vs buggy at the AST-statement level instead of a fuzzy
# whole-body similarity ratio. See seed.make_local_skill.
