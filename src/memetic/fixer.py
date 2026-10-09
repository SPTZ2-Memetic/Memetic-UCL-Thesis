"""Fixer role — attempts to repair a buggy function. Phase-1: MEMORY-LESS.

Difficulty oracle for the generator: its solve-rate over K attempts is what
productivity_reward consumes. The memory-less prompt is the clean baseline;
memory injection is a phase-2 hook (the sentinel insertion point is here; the
renderer comes with the fixer-Case design, since our current Case stores no fix
action).

The <FAILED_TEST_OUTPUT> slot is populated even on the FIRST attempt, from the
bug's own failing tests — so the fixer sees why the code fails rather than
guessing blind. On retries the slot carries the previous attempt's output. The
fixer never sees the canonical solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from memetic.types import Record, Case, FixerCase
from memetic.seed import short_diff
from memetic.interfaces import Policy
from memetic.codeutils import (parse_and_extract_function, extract_function_body,
                                   extract_test_docstring, error_signature, normalize_diff,
                                   CRASH_KINDS, extract_traceback_line, error_kind)
from memetic.verifier import FunctionVerifier, VerifyResult, ExecutionOutcome


@dataclass
class FixAttempt:
    passed: bool
    body: Optional[str]
    reconstructed: Optional[str]
    outcome: ExecutionOutcome
    output: str                     # verifier stderr/stdout (the error, on failure)
    parse_ok: bool = True
    prompt: str = ""                # the fix prompt submitted (for tracing)
    raw_output: str = ""            # raw model output received (for tracing)
    state_key: str = ""             # the STATE queried against fixer memory (for tracing)
    retrieved_rendered: str = ""    # the exact rendered memory block shown (for tracing)
    retrieved_case_ids: List[str] = field(default_factory=list)
    retrieved_summary: List[dict] = field(default_factory=list)  # [{case_id,passed,fix_diff}]


@dataclass
class FixResult:
    solve_rate: float               # passes / attempts_made
    solved: bool
    attempts: List[FixAttempt] = field(default_factory=list)


# Baseline fixer prompt. Keep pristine; memory inserts at the sentinel.
FIXER_PROMPT = """\
You are an expert Python debugging assistant.

You will be given:

1. A problem description.
2. A buggy Python implementation that may fail some hidden unit tests.
3. (Optional) Failed unit test output from running the buggy implementation.

Your task:

- Carefully read the code and identify the bug(s).
- Produce a fixed version of the code that makes all unit tests pass.
- Preserve the original function signature, imports, and I/O format.
- Keep the solution reasonably close to the given implementation.
- Output **only** the full corrected Python code inside a single ```python``` block.

Problem:
<PROBLEM>

Buggy implementation:
<BUGGY_CODE>

Failed unit test output (if available):
```text
<FAILED_TEST_OUTPUT>
```

Now fix the bugs in this code. Return the entire function with the fixed code inside a ```python``` block:
"""

_FIX_SENTINEL = "Now fix the bugs in this code."


def extract_problem(record: Record) -> str:
    instruct = record.get("instruct_prompt")
    if not instruct or not instruct.strip():
        instruct = record.get("complete_prompt") or record.get("code_prompt") or ""
    return instruct


def fixer_state_parts(record: Record, buggy_code: str, test_output: str):
    """The fixer's state, as TWO SEPARATE embeddable parts:
      (code_text, error_text) = (task + buggy code, compact error signature)
    Kept separate -- not concatenated -- because the same code can fail with
    different errors needing different fixes, and mean-pooling a single
    combined string lets one channel's length dominate the other. See
    Memory.read's error_query_text/error_weight, which combines their
    similarities with an explicit weight instead."""
    from memetic.codeutils import error_signature
    q = extract_problem(record)
    code_text = f"{q}\n{buggy_code}".strip()
    error_text = error_signature(test_output or "")
    return code_text, error_text


def fixer_state_key(record: Record, buggy_code: str, test_output: str) -> str:
    """Legacy single-string form (task+code+error concatenated), kept for
    trace/display purposes only. Retrieval uses fixer_state_parts's two
    SEPARATE channels (see Memory.read), not this concatenation -- concat-
    enating them for embedding was the confirmed bug (raw traceback
    boilerplate swamped the signal; even with the compact error_signature
    fix alone, concatenation still risks the error being under-weighted
    purely by relative text length)."""
    code_text, error_text = fixer_state_parts(record, buggy_code, test_output)
    return f"{code_text}\n{error_text}".strip()


def build_fix_prompt(
    record: Record,
    bug_body: str,
    test_output: str = "",
    retrieved_cases: Optional[Sequence["FixerCase"]] = None,
) -> str:
    """Fill the fixer template. Pure (no policy, no I/O).

    With retrieved_cases falsy, byte-identical to the baseline fixer prompt — the
    baseline invariant. bug_body is the BODY; the displayed buggy code is
    reconstructed as code_prompt + bug_body (same form the verifier runs)."""
    buggy_code = (record.get("code_prompt", "") or "") + bug_body

    prompt = (
        FIXER_PROMPT
        .replace("<PROBLEM>", extract_problem(record))
        .replace("<BUGGY_CODE>", buggy_code)
        .replace("<FAILED_TEST_OUTPUT>", test_output or "")
    )

    if retrieved_cases:
        block = render_fixer_cases(retrieved_cases)
        prompt = prompt.replace(_FIX_SENTINEL, block + "\n" + _FIX_SENTINEL, 1)
    return prompt


def explain_bug_general(diff: str, policy, test_context: str = "") -> str:
    """Fixer-scoped: the GENERAL principle/category of mistake a fix
    illustrates, phrased so it could apply to a DIFFERENT bug in different
    code -- not a caption of this one diff.

    test_context (optional): the failing test's docstring + compact error
    signature (see codeutils.extract_test_docstring/error_signature) --
    from the SAME test_output already shown to the fixer verbatim in
    <FAILED_TEST_OUTPUT> (see fixer_state_parts/FIXER_PROMPT); this is
    NOT new information, it lets the model correctly identify WHAT KIND of
    mistake this is (e.g. an off-by-one on a boundary case, vs. a wrong
    aggregation axis) rather than guessing purely from the diff in
    isolation. The prompt explicitly forbids the output from referencing
    this specific test/scenario -- context is for THIS call's reasoning
    only, never for the returned sentence, which must stay general."""
    if not diff.strip():
        return ""
    context_block = ""
    if test_context.strip():
        context_block = (
            "\nFor context, here is what the failing test reported before the "
            "fix (use this ONLY to correctly identify what KIND of mistake this "
            "is -- your answer must NOT mention these specific values, this "
            f"test, or this scenario):\n\n{test_context}\n"
        )
    prompt = (
        "A bug was fixed by this exact change (unified diff, '-' = removed, "
        "'+' = added):\n\n"
        f"{diff}\n"
        f"{context_block}\n"
        "In ONE short sentence, state the GENERAL principle or common category "
        "of mistake this illustrates -- phrased so it applies to a DIFFERENT "
        "bug in DIFFERENT code sharing the same underlying kind of error, not "
        "this specific instance. Do NOT mention variable names, specific "
        "numbers or values, this function, this test, or this scenario. "
        "Reply with only the sentence."
    )
    try:
        out = policy.generate(prompt).strip().splitlines()
        line = next((l.strip() for l in out if l.strip()), "")
        return line[:200]
    except Exception:
        return ""


def render_fixer_cases(cases: Sequence["FixerCase"]) -> str:
    """Render retrieved past fix ATTEMPTS (both passed and failed), clearly
    labelled so the model knows exactly which to imitate and which to avoid.
    Memento-style: cases are shown as retrieved (cosine top-K, no reward
    reranking) -- the ordering here (passed first) is DISPLAY ONLY."""
    if not cases:
        return ""
    passed = [c for c in cases if c.passed]
    failed = [c for c in cases if not c.passed]
    out = ["\nHere are past attempts at fixing SIMILAR bugs, and whether they "
           "worked. Use them as guidance for THIS fix — do not copy verbatim, "
           "adapt the idea to this code."]
    if passed:
        out.append("\nThis WORKED before — you SHOULD try something similar:")
        for i, c in enumerate(passed, 1):
            expl = f"  reason it worked: {c.explanation}" if c.explanation else ""
            out.append(f"  {i}. fix made: {c.fix_diff}\n{expl}".rstrip())
    if failed:
        out.append("\nThis did NOT work before — you should NOT try the same thing:")
        for i, c in enumerate(failed, 1):
            out.append(f"  {i}. fix attempted (failed): {c.fix_diff}")
    out.append("")
    return "\n".join(out)


def _one_attempt(record: Record, bug_body: str, policy: Policy,
                 verifier: FunctionVerifier, test_output: str,
                 memory=None, k_retrieve: int = 4, error_weight: float = 0.4):
    """One fix attempt. If `memory` is given: build the state as TWO separate
    parts (task+code, compact error), then branch retrieval by error KIND:
      - CRASH_KINDS (a real traceback frame exists inside the candidate
        function): extract that EXACT source line and use it as a precise
        construct query (Memory.read_by_buggy_constructs) -- structurally
        cannot repeat the earlier "search the whole function" localization
        flaw, since there is no ambiguity about which line to search with.
      - everything else (assertion/other/timeout -- NO frame exists inside
        the candidate, confirmed by direct inspection of real unittest
        output): construct-matching has nothing reliable to localize to, so
        fall back to the existing whole-code+error blended retrieval
        (Memory.read), unchanged.
    Falls back to the assertion-branch behavior if extraction fails even for
    a crash-kind error (e.g. entry_point name mismatch) -- never silently
    returns nothing just because localization didn't work.
    Renders both passed and failed past attempts, THEN generates.
    Returns (FixAttempt, retrieved_cases, code_text, error_text) so the
    caller can write a case back using the SAME state/retrieval context."""
    entry_point = record.get("entry_point") or "task_func"
    buggy_code = (record.get("code_prompt", "") or "") + bug_body
    code_text, error_text = fixer_state_parts(record, buggy_code, test_output)
    state_key = f"{code_text}\n{error_text}".strip()   # for tracing/display only

    retrieved = []
    if memory is not None:
        kind = error_kind(test_output)
        localized_line = ""
        if kind in CRASH_KINDS:
            localized_line = extract_traceback_line(test_output, entry_point)
        if localized_line and hasattr(memory, "read_by_buggy_constructs"):
            retrieved = memory.read_by_buggy_constructs(
                localized_line, k=k_retrieve,
                error_query_text=error_text, error_weight=error_weight)
        else:
            retrieved = memory.read(code_text, k=k_retrieve,
                                    error_query_text=error_text, error_weight=error_weight)
    rendered = render_fixer_cases(retrieved) if retrieved else ""
    fprompt = build_fix_prompt(record, bug_body, test_output, retrieved_cases=retrieved)

    retr_ids = [c.case_id for c in retrieved]
    retr_summary = [{"case_id": c.case_id, "passed": c.passed, "fix_diff": c.fix_diff,
                     "explanation": c.explanation} for c in retrieved]

    raw = policy.generate(fprompt)
    body, _ = parse_and_extract_function(raw, entry_point)
    if body is None:
        att = FixAttempt(False, None, None, ExecutionOutcome.CRASHED,
                         "fix did not parse into a function body", parse_ok=False,
                         prompt=fprompt, raw_output=raw, state_key=state_key,
                         retrieved_rendered=rendered, retrieved_case_ids=retr_ids,
                         retrieved_summary=retr_summary)
        return att, retrieved, code_text, error_text, buggy_code
    reconstructed = (record.get("code_prompt", "") or "") + body
    res: VerifyResult = verifier.verify(record, reconstructed)
    att = FixAttempt(res.passed, body, reconstructed, res.outcome, res.output,
                     prompt=fprompt, raw_output=raw, state_key=state_key,
                     retrieved_rendered=rendered, retrieved_case_ids=retr_ids,
                     retrieved_summary=retr_summary)
    return att, retrieved, code_text, error_text, buggy_code


def fix_bug(
    record: Record,
    bug_body: str,
    policy: Policy,
    verifier: FunctionVerifier,
    *,
    k: int = 4,
    bug_test_output: str = "",
    stop_on_first_pass: bool = False,
    memory=None,                # fixer Memory (separate bank from the generator's)
    fixer_read_k: int = 4,       # cases retrieved from fixer memory PER ATTEMPT
    error_weight: float = 0.4,  # weight on the standardized error channel vs
                                # code channel (see Memory.read); calibrated
                                # default, see project notes
    use_generalized_explanation: bool = False,  # False (default) = explain_bug,
                                # the ORIGINAL specific-instance caption --
                                # exact backward compatibility. True =
                                # explain_bug_general, which asks for the
                                # TRANSFERABLE principle instead of a caption
                                # of this one diff -- confirmed in the oracle
                                # experiment to measurably help (oracle_general
                                # beat plain oracle on IDENTICAL underlying
                                # content, same fix_diff, different phrasing
                                # only). Opt-in, not yet the default, pending
                                # a fuller validation pass on the real write
                                # path (only tested via the oracle experiment
                                # so far).
    bug_case_id: Optional[str] = None,   # links FixerCases back to the generator Case
    step: Optional[int] = None,
    explain_policy: Optional[Policy] = None,  # defaults to `policy` if None
    explain_success_only: bool = True,
    write_back: bool = True,     # if False, RETRIEVE from memory but never WRITE
                                 # to it -- for held-out evaluation, so scoring a
                                 # bank never mutates it.
) -> FixResult:
    """K attempts on one bug. Each attempt: retrieve from `memory` (if given)
    using the CURRENT state (task+buggy+latest error) -- re-retrieved every
    attempt, per design -- generate, verify, then WRITE a FixerCase for this
    attempt (both pass and fail) back into `memory`. solve_rate = passes/K.

    Attempt 1 sees the bug's own failing output (`bug_test_output`); each later
    attempt sees the previous attempt's output, both for the prompt's
    <FAILED_TEST_OUTPUT> slot AND as part of the next retrieval's state key."""
    from memetic.types import FixerCase
    from memetic.seed import explain_bug

    attempts: List[FixAttempt] = []
    passes = 0
    feedback = bug_test_output
    explainer = explain_policy or policy

    for i in range(k):
        att, retrieved, code_text, error_text, buggy_code = _one_attempt(
            record, bug_body, policy, verifier, feedback, memory=memory,
            k_retrieve=fixer_read_k, error_weight=error_weight)
        attempts.append(att)
        passes += int(att.passed)

        if write_back and memory is not None and att.body is not None:
            diff = short_diff(bug_body, att.body)
            if diff.strip() and diff != "no textual change":
                # LIVE DUPLICATE GATE: same (task_id, normalized diff) as an
                # EXISTING case in the bank -> skip the write entirely. Uses
                # the SAME normalize_diff as the offline dedup script so the
                # two can never silently disagree on what counts as a
                # duplicate. Directly targets the CONFIRMED failure mode: one
                # task recurring across self-play steps writing many near-
                # identical cases (measured: 74% of a real bank was exactly
                # this).
                task_id = record.get("task_id")
                norm = normalize_diff(diff)
                is_dup = any(
                    c.task_id == task_id and normalize_diff(c.fix_diff) == norm
                    for c in memory.cases
                )
                if not is_dup:
                    case = FixerCase(
                        key_text=code_text, error_text=error_text,
                        question=extract_problem(record),
                        buggy_code=buggy_code, error=feedback, fix_diff=diff,
                        passed=att.passed, reward=1.0 if att.passed else 0.0,
                        bug_case_id=bug_case_id, task_id=task_id,
                        attempt_idx=i, step=step,
                    )
                    if explainer is not None and (att.passed or not explain_success_only):
                        if use_generalized_explanation:
                            # grounded-but-non-leaking context: the SAME
                            # test_output already shown to the fixer verbatim,
                            # just extracted down to docstring+signature --
                            # not new information (see explain_bug_general).
                            test_context = " ".join(
                                x for x in (extract_test_docstring(feedback),
                                           error_signature(feedback)) if x)
                            case.explanation = explain_bug_general(diff, explainer, test_context)
                        else:
                            case.explanation = explain_bug(diff, explainer)
                    memory.write(case)

        feedback = att.output        # thread into next prompt AND next retrieval state
        if att.passed and stop_on_first_pass:
            break

    made = len(attempts)
    return FixResult(solve_rate=passes / made if made else 0.0,
                     solved=passes > 0, attempts=attempts)
