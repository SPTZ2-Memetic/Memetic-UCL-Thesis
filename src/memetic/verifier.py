"""Function-level verifier — runs a candidate function against its unit tests in
an isolated subprocess and classifies the outcome.

Self-contained (assembly, classifier, and outcome in one module), keyed to this
system's records: verify(record, candidate_source) instead of
verify(Task, Solution).

Outcome convention (v(x,c)=1 iff ALL tests pass), via child exit codes:
  0  -> PASSED_ALL     every test green            (fixer: solved | seed: canonical ok)
  42 -> FAILED_TESTS   tests RAN, >=1 failed/errored  -> this is a VALID BUG
  43 -> CRASHED        import/collection failure (no tests ran)  -> NOT a valid bug
  (timeout)-> TIMEOUT
  other nonzero -> CRASHED
The 42/43 split is what makes an *exception-raising* bug count as a valid bug
(a failed test) rather than a crash.

SECURITY: subprocess + hard timeout only. Acceptable for local single-user
research. Do NOT run untrusted code from this on shared/networked machines.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from memetic.types import Record


class ExecutionOutcome(str, Enum):
    PASSED_ALL = "PASSED_ALL"
    FAILED_TESTS = "FAILED_TESTS"
    CRASHED = "CRASHED"
    TIMEOUT = "TIMEOUT"


@dataclass
class VerifyResult:
    outcome: ExecutionOutcome
    output: str

    @property
    def passed(self) -> bool:
        """All tests green (fixer solved it / canonical is valid)."""
        return self.outcome == ExecutionOutcome.PASSED_ALL

    @property
    def is_valid_bug(self) -> bool:
        """Tests ran and at least one failed -> a genuine, non-crashing bug."""
        return self.outcome == ExecutionOutcome.FAILED_TESTS


# ---- module assembly -----------------------------------------------------
# The candidate (full reconstructed function: code_prompt + body) and the test
# code go in candidate.py; a separate runner.py IMPORTS it (so the test file's
# own __main__ guard never fires) and controls the exit code.
_RUNNER = """\
import os, sys, unittest
try:
    import candidate as _m
except BaseException as e:
    sys.stderr.write("IMPORT/COLLECT ERROR: %r\\n" % (e,))
    os._exit(43)
# loadTestsFromModule (not a manual scan) to honour unittest's load_tests
# protocol -- standard unittest discovery. candidate is IMPORTED, not run as
# __main__, so a test file's own `if __name__=='__main__': unittest.main()`
# never fires and cannot hijack the exit code. os._exit: forceful, so leftover
# threads/atexit handlers in the candidate can't hang or alter the code.
_res = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(
    unittest.TestLoader().loadTestsFromModule(_m)
)
if _res.testsRun == 0:
    os._exit(43)   # collection/import failure -> CRASHED
os._exit(0 if _res.wasSuccessful() else 42)
"""


def assemble_candidate_module(candidate_source: str, test_code: str) -> str:
    """candidate.py contents: the function under test + its tests. candidate_source
    must be a FULL function (code_prompt + body), i.e. importable as-is."""
    return candidate_source.rstrip("\n") + "\n\n" + (test_code or "") + "\n"


def _classify(exit_code: int, timed_out: bool) -> ExecutionOutcome:
    if timed_out:
        return ExecutionOutcome.TIMEOUT
    if exit_code == 0:
        return ExecutionOutcome.PASSED_ALL
    if exit_code == 42:
        return ExecutionOutcome.FAILED_TESTS
    return ExecutionOutcome.CRASHED   # 43 or any other nonzero


# ---- verifier ------------------------------------------------------------
class FunctionVerifier:
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_output_chars: int = 10_000,
        python_executable: Optional[str] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        # point at a venv with pinned libs for BigCodeBench if needed; else this interpreter
        self.python_executable = python_executable or sys.executable

    def verify(self, record: Record, candidate_source: str) -> VerifyResult:
        """Run candidate_source (a full function) against record['test'].
        Never raises — every failure becomes a classified VerifyResult."""
        test_code = record.get("test")
        if not test_code:
            return VerifyResult(ExecutionOutcome.CRASHED, "missing 'test' in record")

        module_src = assemble_candidate_module(candidate_source, test_code)
        exit_code, stdout, stderr, timed_out = self._run(module_src)
        outcome = _classify(exit_code, timed_out)
        output = (stderr + stdout)[-self.max_output_chars:]
        return VerifyResult(outcome, output)

    def _run(self, module_src: str) -> Tuple[int, str, str, bool]:
        tmp = tempfile.mkdtemp(prefix="mself_verify_")
        try:
            (Path(tmp) / "candidate.py").write_text(module_src, encoding="utf-8")
            (Path(tmp) / "runner.py").write_text(_RUNNER, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = "0"
            env["MPLBACKEND"] = "Agg"          # headless matplotlib
            try:
                r = subprocess.run(
                    [self.python_executable, "runner.py"],
                    capture_output=True, text=True,
                    timeout=self.timeout_seconds, cwd=tmp, env=env,
                )
                return r.returncode, r.stdout, r.stderr, False
            except subprocess.TimeoutExpired as e:
                so = e.stdout or ""; se = e.stderr or ""
                if isinstance(so, bytes): so = so.decode("utf-8", "replace")
                if isinstance(se, bytes): se = se.decode("utf-8", "replace")
                return -1, so, se, True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def verify_many(
        self,
        pairs: List[Tuple[Record, str]],
        max_workers: Optional[int] = None,
    ) -> List[VerifyResult]:
        """Verify many (record, candidate) pairs concurrently, order preserved.
        Each verify() is its own subprocess; threads just overlap the waiting."""
        if not pairs:
            return []
        workers = max_workers or min(8, (os.cpu_count() or 1))
        results: List[Optional[VerifyResult]] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.verify, rec, cand): i
                    for i, (rec, cand) in enumerate(pairs)}
            for fut in futs:
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = VerifyResult(ExecutionOutcome.CRASHED, f"{type(e).__name__}: {e}")
        return results  # type: ignore[return-value]
