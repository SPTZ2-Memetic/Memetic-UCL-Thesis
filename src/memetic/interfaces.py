"""Policy interface — the one thing every role depends on.

A Policy is "string in, string out," frozen (never trained here). The gateway
client in policy_prometheus.py satisfies this contract directly.

Structural (Protocol), so a test double just needs a `.generate` method — no
subclassing required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    trainable: bool = False

    def generate(self, prompt: str, **kwargs) -> str: ...
