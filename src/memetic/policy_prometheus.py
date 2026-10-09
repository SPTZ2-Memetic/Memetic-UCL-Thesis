"""LLM policy backed by an OpenAI-compatible gateway.

Same Policy contract as a plain OpenAI client; differs only in the base URL and
an optional custom CA certificate required to trust a private endpoint. Frozen —
called, never trained.
"""
from __future__ import annotations
import httpx
from openai import OpenAI

from .interfaces import Policy

class PrometheusPolicy(Policy):
    trainable = False

    def __init__(self, model: str, api_base: str, ca_cert: str,
                 api_key_env: str = "GATEWAY_API_KEY",
                 temperature: float = 0.2, max_tokens: int = 2048,
                 system_prompt: str | None = None,
                 timeout_seconds: float = 120.0):
        import os
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        # Custom HTTP client that trusts the gateway's CA (ca_cert may be a path
        # to a private CA bundle, or True to use the system trust store).
        # IMPORTANT: explicit timeout -- httpx's default is 5s (connect/read/
        # write/pool), far too short for real LLM completions at max_tokens
        # this size. Without this, a healthy-but-slow call gets killed by the
        # client itself, triggers the openai SDK's automatic retry-with-
        # backoff, and each retry re-pays the full generation cost -- this
        # was silently producing the multi-minute gaps observed in practice.
        http_client = httpx.Client(
            verify=ca_cert,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        self._client = OpenAI(
            api_key=os.environ[api_key_env],
            base_url=f"{api_base}/v1",
            http_client=http_client,
            max_retries=2,  # explicit, so it's a deliberate choice, not a default
        )

    def generate(self, prompt: str, **kwargs) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

