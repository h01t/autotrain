"""LLM client with retry, backoff, and provider abstraction (Anthropic + Ollama)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
import structlog

log = structlog.get_logger()

# Pricing per 1M tokens: (input, output)
_COST_RATES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-6-20250819": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-opus-4-6-20250819": (15.00, 75.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}
_DEFAULT_RATE = (3.00, 15.00)


@dataclass
class AgentResponse:
    """Parsed response from the LLM agent."""

    raw_text: str
    input_tokens: int
    output_tokens: int
    cost_estimate: float  # Rough cost in USD


class AgentClient:
    """LLM client supporting Anthropic and Ollama providers."""

    def __init__(
        self,
        provider: str = "anthropic",
        api_base: str = "http://localhost:11434",
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        hard_timeout_seconds: int = 120,
        temperature: float = 0.3,
    ) -> None:
        self._provider = provider
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        self._hard_timeout = hard_timeout_seconds
        self._temperature = temperature

        if provider == "anthropic":
            import anthropic

            self._anthropic_client = anthropic.Anthropic()

    def call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Call the LLM with retry and backoff."""
        if self._provider == "ollama":
            return self._call_ollama(system_prompt, user_message)
        if self._provider == "deepseek":
            return self._call_openai_compat(
                system_prompt, user_message, max_tokens,
                base_url="https://api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
            )
        return self._call_anthropic(system_prompt, user_message, max_tokens)

    def _call_anthropic(
        self, system_prompt: str, user_message: str, max_tokens: int,
    ) -> AgentResponse:
        """Call Anthropic API with retry and backoff."""
        import anthropic

        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = self._anthropic_client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=self._temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    timeout=self._hard_timeout,
                )

                raw_text = response.content[0].text if response.content else ""
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                in_rate, out_rate = _COST_RATES.get(self._model, _DEFAULT_RATE)
                cost = (input_tokens * in_rate / 1_000_000) + (
                    output_tokens * out_rate / 1_000_000
                )

                log.info(
                    "llm_call_success",
                    provider="anthropic",
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=f"${cost:.4f}",
                    attempt=attempt + 1,
                )

                return AgentResponse(
                    raw_text=raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_estimate=cost,
                )

            except anthropic.RateLimitError as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning("llm_rate_limited", attempt=attempt + 1, backoff=backoff)
                time.sleep(backoff)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    backoff = self._retry_base * (2**attempt)
                    log.warning(
                        "llm_server_error",
                        status=e.status_code,
                        attempt=attempt + 1,
                        backoff=backoff,
                    )
                    time.sleep(backoff)
                else:
                    log.error(
                        "llm_client_error", status=e.status_code, error=str(e),
                    )
                    raise

            except anthropic.APIConnectionError as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning(
                    "llm_connection_error",
                    attempt=attempt + 1,
                    backoff=backoff,
                    error=str(e),
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"Anthropic API unreachable after {self._max_retries} retries"
        ) from last_error

    def _call_ollama(
        self, system_prompt: str, user_message: str,
    ) -> AgentResponse:
        """Call Ollama REST API with retry and backoff."""
        url = f"{self._api_base}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        last_error = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    url, json=payload, timeout=self._hard_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                raw_text = data.get("message", {}).get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)

                log.info(
                    "llm_call_success",
                    provider="ollama",
                    model=self._model,
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    cost="$0.0000",
                    attempt=attempt + 1,
                )

                return AgentResponse(
                    raw_text=raw_text,
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    cost_estimate=0.0,
                )

            except requests.ConnectionError as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning(
                    "llm_connection_error",
                    provider="ollama",
                    attempt=attempt + 1,
                    backoff=backoff,
                    error=str(e),
                )
                time.sleep(backoff)

            except requests.Timeout as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning(
                    "llm_timeout",
                    provider="ollama",
                    attempt=attempt + 1,
                    backoff=backoff,
                )
                time.sleep(backoff)

            except requests.HTTPError as e:
                last_error = e
                if resp.status_code >= 500:
                    backoff = self._retry_base * (2**attempt)
                    log.warning(
                        "llm_server_error",
                        provider="ollama",
                        status=resp.status_code,
                        attempt=attempt + 1,
                    )
                    time.sleep(backoff)
                else:
                    log.error(
                        "llm_client_error",
                        provider="ollama",
                        status=resp.status_code,
                        error=str(e),
                    )
                    raise

        raise RuntimeError(
            f"Ollama API unreachable after {self._max_retries} retries"
        ) from last_error

    def _call_openai_compat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        base_url: str,
        api_key_env: str,
    ) -> AgentResponse:
        """Call an OpenAI-compatible API (DeepSeek, etc.) with retry."""
        import os

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"${api_key_env} environment variable not set")

        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        last_error = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers,
                    timeout=self._hard_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                raw_text = choice["message"]["content"]
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                in_rate, out_rate = _COST_RATES.get(
                    self._model, _DEFAULT_RATE,
                )
                cost = (input_tokens * in_rate / 1_000_000) + (
                    output_tokens * out_rate / 1_000_000
                )

                log.info(
                    "llm_call_success",
                    provider=self._provider,
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=f"${cost:.4f}",
                    attempt=attempt + 1,
                )

                return AgentResponse(
                    raw_text=raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_estimate=cost,
                )

            except requests.ConnectionError as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning(
                    "llm_connection_error",
                    provider=self._provider,
                    attempt=attempt + 1,
                    backoff=backoff,
                    error=str(e),
                )
                time.sleep(backoff)

            except requests.Timeout as e:
                last_error = e
                backoff = self._retry_base * (2**attempt)
                log.warning(
                    "llm_timeout",
                    provider=self._provider,
                    attempt=attempt + 1,
                    backoff=backoff,
                )
                time.sleep(backoff)

            except requests.HTTPError as e:
                last_error = e
                if resp.status_code >= 500 or resp.status_code == 429:
                    backoff = self._retry_base * (2**attempt)
                    log.warning(
                        "llm_server_error",
                        provider=self._provider,
                        status=resp.status_code,
                        attempt=attempt + 1,
                    )
                    time.sleep(backoff)
                else:
                    log.error(
                        "llm_client_error",
                        provider=self._provider,
                        status=resp.status_code,
                        error=resp.text[:200],
                    )
                    raise

        raise RuntimeError(
            f"{self._provider} API unreachable after {self._max_retries} retries"
        ) from last_error
