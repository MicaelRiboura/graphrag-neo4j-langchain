"""Token and USD cost monitoring for indexing/query processes."""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import tiktoken
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class UsageRow:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenCostTracker:
    """Thread-safe accumulator for token usage and estimated USD cost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process = "unknown"
        self._by_model: Dict[str, UsageRow] = defaultdict(UsageRow)

    def reset(self, process: str) -> None:
        with self._lock:
            self._process = process.strip().lower() if process else "unknown"
            self._by_model = defaultdict(UsageRow)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        if input_tokens < 0:
            input_tokens = 0
        if output_tokens < 0:
            output_tokens = 0
        key = (model or "unknown-model").strip()
        with self._lock:
            row = self._by_model[key]
            row.input_tokens += int(input_tokens)
            row.output_tokens += int(output_tokens)

    @staticmethod
    def _parse_price_config() -> Dict[str, Dict[str, float]]:
        raw = os.environ.get("GRAPHRAG_MODEL_USD_PER_1K_TOKENS", "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for model, data in (parsed or {}).items():
            if not isinstance(model, str) or not isinstance(data, dict):
                continue
            in_price = float(data.get("input", 0) or 0)
            out_price = float(data.get("output", data.get("input", 0)) or 0)
            out[model] = {"input": in_price, "output": out_price}
        return out

    def _price_for(self, model: str) -> Dict[str, float]:
        cfg = self._parse_price_config()
        return cfg.get(model.strip(), {"input": 0.0, "output": 0.0})

    def summary_lines(self) -> list[str]:
        with self._lock:
            rows = dict(self._by_model)
            process = self._process
        if not rows:
            return [f"[token-monitor] Processo={process} | sem uso de tokens detectado."]
        total_in = 0
        total_out = 0
        total_cost = 0.0
        lines = [f"[token-monitor] Processo={process}"]
        for model, usage in sorted(rows.items()):
            prices = self._price_for(model)
            in_cost = (usage.input_tokens / 1000.0) * prices["input"]
            out_cost = (usage.output_tokens / 1000.0) * prices["output"]
            subtotal = in_cost + out_cost
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_cost += subtotal
            lines.append(
                f"[token-monitor] modelo={model} input={usage.input_tokens} output={usage.output_tokens} "
                f"total={usage.total_tokens} custo_usd={subtotal:.6f}"
            )
        lines.append(
            f"[token-monitor] TOTAL input={total_in} output={total_out} total={total_in + total_out} "
            f"custo_usd={total_cost:.6f}"
        )
        return lines

    def totals(self) -> Dict[str, float]:
        with self._lock:
            rows = dict(self._by_model)
        total_in = 0
        total_out = 0
        total_cost = 0.0
        for model, usage in rows.items():
            prices = self._price_for(model)
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_cost += (usage.input_tokens / 1000.0) * prices["input"]
            total_cost += (usage.output_tokens / 1000.0) * prices["output"]
        return {
            "input_tokens": float(total_in),
            "output_tokens": float(total_out),
            "total_tokens": float(total_in + total_out),
            "cost_usd": total_cost,
        }

    def print_summary(self) -> None:
        for line in self.summary_lines():
            print(line)


TRACKER = TokenCostTracker()


class LangChainUsageCallback(BaseCallbackHandler):
    """Capture token usage emitted by langchain OpenAI responses."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or {}
        model_name = llm_output.get("model_name") or "unknown-model"
        prompt = int(token_usage.get("prompt_tokens", 0) or 0)
        completion = int(token_usage.get("completion_tokens", 0) or 0)
        if prompt or completion:
            TRACKER.add_usage(model_name, prompt, completion)
        return None


_CALLBACK = LangChainUsageCallback()


def tracked_chat_openai(*, model: str, temperature: float, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        callbacks=[_CALLBACK],
    )


class TrackedOpenAIEmbeddings:
    """Embeddings wrapper that estimates token usage via tiktoken."""

    def __init__(self, *, model: str, openai_api_key: str):
        self.model = model
        self._inner = OpenAIEmbeddings(model=model, openai_api_key=openai_api_key)

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        total = sum(self._count_tokens(t or "") for t in texts)
        if total:
            TRACKER.add_usage(self.model, total, 0)
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        total = self._count_tokens(text or "")
        if total:
            TRACKER.add_usage(self.model, total, 0)
        return self._inner.embed_query(text)
