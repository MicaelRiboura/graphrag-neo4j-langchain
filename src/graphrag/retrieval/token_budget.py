"""Contagem de tokens e empacotamento de contexto (alinhado ao orçamento do GraphRAG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if _ENC is None:
        approx_chars = max_tokens * 4
        return text[:approx_chars] + ("\n...[truncated]" if len(text) > approx_chars else "")
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens]) + "\n...[truncated]"


@dataclass
class RankedContextChunk:
    """Candidato único ao contexto local após ranking."""

    role: str
    content: str
    score: float
    metadata: dict

    def token_len(self) -> int:
        return count_tokens(self.content)


def pack_chunks_by_token_budget(
    chunks: Sequence[RankedContextChunk],
    max_tokens: int,
    reserve_per_chunk: int = 0,
) -> List[RankedContextChunk]:
    """
    Ordena por score (desc) e inclui chunks até esgotar max_tokens.
    Chunks grandes são truncados se couberem parcialmente no orçamento restante.
    """
    if max_tokens <= 0:
        return []
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    out: List[RankedContextChunk] = []
    used = 0
    overhead = max(0, reserve_per_chunk)

    for ch in sorted_chunks:
        need = ch.token_len() + overhead
        remaining = max_tokens - used
        if remaining <= 0:
            break
        if need <= remaining:
            out.append(ch)
            used += need
            continue
        # tenta truncar conteúdo para caber no restante
        budget = remaining - overhead
        if budget < 32:
            continue
        truncated = truncate_to_tokens(ch.content, budget)
        if count_tokens(truncated) < 16:
            continue
        tlen = count_tokens(truncated) + overhead
        out.append(
            RankedContextChunk(
                role=ch.role,
                content=truncated,
                score=ch.score,
                metadata={**ch.metadata, "truncated": True},
            )
        )
        used += tlen

    return out
