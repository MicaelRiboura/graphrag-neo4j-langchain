"""Resolução canónica de entidades (variações de nome → mesma entidade no grafo), estilo GraphRAG."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from graphrag.config import ENTITY_RESOLUTION_MAX_ALIASES

def normalize_token(s: str) -> str:
    """Normaliza nome/tipo para comparação (case, espaços)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def make_entity_key(name: str, entity_type: str) -> str:
    """
    Chave estável para MERGE: `nome_normalizado|tipo_normalizado`.
    Dois extratos com "Texas" / "TEXAS" e mesmo tipo colidem na mesma entidade.
    """
    nt = normalize_token(entity_type)
    return f"{normalize_token(name)}|{nt if nt else 'entity'}"


def merge_canonical_name_and_aliases(
    existing_name: Optional[str],
    existing_aliases: Optional[List[str]],
    incoming_name: str,
    *,
    max_aliases: int = ENTITY_RESOLUTION_MAX_ALIASES,
) -> Tuple[str, List[str]]:
    """Escolhe nome canónico (heurística: string mais longa) e lista de aliases únicos."""
    names = set()
    for a in existing_aliases or []:
        t = (a or "").strip()
        if t:
            names.add(t)
    en = (existing_name or "").strip()
    if en:
        names.add(en)
    inc = (incoming_name or "").strip()
    if inc:
        names.add(inc)
    if not names:
        return inc or "", []
    canonical = max(names, key=len)
    aliases = sorted(n for n in names if n != canonical)[:max(0, max_aliases)]
    return canonical, aliases


def backfill_entity_keys_cypher(session) -> None:
    """Preenche `entity_key` em entidades antigas (antes da resolução)."""
    session.run(
        """
        MATCH (e:Entity)
        WHERE e.entity_key IS NULL AND e.name IS NOT NULL
        SET e.entity_key = toLower(trim(e.name)) + '|' + toLower(trim(toString(coalesce(e.type, 'entity'))))
        """
    )


def lookup_entity_key_by_surface_form(session, surface: str) -> Optional[str]:
    """Resolve texto livre contra `name` ou `aliases` no Neo4j."""
    raw = (surface or "").strip()
    if not raw:
        return None
    row = session.run(
        """
        MATCH (e:Entity)
        WHERE toLower(trim(e.name)) = toLower(trim($raw))
           OR ANY(a IN coalesce(e.aliases, []) WHERE toLower(trim(toString(a))) = toLower(trim($raw)))
        RETURN e.entity_key AS k
        LIMIT 1
        """,
        raw=raw,
    ).single()
    return str(row["k"]) if row and row.get("k") else None


def resolve_surface_to_entity_key(
    session,
    surface: str,
    norm_to_keys: Dict[str, List[str]],
) -> Optional[str]:
    """Resolve menção textual → entity_key (chunk atual + Neo4j)."""
    raw = (surface or "").strip()
    if not raw:
        return None
    nn = normalize_token(raw)
    keys = norm_to_keys.get(nn, [])
    uniq = list(dict.fromkeys(keys))
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) > 1:
        return uniq[0]
    return lookup_entity_key_by_surface_form(session, raw)
