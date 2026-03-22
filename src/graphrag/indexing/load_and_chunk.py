"""Carregar documentos, enriquecer metadados, partir por tokens ou caracteres; persistir Document/TextUnit no Neo4j."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graphrag.config import (
    CHUNK_STRATEGY,
    TEXT_UNIT_CHUNK_OVERLAP_CHARS,
    TEXT_UNIT_CHUNK_OVERLAP_TOKENS,
    TEXT_UNIT_CHUNK_SIZE_CHARS,
    TEXT_UNIT_CHUNK_SIZE_TOKENS,
)
from graphrag.retrieval.token_budget import count_tokens
from graphrag.store.neo4j_graph import get_neo4j_graph


def enrich_document_metadata(docs: List[Document]) -> None:
    """Metadados estilo GraphRAG: título, caminho, extensão, tipo de fonte."""
    for d in docs:
        src = str(d.metadata.get("source") or "")
        if src:
            p = Path(src)
            try:
                abs_path = str(p.resolve())
            except OSError:
                abs_path = src
            d.metadata.setdefault("doc_title", p.stem or "untitled")
            d.metadata.setdefault("file_path", abs_path)
            d.metadata.setdefault("file_ext", p.suffix.lower().lstrip(".") or "txt")
        else:
            d.metadata.setdefault("doc_title", "untitled")
            d.metadata.setdefault("file_path", "")
            d.metadata.setdefault("file_ext", "txt")
        d.metadata.setdefault("source_type", "text_file")


def load_documents_from_dir(
    input_dir: str | Path,
    glob: str = "**/*.txt",
    loader_kwargs: Optional[dict] = None,
) -> List[Document]:
    """Load documents from a directory. Default: .txt files with TextLoader."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(str(input_path))
    loader = DirectoryLoader(
        str(input_path),
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs=loader_kwargs or {"encoding": "utf-8"},
    )
    docs = loader.load()
    enrich_document_metadata(docs)
    return docs


def chunk_documents(
    docs: List[Document],
    *,
    strategy: Optional[str] = None,
) -> List[Document]:
    """
    Partição recursiva: por **tokens** (cl100k_base, default) ou por **caracteres** (legado).
    """
    strat = (strategy or CHUNK_STRATEGY or "tokens").lower().strip()
    if strat == "chars":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_UNIT_CHUNK_SIZE_CHARS,
            chunk_overlap=TEXT_UNIT_CHUNK_OVERLAP_CHARS,
            length_function=len,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_UNIT_CHUNK_SIZE_TOKENS,
            chunk_overlap=TEXT_UNIT_CHUNK_OVERLAP_TOKENS,
            length_function=count_tokens,
        )
    return splitter.split_documents(docs)


def persist_documents_and_chunks_to_neo4j(
    docs: List[Document],
    chunks: List[Document],
    doc_id_attr: str = "source",
) -> List[dict]:
    """
    Cria Document (metadados ricos) e TextUnits com chunk_index por documento,
    token_count / char_count e referência ao ficheiro de origem.
    """
    driver = get_neo4j_graph()._driver
    chunk_records: List[dict] = []
    seen_docs: set[str] = set()
    per_doc_chunk_idx: dict[str, int] = defaultdict(int)

    with driver.session() as session:
        for doc in docs:
            doc_id = str(doc.metadata.get(doc_id_attr) or doc.metadata.get("source") or id(doc))
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.source = $source,
                    d.title = $title,
                    d.file_path = $file_path,
                    d.file_ext = $file_ext,
                    d.source_type = $source_type
                """,
                id=doc_id,
                source=doc_id,
                title=doc.metadata.get("doc_title", ""),
                file_path=doc.metadata.get("file_path", ""),
                file_ext=doc.metadata.get("file_ext", ""),
                source_type=doc.metadata.get("source_type", "text_file"),
            )

        for chunk in chunks:
            doc_id = str(chunk.metadata.get(doc_id_attr) or chunk.metadata.get("source") or "unknown")
            cidx = per_doc_chunk_idx[doc_id]
            per_doc_chunk_idx[doc_id] += 1
            tu_id = f"{doc_id}_{cidx}"
            text = chunk.page_content
            nt = count_tokens(text)
            nc = len(text)
            title = chunk.metadata.get("doc_title", "")
            fp = chunk.metadata.get("file_path", "")
            ext = chunk.metadata.get("file_ext", "")
            st = chunk.metadata.get("source_type", "text_file")

            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.source = $doc_id,
                    d.title = $title,
                    d.file_path = $file_path,
                    d.file_ext = $file_ext,
                    d.source_type = $source_type
                WITH d
                MERGE (t:TextUnit {id: $tu_id})
                SET t.text = $text,
                    t.chunk_index = $chunk_index,
                    t.token_count = $token_count,
                    t.char_count = $char_count,
                    t.doc_title = $title,
                    t.source_file = $file_path
                MERGE (d)-[:HAS_CHUNK]->(t)
                """,
                doc_id=doc_id,
                tu_id=tu_id,
                text=text,
                chunk_index=cidx,
                token_count=nt,
                char_count=nc,
                title=title,
                file_path=fp,
                file_ext=ext,
                source_type=st,
            )
            chunk_records.append({"tu_id": tu_id, "text": text})

    return chunk_records


def run_load_and_chunk(
    input_dir: str | Path,
    *,
    strategy: Optional[str] = None,
) -> List[dict]:
    """Load from directory, chunk (tokens ou chars), persist to Neo4j; return list of {tu_id, text}."""
    docs = load_documents_from_dir(input_dir)
    chunks = chunk_documents(docs, strategy=strategy)
    return persist_documents_and_chunks_to_neo4j(docs, chunks)
