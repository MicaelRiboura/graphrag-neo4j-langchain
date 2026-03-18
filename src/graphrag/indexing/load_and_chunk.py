"""Load documents and chunk into TextUnits; create Document and TextUnit nodes in Neo4j."""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from graphrag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graphrag.store.neo4j_graph import get_neo4j_graph


# Default chunk size (chars) and overlap
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


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
    return loader.load()


def chunk_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def persist_documents_and_chunks_to_neo4j(
    docs: List[Document],
    chunks: List[Document],
    doc_id_attr: str = "source",
) -> List[dict]:
    """Create Document and TextUnit nodes in Neo4j and link them. Returns list of {tu_id, text} for downstream extraction."""
    driver = get_neo4j_graph()._driver
    chunk_records = []
    with driver.session() as session:
        seen_docs = set()
        for doc in docs:
            doc_id = doc.metadata.get(doc_id_attr) or doc.metadata.get("source", str(id(doc)))
            if doc_id not in seen_docs:
                session.run(
                    "MERGE (d:Document {id: $id}) SET d.source = $source",
                    id=doc_id,
                    source=doc_id,
                )
                seen_docs.add(doc_id)
        for i, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get(doc_id_attr) or chunk.metadata.get("source", "unknown")
            tu_id = f"{doc_id}_{i}"
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                CREATE (t:TextUnit {id: $tu_id, text: $text})
                MERGE (d)-[:HAS_CHUNK]->(t)
                """,
                doc_id=doc_id,
                tu_id=tu_id,
                text=chunk.page_content,
            )
            chunk_records.append({"tu_id": tu_id, "text": chunk.page_content})
    return chunk_records


def run_load_and_chunk(input_dir: str | Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """Load from directory, chunk, persist to Neo4j; return list of {tu_id, text} for extraction."""
    docs = load_documents_from_dir(input_dir)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return persist_documents_and_chunks_to_neo4j(docs, chunks)
