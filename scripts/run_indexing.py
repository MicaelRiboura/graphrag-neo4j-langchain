"""Run the full GraphRAG indexing pipeline: load -> chunk -> extract -> communities -> reports -> embed."""

import argparse
import sys
from pathlib import Path

# Project root and src on path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

_env = root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

from graphrag.indexing.load_and_chunk import run_load_and_chunk
from graphrag.indexing.extract_graph import run_extract_on_chunks
from graphrag.indexing.communities import run_communities
from graphrag.indexing.reports import run_reports
from graphrag.indexing.embed import run_embed_all


def main():
    parser = argparse.ArgumentParser(description="GraphRAG indexing pipeline")
    parser.add_argument("--input-dir", "-i", type=str, default="./docs", help="Directory with .txt documents")
    parser.add_argument("--skip-load", action="store_true", help="Skip load/chunk (use existing Neo4j data)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip entity/relationship extraction")
    parser.add_argument("--skip-communities", action="store_true", help="Skip community detection")
    parser.add_argument("--skip-reports", action="store_true", help="Skip community reports")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    chunk_records = []

    if not args.skip_load:
        if not input_path.is_dir():
            print(f"Input directory not found: {input_path}")
            sys.exit(1)
        print("Loading and chunking documents...")
        chunk_records = run_load_and_chunk(input_path)
        print(f"Created {len(chunk_records)} TextUnits.")
    else:
        # For skip_load we need chunk_records for extract; if skip_extract too, we skip both
        if not args.skip_extract:
            from graphrag.store.neo4j_graph import get_neo4j_graph
            driver = get_neo4j_graph()._driver
            with driver.session() as session:
                result = session.run("MATCH (t:TextUnit) RETURN t.id AS tu_id, t.text AS text")
                chunk_records = [{"tu_id": r["tu_id"], "text": r["text"] or ""} for r in result]
            print(f"Loaded {len(chunk_records)} TextUnits from Neo4j for extraction.")
        else:
            chunk_records = []

    if chunk_records and not args.skip_extract:
        print("Extracting entities and relationships...")
        run_extract_on_chunks(chunk_records)
        print("Extraction done.")

    if not args.skip_communities:
        print("Running community detection...")
        comms = run_communities()
        print(f"Found {len(comms)} communities.")
    else:
        print("Skipping community detection.")

    if not args.skip_reports:
        print("Generating community reports...")
        run_reports()
        print("Reports done.")
    else:
        print("Skipping reports.")

    if not args.skip_embed:
        print("Computing and storing embeddings...")
        run_embed_all()
        print("Embedding done.")
    else:
        print("Skipping embedding.")

    print("Indexing pipeline finished.")


if __name__ == "__main__":
    main()
