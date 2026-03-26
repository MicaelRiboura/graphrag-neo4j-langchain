"""CLI to run a GraphRAG query (optional entry point)."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

env_file = root / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass

from graphrag.graph import run_query
from graphrag.monitoring.token_cost import TRACKER


if __name__ == "__main__":
    TRACKER.reset("query")
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main themes?"
    print(run_query(q))
    TRACKER.print_summary()
