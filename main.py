"""Entry point: load env and run_query."""

import sys
from pathlib import Path

# Ensure project root is on path and load .env
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

# Optional: install src for development so "graphrag" is importable
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from graphrag.monitoring.token_cost import TRACKER


def main():
    TRACKER.reset("query")
    from graphrag.graph import run_query

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this knowledge base about?"
    print(run_query(question))
    TRACKER.print_summary()


if __name__ == "__main__":
    main()
