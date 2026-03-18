"""Configuration: env vars and constants."""

import os
from pathlib import Path

# Load .env from project root if present
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Vector index names in Neo4j
INDEX_NAME_TEXT_UNITS = "text_units_embedding"
INDEX_NAME_ENTITIES = "entities_embedding"
INDEX_NAME_REPORTS = "community_reports_embedding"

# Embedding dimensions (OpenAI text-embedding-3-small)
EMBEDDING_DIMENSION = 1536

# Retrieval
RETRIEVAL_TOP_K = 5
GLOBAL_REPORTS_TOP_K = 5

# LLM
LLM_MODEL = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
