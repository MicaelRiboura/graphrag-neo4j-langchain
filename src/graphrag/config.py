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
# Legado: tamanho pequeno para retrieve único: use GLOBAL_REPORTS_POOL_K no fluxo global map-reduce.
GLOBAL_REPORTS_TOP_K = int(os.environ.get("GRAPHRAG_GLOBAL_REPORTS_TOP_K", "5"))

# Global search (map-reduce sobre CommunityReport, estilo GraphRAG)
GLOBAL_REPORTS_POOL_K = int(
    os.environ.get("GRAPHRAG_GLOBAL_REPORTS_POOL_K", "40")
)  # quantos relatórios puxar do índice vetorial antes do map
GLOBAL_MAP_BATCH_SIZE = int(os.environ.get("GRAPHRAG_GLOBAL_MAP_BATCH_SIZE", "8"))
GLOBAL_MAP_MAX_POINTS_PER_BATCH = int(
    os.environ.get("GRAPHRAG_GLOBAL_MAP_MAX_POINTS", "12")
)
GLOBAL_REDUCE_TOP_POINTS = int(os.environ.get("GRAPHRAG_GLOBAL_REDUCE_TOP_POINTS", "35"))
GLOBAL_MAP_MAX_CONCURRENT = int(os.environ.get("GRAPHRAG_GLOBAL_MAP_MAX_CONCURRENT", "4"))

# Local search (GraphRAG-style: entities → fan-out to text, rels, neighbors, community reports)
LOCAL_ENTITY_K_PER_QUERY = int(os.environ.get("GRAPHRAG_LOCAL_ENTITY_K_PER_QUERY", "5"))
LOCAL_ENTITY_TOP_K = int(os.environ.get("GRAPHRAG_LOCAL_ENTITY_TOP_K", "12"))
# Orçamento de tokens para o contexto local (estilo max_data_tokens do GraphRAG)
LOCAL_MAX_DATA_TOKENS = int(os.environ.get("GRAPHRAG_LOCAL_MAX_DATA_TOKENS", "8000"))
# Pool de candidatos antes do ranking + pack por tokens (ampliar recuperação bruta)
LOCAL_TEXT_UNITS_FROM_GRAPH_POOL = int(os.environ.get("GRAPHRAG_LOCAL_TEXT_UNITS_GRAPH_POOL", "24"))
LOCAL_TEXT_UNITS_VECTOR_POOL = int(os.environ.get("GRAPHRAG_LOCAL_TEXT_UNITS_VECTOR_POOL", "12"))
LOCAL_RELATIONSHIP_LINES_POOL = int(os.environ.get("GRAPHRAG_LOCAL_REL_POOL", "80"))
LOCAL_NEIGHBOR_ENTITIES_POOL = int(os.environ.get("GRAPHRAG_LOCAL_NEIGHBOR_POOL", "32"))
LOCAL_ENTITY_LINKED_REPORTS_POOL = int(os.environ.get("GRAPHRAG_LOCAL_REPORTS_POOL", "6"))
LOCAL_CLAIMS_POOL = int(os.environ.get("GRAPHRAG_LOCAL_CLAIMS_POOL", "40"))
LOCAL_COVARIATES_POOL = int(os.environ.get("GRAPHRAG_LOCAL_COVARIATES_POOL", "50"))
# Legado / teto de documentos após pack (síntese e debug)
LOCAL_TEXT_UNITS_FROM_GRAPH_CAP = int(os.environ.get("GRAPHRAG_LOCAL_TEXT_UNITS_GRAPH_CAP", "10"))
LOCAL_TEXT_UNITS_VECTOR_CAP = int(os.environ.get("GRAPHRAG_LOCAL_TEXT_UNITS_VECTOR_CAP", "6"))
LOCAL_RELATIONSHIP_LINES_CAP = int(os.environ.get("GRAPHRAG_LOCAL_REL_LINES_CAP", "40"))
LOCAL_NEIGHBOR_ENTITIES_CAP = int(os.environ.get("GRAPHRAG_LOCAL_NEIGHBOR_CAP", "16"))
LOCAL_ENTITY_LINKED_REPORTS_CAP = int(os.environ.get("GRAPHRAG_LOCAL_REPORTS_CAP", "3"))
LOCAL_SYNTH_CONTEXT_DOC_CAP = int(os.environ.get("GRAPHRAG_LOCAL_SYNTH_DOC_CAP", "64"))
CYPHER_PROMPT_CONTEXT_DOCS = int(os.environ.get("GRAPHRAG_CYPHER_CONTEXT_DOCS", "12"))

# Comunidades hierárquicas (indexação estilo GraphRAG)
COMMUNITY_MAX_LEVELS = int(os.environ.get("GRAPHRAG_COMMUNITY_MAX_LEVELS", "6"))
COMMUNITY_MIN_SIZE = int(os.environ.get("GRAPHRAG_COMMUNITY_MIN_SIZE", "2"))

# Consolidação de descrição de entidade ao longo do corpus (gap GraphRAG: não só a primeira menção)
ENTITY_DESCRIPTION_MAX_CHARS = int(
    os.environ.get("GRAPHRAG_ENTITY_DESCRIPTION_MAX_CHARS", "12000")
)

# LLM
LLM_MODEL = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4.1-2025-04-14")
EMBEDDING_MODEL = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
