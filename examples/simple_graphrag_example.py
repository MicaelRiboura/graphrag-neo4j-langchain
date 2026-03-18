"""
Exemplo simples: indexação + retorno de respostas com GraphRAG.

Requisitos:
- Neo4j rodando (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD no .env)
- OPENAI_API_KEY no .env
- Documentos em docs/ (por padrão usa docs/sample.txt incluído no repo)

Uso:
  python examples/simple_graphrag_example.py
  python examples/simple_graphrag_example.py "Sua pergunta aqui"
"""

import sys
from pathlib import Path

# Raiz do projeto e src no path
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

# Diretório de documentos (padrão: docs na raiz)
DOCS_DIR = root / "docs"
DEFAULT_QUESTION = "Como é o clima do planeta onde o mestre de Ahsoka Tano passou a infância?"


def run_indexing(input_dir: Path) -> None:
    """Executa o pipeline de indexação: load -> chunk -> extract -> communities -> reports -> embed."""
    from graphrag.indexing.load_and_chunk import run_load_and_chunk
    from graphrag.indexing.extract_graph import run_extract_on_chunks
    from graphrag.indexing.communities import run_communities
    from graphrag.indexing.reports import run_reports
    from graphrag.indexing.embed import run_embed_all

    if not input_dir.is_dir():
        print(f"Diretório de entrada não encontrado: {input_dir}")
        print("Crie a pasta docs/ e coloque arquivos .txt ou use o sample.txt incluído.")
        sys.exit(1)

    print("1. Carregando e fragmentando documentos...")
    chunk_records = run_load_and_chunk(input_dir)
    print(f"   Criados {len(chunk_records)} TextUnits.")

    if chunk_records:
        print("2. Extraindo entidades e relacionamentos...")
        run_extract_on_chunks(chunk_records)
        print("   Extração concluída.")

    print("3. Detecção de comunidades...")
    comms = run_communities()
    print(f"   Encontradas {len(comms)} comunidades.")

    print("4. Gerando relatórios das comunidades...")
    run_reports()
    print("   Relatórios concluídos.")

    print("5. Calculando e armazenando embeddings...")
    run_embed_all()
    print("Embeddings concluídos.")

    print("Indexação finalizada.\n")


def run_question(question: str) -> str:
    """Executa uma pergunta no grafo compilado e retorna a resposta."""
    from graphrag.graph import run_query
    return run_query(question)


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print("=" * 60)
    print("Exemplo GraphRAG: indexação + consulta")
    print("=" * 60)

    # print("\n--- Fase 1: Indexação ---\n")
    # run_indexing(DOCS_DIR)

    print("--- Fase 2: Consulta ---\n")
    print(f"Pergunta: {question}\n")
    answer = run_question(question)
    print("Resposta:")
    print("-" * 40)
    print(answer)
    print("-" * 40)


if __name__ == "__main__":
    main()
