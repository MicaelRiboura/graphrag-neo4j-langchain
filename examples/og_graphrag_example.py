"""
Exemplo de indexacao + consulta para o dataset US Oil & Gas preparado.

Uso:
  python examples/og_graphrag_example.py
  python examples/og_graphrag_example.py --skip-indexing
  python examples/og_graphrag_example.py --question "Qual estado teve maior volume de gas?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Raiz do projeto e src no path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_env = ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env)
    except ImportError:
        pass


DEFAULT_INPUT_DIR = ROOT / "docs" / "og_us_production"
DEFAULT_QUESTION = (
    "Resuma os principais estados, commodities e tipos de disposition "
    "mencionados nos documentos de producao de oil and gas."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exemplo GraphRAG para o dataset US Oil & Gas preparado."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Diretorio com .txt preparados (padrao: docs/og_us_production)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Pergunta para executar apos a indexacao",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Pula indexacao e executa somente consulta",
    )
    parser.add_argument(
        "--extract-domain",
        type=str,
        default="oil_gas",
        help="Valor de GRAPHRAG_EXTRACT_DOMAIN (padrao: oil_gas)",
    )
    return parser.parse_args()


def run_indexing(input_dir: Path) -> None:
    from graphrag.indexing.load_and_chunk import run_load_and_chunk
    from graphrag.indexing.extract_graph import run_extract_on_chunks
    from graphrag.indexing.communities import run_communities
    from graphrag.indexing.reports import run_reports
    from graphrag.indexing.embed import run_embed_all

    if not input_dir.is_dir():
        print(f"Diretorio de entrada nao encontrado: {input_dir}")
        print("Execute antes: python examples/og_prepare_graphrag_input.py --demo")
        sys.exit(1)

    print("1. Carregando e fragmentando documentos...")
    chunk_records = run_load_and_chunk(input_dir)
    print(f"   Criados {len(chunk_records)} TextUnits.")

    # if chunk_records:
    print("2. Extraindo entidades e relacionamentos...")
    run_extract_on_chunks(chunk_records)
    print("   Extracao concluida.")

    print("3. Deteccao de comunidades...")
    comms = run_communities()
    print(f"   Encontradas {len(comms)} comunidades.")

    print("4. Gerando relatorios das comunidades...")
    run_reports()
    print("   Relatorios concluidos.")

    print("5. Calculando e armazenando embeddings...")
    run_embed_all()
    print("   Embeddings concluidos.")
    print("Indexacao finalizada.\n")


def run_question(question: str) -> str:
    from graphrag.graph import run_query

    return run_query(question)


def main() -> None:
    args = parse_args()

    if args.extract_domain:
        os.environ["GRAPHRAG_EXTRACT_DOMAIN"] = args.extract_domain

    print("=" * 64)
    print("Exemplo GraphRAG: dataset US Oil & Gas (indexacao + consulta)")
    print("=" * 64)
    print(f"Input dir: {args.input_dir}")
    print(f"Extract domain: {os.environ.get('GRAPHRAG_EXTRACT_DOMAIN', '')}")

    if not args.skip_indexing:
        print("\n--- Fase 1: Indexacao ---\n")
        run_indexing(args.input_dir)
    else:
        print("\n--- Fase 1: Indexacao pulada (--skip-indexing) ---\n")

    print("--- Fase 2: Consulta ---\n")
    print(f"Pergunta: {args.question}\n")
    answer = run_question(args.question)
    print("Resposta:")
    print("-" * 40)
    print(answer)
    print("-" * 40)


if __name__ == "__main__":
    main()
