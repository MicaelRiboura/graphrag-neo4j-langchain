"""
Avalia somente a fase de consulta do GraphRAG (Oil & Gas) com base em
perguntas + gabaritos e exporta metricas para CSV.

Uso:
  python examples/og_graphrag_query_eval.py
  python examples/og_graphrag_query_eval.py --qa-csv examples/og_qa_eval_set.csv
  python examples/og_graphrag_query_eval.py --output-csv outputs/og_query_eval_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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

from langchain_openai import ChatOpenAI

from graphrag.config import OPENAI_API_KEY, LLM_MODEL
from graphrag.graph import get_compiled_graph


DEFAULT_OUTPUT_CSV = ROOT / "outputs" / "og_query_eval_results.csv"
DEFAULT_EXTRACT_DOMAIN = "oil_gas"


class MetricScores(BaseModel):
    """Pontuacoes por metrica em escala 0..1."""

    answer_correctness: float = Field(
        ge=0.0, le=1.0, description="Correcao da resposta em relacao ao gabarito."
    )
    context_comprehensiveness_recall: float = Field(
        ge=0.0, le=1.0, description="Cobertura de fatos essenciais no contexto recuperado."
    )
    faithfulness_groundedness: float = Field(
        ge=0.0, le=1.0, description="Fidelidade da resposta ao contexto fornecido."
    )
    reasoning_path_evaluation: float = Field(
        ge=0.0, le=1.0, description="Qualidade do caminho de raciocinio e decomposicao."
    )
    notes: str = Field(
        default="", description="Justificativa curta para as pontuacoes."
    )


@dataclass
class QAPair:
    """Entrada de avaliacao pergunta + gabarito."""

    question: str
    reference_answer: str
    expected_points: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia fase de consulta do GraphRAG Oil & Gas e salva metricas em CSV."
    )
    parser.add_argument(
        "--qa-csv",
        type=Path,
        default=None,
        help=(
            "CSV com colunas: question,reference_answer,expected_points(opcional). "
            "Se ausente, usa um conjunto demo embutido."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Arquivo CSV de saida com resultados e metricas.",
    )
    parser.add_argument(
        "--extract-domain",
        type=str,
        default=DEFAULT_EXTRACT_DOMAIN,
        help="Valor de GRAPHRAG_EXTRACT_DOMAIN (padrao: oil_gas).",
    )
    return parser.parse_args()


def load_qa_pairs(qa_csv: Path | None) -> list[QAPair]:
    if qa_csv is None:
        # Demo minimo para facilitar uso imediato.
        return [
            QAPair(
                question=(
                    "Which county in Texas recorded the highest volume of Gas (Mcf) "
                    "for the Sales-Royalty Due-MEASURED disposition in 2024, and what "
                    "was that exact volume?"
                ),
                reference_answer=(
                    "Tarrant County recorded the highest 2024 Gas (Mcf) volume for "
                    "Sales-Royalty Due-MEASURED, with 1,806,166.00 Mcf."
                ),
                expected_points=(
                    "County must be Tarrant; value must be 1,806,166.00 Mcf; "
                    "scope is Texas + 2024 + Sales-Royalty Due-MEASURED + Gas (Mcf)."
                ),
            )
        ]

    if not qa_csv.exists():
        raise FileNotFoundError(f"Arquivo QA nao encontrado: {qa_csv}")

    pairs: list[QAPair] = []
    with qa_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question", "reference_answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV de QA invalido. Colunas obrigatorias ausentes: {sorted(missing)}"
            )
        for row in reader:
            question = (row.get("question") or "").strip()
            reference_answer = (row.get("reference_answer") or "").strip()
            expected_points = (row.get("expected_points") or "").strip()
            if not question or not reference_answer:
                continue
            pairs.append(
                QAPair(
                    question=question,
                    reference_answer=reference_answer,
                    expected_points=expected_points,
                )
            )

    if not pairs:
        raise ValueError("Nenhum par pergunta/gabarito valido encontrado no CSV.")
    return pairs


def build_context_text(result: dict[str, Any]) -> str:
    context_docs = result.get("context_docs") or []
    community_reports = result.get("community_reports") or []
    cypher_result = result.get("cypher_result")

    parts: list[str] = []
    for idx, doc in enumerate(context_docs, start=1):
        if isinstance(doc, dict):
            content = doc.get("page_content", "")
            meta = doc.get("metadata", {})
        else:
            content = str(doc)
            meta = {}
        parts.append(f"[context_doc_{idx}] {content}\nmetadata={meta}")

    if community_reports:
        for idx, report in enumerate(community_reports, start=1):
            parts.append(f"[community_report_{idx}] {report}")

    if cypher_result is not None:
        parts.append(f"[cypher_result] {cypher_result}")

    return "\n\n".join(parts) if parts else "No context available."


def evaluate_metrics(
    *,
    llm_judge: ChatOpenAI,
    question: str,
    reference_answer: str,
    expected_points: str,
    generated_answer: str,
    query_result: dict[str, Any],
) -> MetricScores:
    # Caminho de raciocinio observado no grafo para apoiar a metrica 4.
    reasoning_path = {
        "search_type": query_result.get("search_type"),
        "subqueries": [getattr(sq, "sub_query", str(sq)) for sq in (query_result.get("subqueries") or [])],
    }
    context_text = build_context_text(query_result)

    judge_prompt = f"""
You are an evaluator for a GraphRAG QA system. Score each metric from 0.0 to 1.0.
Return ONLY valid JSON matching this schema:
{{
  "answer_correctness": float,
  "context_comprehensiveness_recall": float,
  "faithfulness_groundedness": float,
  "reasoning_path_evaluation": float,
  "notes": "short rationale"
}}

Scoring criteria:
1) answer_correctness:
   - Compare generated answer with reference answer and expected points.
   - 1.0 if semantically correct and complete.
2) context_comprehensiveness_recall:
   - Evaluate whether retrieved context includes the essential facts needed.
   - If crucial facts are missing from context, reduce score.
3) faithfulness_groundedness:
   - Evaluate if generated answer is fully supported by retrieved context/cypher result.
   - Penalize hallucinations or unsupported claims.
4) reasoning_path_evaluation:
   - Evaluate if route decision + subqueries form a coherent path for answering.
   - Penalize irrelevant or insufficient decomposition.

Input:
Question:
{question}

Reference answer:
{reference_answer}

Expected points (optional):
{expected_points or "(none)"}

Generated answer:
{generated_answer}

Observed reasoning path:
{json.dumps(reasoning_path, ensure_ascii=True)}

Retrieved context:
{context_text}
""".strip()

    structured = llm_judge.with_structured_output(MetricScores)
    return structured.invoke(judge_prompt)


def run_query(compiled_graph, question: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"eval-{abs(hash(question)) % 100000}"}}
    initial = {"question": question}
    result = compiled_graph.invoke(initial, config=config)
    return result


def main() -> None:
    args = parse_args()
    if args.extract_domain:
        os.environ["GRAPHRAG_EXTRACT_DOMAIN"] = args.extract_domain

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY nao configurada no ambiente/.env.")

    qa_pairs = load_qa_pairs(args.qa_csv)

    print("=" * 72)
    print("Avaliacao GraphRAG Oil & Gas - Fase de Consulta")
    print("=" * 72)
    print(f"Total de perguntas: {len(qa_pairs)}")
    print(f"Modelo LLM (resposta/juiz): {LLM_MODEL}")

    compiled = get_compiled_graph()
    llm_judge = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)

    rows: list[dict[str, Any]] = []
    for idx, qa in enumerate(qa_pairs, start=1):
        print(f"\n[{idx}/{len(qa_pairs)}] Pergunta: {qa.question}")
        result = run_query(compiled, qa.question)
        generated_answer = result.get("final_answer", "")

        metric_scores = evaluate_metrics(
            llm_judge=llm_judge,
            question=qa.question,
            reference_answer=qa.reference_answer,
            expected_points=qa.expected_points,
            generated_answer=generated_answer,
            query_result=result,
        )
        overall = (
            metric_scores.answer_correctness
            + metric_scores.context_comprehensiveness_recall
            + metric_scores.faithfulness_groundedness
            + metric_scores.reasoning_path_evaluation
        ) / 4.0

        row = {
            "question": qa.question,
            "reference_answer": qa.reference_answer,
            "expected_points": qa.expected_points,
            "generated_answer": generated_answer,
            "search_type": result.get("search_type", ""),
            "subqueries": " | ".join(
                [getattr(sq, "sub_query", str(sq)) for sq in (result.get("subqueries") or [])]
            ),
            "answer_correctness": metric_scores.answer_correctness,
            "context_comprehensiveness_recall": metric_scores.context_comprehensiveness_recall,
            "faithfulness_groundedness": metric_scores.faithfulness_groundedness,
            "reasoning_path_evaluation": metric_scores.reasoning_path_evaluation,
            "overall_score": overall,
            "notes": metric_scores.notes,
        }
        rows.append(row)

        print(
            "Scores => "
            f"AC={metric_scores.answer_correctness:.2f}, "
            f"CCR={metric_scores.context_comprehensiveness_recall:.2f}, "
            f"FG={metric_scores.faithfulness_groundedness:.2f}, "
            f"RPE={metric_scores.reasoning_path_evaluation:.2f}, "
            f"Overall={overall:.2f}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "question",
        "reference_answer",
        "expected_points",
        "generated_answer",
        "search_type",
        "subqueries",
        "answer_correctness",
        "context_comprehensiveness_recall",
        "faithfulness_groundedness",
        "reasoning_path_evaluation",
        "overall_score",
        "notes",
    ]

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "-" * 72)
    print(f"Resultados salvos em: {args.output_csv}")
    print("-" * 72)


if __name__ == "__main__":
    main()
