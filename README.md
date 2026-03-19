# GraphRAG with Neo4j, LangGraph and LangChain

Implementação da abordagem GraphRAG (Microsoft) em Python usando Neo4j, LangGraph e LangChain.

## Setup

1. Crie um ambiente virtual e instale as dependências:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -e .
   ```

2. Copie `.env.example` para `.env` e preencha:

   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `OPENAI_API_KEY`

## Uso

- **Exemplo completo (indexação + consulta):**  
  `python examples/simple_graphrag_example.py`  
  Indexa os documentos em `docs/` (incluindo `docs/sample.txt`) e responde à pergunta padrão. Para outra pergunta:  
  `python examples/simple_graphrag_example.py "Sua pergunta"`

- **Só consulta:** `python main.py "Sua pergunta"` ou importe `run_query` de `graphrag.graph`.

- **Só indexação:** `python scripts/run_indexing.py --input-dir ./docs`

## Dataset US Oil & Gas (Kaggle / ONRR)

1. Gere ficheiros narrativos a partir de `OGORBcsv_cleaned.csv`:

   `python examples/og_prepare_graphrag_input.py --demo`

2. (Opcional) `GRAPHRAG_EXTRACT_DOMAIN=oil_gas` para prompts de extração focados em petróleo/gás.

3. Indexação: `python scripts/run_indexing.py --input-dir docs/og_us_production`

   Exemplo indexação + consulta: `python examples/og_graphrag_example.py`

4. Grafo estruturado opcional (volumes no Neo4j):  
   `python examples/og_prepare_graphrag_input.py --demo --neo4j --no-txt`

Detalhes: [docs/og_us_production/README.md](docs/og_us_production/README.md).
