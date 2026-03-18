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
