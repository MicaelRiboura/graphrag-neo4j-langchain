# Fluxo completo (GraphRAG + Neo4j + LangGraph/LangChain) — Óleo & Gás

Este documento descreve o **fluxo end‑to‑end** do projeto, separado em duas fases:

- **Fluxo de indexação**: como os dados entram, são particionados, extraídos, conectados (comunidades) e preparados para recuperação (vetores).
- **Fluxo de query**: como uma pergunta é roteada (local vs global), recupera contexto (grafo + vetores), executa Cypher e sintetiza a resposta.

O foco aqui é o **dataset de Óleo & Gás (ONRR/Kaggle)** suportado pelo projeto, incluindo **ontologias**, **tipos de relações**, **perguntas avaliadas** e **métricas de avaliação**.

---

## Visão geral da arquitetura

O sistema combina dois “planos” de conhecimento:

- **Plano textual (GraphRAG clássico)**: documentos narrativos em `.txt` → `Document`/`TextUnit` → extração de `Entity`, `RELATES_TO`, `Claim`, `Covariate` → comunidades e `CommunityReport`.
- **Plano estruturado (opcional, quantitativo e exato)**: para o dataset O&G, um subgrafo com `Observation` ligado a `State`, `Commodity`, `Disposition` e `TimePeriod` permite **consultas numéricas exatas** via Cypher (sem depender de sumarização textual).

Na consulta, o sistema usa **LangGraph** para orquestrar:

- **Local search**: busca híbrida (vetor + fan‑out no grafo) para perguntas específicas.
- **Global search**: busca por `CommunityReport` + map‑reduce (estilo Microsoft GraphRAG) para perguntas “temáticas”/macro.

---

## Fase 1 — Fluxo de indexação

### 1) Preparação do dataset (Óleo & Gás)

O projeto fornece um utilitário que prepara `OGORBcsv_cleaned.csv` (já limpo) para indexação GraphRAG:

- **Entrada**: `OGORBcsv_cleaned.csv`
- **Transformação**:
  - normalização de datas e volumes
  - filtragem (por estados/commodities/anos) e modo demo
  - **agregação** por chaves (estado, ano, commodity, disposição, county/offshore)
- **Saídas (duas opções, que podem coexistir)**:
  - **Narrativas `.txt`** em `docs/og_us_production/` (para GraphRAG textual)
  - **Subgrafo estruturado** no Neo4j (opcional) para queries exatas

Arquivos relevantes:

- `examples/og_prepare_graphrag_input.py`
- `docs/og_us_production/README.md`

#### 1.1) Narrativas `.txt` (TextUnits)

O script gera arquivos por **estado/ano**, por exemplo `TX_2024.txt`, contendo múltiplos blocos narrativos agregados. Cada bloco inclui:

- State, Year, Commodity
- Disposition (descrição e código)
- Total reported volume (com 2 casas)
- Aggregated rows (quantas linhas do CSV agregadas)
- Period covered (min/max date)
- County reference e Offshore region

Essas narrativas são a base do **plano textual** (GraphRAG).

#### 1.2) Subgrafo estruturado (opcional)

Quando executado com `--neo4j --no-txt`, o script escreve nós e relações para consultas quantitativas:

- Nós: `State`, `Commodity`, `Disposition`, `TimePeriod`, `Observation`
- `Observation` contém propriedades como `volume`, `row_count`, `min_date`, `max_date`, `county`, `offshore_region`

Relações criadas:

- `(o:Observation)-[:FOR_STATE]->(s:State)`
- `(o:Observation)-[:FOR_COMMODITY]->(c:Commodity)`
- `(o:Observation)-[:FOR_DISPOSITION]->(d:Disposition)`
- `(o:Observation)-[:IN_YEAR]->(t:TimePeriod)`

Isso permite responder perguntas com “**qual foi o volume exato** …” com alta precisão.

---

### 2) Ingestão e chunking (Document/TextUnit)

Depois de gerar os `.txt`, a indexação GraphRAG começa carregando e particionando texto em unidades menores.

Arquivo principal:

- `src/graphrag/indexing/load_and_chunk.py`

Passos:

- Carrega `.txt` do diretório de entrada (`--input-dir`), por padrão `**/*.txt`.
- Enriquecimento de metadados (GraphRAG‑style): `doc_title`, `file_path`, `file_ext`, `source_type`.
- Chunking com `RecursiveCharacterTextSplitter`:
  - estratégia por **tokens** (default) ou por **caracteres** (legado)
  - tamanhos/overlaps configuráveis via `graphrag.config`
- Persistência no Neo4j:
  - `(:Document {id, source, title, file_path, file_ext, source_type})`
  - `(:TextUnit {id, text, chunk_index, token_count, char_count, doc_title, source_file})`
  - relação `(:Document)-[:HAS_CHUNK]->(:TextUnit)`

---

### 3) Extração LLM (Entidades, Relações, Claims, Covariates)

Cada `TextUnit` passa por uma extração estruturada via LLM, retornando um objeto `ExtractedGraph` com:

- `Entity`: `{name, type, description}`
- `Relationship`: `{source, target, type, description}`
- `Claim`: `{subject, text, claim_type, status}`
- `Covariate`: `{entity_name, attribute, value, unit, covariate_kind}`

Arquivo principal:

- `src/graphrag/indexing/extract_graph.py`

#### 3.1) Prompt/ontologia de domínio (Óleo & Gás)

Quando `GRAPHRAG_EXTRACT_DOMAIN=oil_gas`, o sistema usa um prompt especializado que:

- **prioriza** tipos/entidades como: `State`, `County`, `OffshoreRegion`, `Commodity`, `DispositionType`, `TimePeriod`, `Measurement`
- sugere tipos de relacionamento (exemplos): `LOCATED_IN`, `HAS_DISPOSITION`, `HAS_COMMODITY`, `IN_PERIOD`, `MEASURED_VOLUME`

Importante:

- No armazenamento do Neo4j, essas relações extraídas são normalizadas em um *relacionamento único* `RELATES_TO` com um atributo `type` (ver abaixo). Ou seja, a “tipagem semântica” da relação fica em `r.type`.

#### 3.2) Resolução canônica de entidades

Antes de gravar entidades no Neo4j, o pipeline:

- normaliza tokens (case/whitespace)
- gera uma chave estável `entity_key = normalize(name) + '|' + normalize(type)`
- escolhe um nome canônico (heurística: string mais longa)
- mantém aliases limitados (`ENTITY_RESOLUTION_MAX_ALIASES`)

Arquivo:

- `src/graphrag/indexing/entity_resolution.py`

#### 3.3) Persistência no Neo4j (grafo textual)

O `persist_extraction_to_neo4j` cria/atualiza os seguintes nós e relações:

- Nós:
  - `(:Entity {entity_key, name, type, description, aliases, embedding?})`
  - `(:Claim {id, text, subject, source_tu, claim_type, status})`
  - `(:Covariate {id, name, value, unit, source_tu, entity_name, covariate_kind})`

- Ligações “texto → conhecimento”:
  - `(t:TextUnit)-[:MENTIONS]->(e:Entity)`
  - `(t:TextUnit)-[:EVIDENCE_FOR]->(c:Claim)`
  - `(t:TextUnit)-[:EVIDENCE_FOR]->(v:Covariate)`

- Ligações “entidade → fatos/atributos”:
  - `(e:Entity)-[:HAS_CLAIM]->(c:Claim)`
  - `(e:Entity)-[:HAS_COVARIATE]->(v:Covariate)`

- Relações entre entidades (com tipo semântico na propriedade):
  - `(a:Entity)-[:RELATES_TO {type, description}]->(b:Entity)`

---

### 4) Comunidades hierárquicas (GraphRAG Communities)

Após extrair um subgrafo de entidades e relações, o pipeline detecta **comunidades** (clusters) e opcionalmente cria uma hierarquia.

Arquivo principal:

- `src/graphrag/indexing/communities.py`

O que é persistido:

- `(:Community {id, level})`
- nível 0: `(e:Entity)-[:IN_COMMUNITY]->(c:Community)`
- níveis superiores: `(child:Community)-[:PART_OF]->(parent:Community)`

Algoritmo:

- Preferência por `igraph` (Leiden; fallback para multilevel).
- Fallback para `networkx` (Louvain) se `igraph` não estiver disponível.

---

### 5) Community Reports (resumos por comunidade)

O pipeline gera `CommunityReport` para:

- **nível 0**: sumariza “entidades + relações” do cluster
- **níveis superiores**: sumariza *bottom‑up* a partir de relatórios das comunidades filhas

Arquivo principal:

- `src/graphrag/indexing/reports.py`

Persistência:

- `(:CommunityReport {community_id, content, embedding?})`
- `(c:Community)-[:HAS_REPORT]->(r:CommunityReport)`

No domínio Óleo & Gás (`oil_gas`), o relatório é **estruturado e analítico**, enfatizando:

- overview do escopo (estado/commodity/anos)
- maiores contribuintes e volumes dominantes
- tendências temporais e comparações
- anomalias (p.ex., volumes negativos; ajustes; flaring)

---

### 6) Ancoragem de claims/covariates em comunidades

Para alinhar fatos/atributos ao modelo de comunidades, o pipeline adiciona:

- `(c:Claim)-[:ANCHORED_IN_COMMUNITY]->(comm:Community)`
- `(v:Covariate)-[:ANCHORED_IN_COMMUNITY]->(comm:Community)`

Arquivo:

- `src/graphrag/indexing/graph_links.py`

Regra:

- o claim/covariate ancora na comunidade de nível 0 da entidade sujeito (via `:IN_COMMUNITY`).

---

### 7) Embeddings + índices vetoriais (Neo4j Vector Index)

Para suportar busca semântica/híbrida, o pipeline calcula embeddings e cria índices vetoriais para:

- `TextUnit.text`
- `Entity.description` (ou `name` como fallback)
- `CommunityReport.content`

Arquivo:

- `src/graphrag/indexing/embed.py`

Criação de índice (Neo4j 5.x):

- `CREATE VECTOR INDEX ... FOR (n:Label) ON (n.embedding) ... similarity cosine`

---

## Fase 2 — Fluxo de query

A consulta é orquestrada por um **StateGraph** (LangGraph) que decide o caminho:

- **local**: perguntas específicas/factuais
- **global**: perguntas sobre tendências/temas gerais

Arquivo:

- `src/graphrag/graph/query_graph.py`

### 1) Router (local vs global)

O roteador é um classificador LLM que retorna `search_type ∈ {local, global}`.

Arquivo:

- `src/graphrag/chains/router.py`

Heurística semântica (no prompt):

- **local**: entidades/fatos concretos (“Qual foi o volume…”, “Qual condado…”, etc.)
- **global**: temas/tendências/sumarização (“Quais padrões…”, “Resumo geral…”, etc.)

### 2) Caminho local (GraphRAG “local search” + Cypher QA)

Ordem (LangGraph):

1. `decompose`
2. `local_retrieve`
3. `graph_qa`
4. `synthesize`

#### 2.1) Decompose (quebra em subqueries)

Gera 1 a 3 subconsultas, tipicamente:

- uma subquery para recuperação semântica
- outra para consulta estruturada no grafo

Arquivo:

- `src/graphrag/chains/decompose.py`

#### 2.2) Local retrieve (contexto híbrido)

Objetivo: montar um **contexto rico e balanceado** com orçamento de tokens, unindo:

- **Entidades âncora** por similaridade vetorial (`Entity.embedding`)
- Fan‑out no grafo:
  - linhas de relacionamentos `RELATES_TO`
  - entidades vizinhas (1 salto)
  - `CommunityReport` relacionados a entidades âncora (via `IN_COMMUNITY`)
  - `Claim` e `Covariate` ligados às entidades âncora
  - `TextUnit` que mencionam as âncoras (`TextUnit-[:MENTIONS]->Entity`)
- Complemento com **TextUnits** por busca vetorial (`TextUnit.embedding`)

Arquivo:

- `src/graphrag/retrieval/local_search.py`

Saídas principais:

- `context_docs`: lista de “documentos” (strings + metadados) usados na síntese e, opcionalmente, no prompt de Cypher
- `seed_entities`: nomes das entidades âncora selecionadas

#### 2.3) Graph QA (Cypher via LangChain)

Executa `GraphCypherQAChain` sobre o Neo4j:

- O LLM gera uma query Cypher com base no **schema introspectado** do Neo4j.
- Opcionalmente, injeta contexto do `local_retrieve` no prompt de Cypher para favorecer filtros/joins corretos.
- `return_direct=True` retorna o resultado “cru” do Cypher como `cypher_result`.

Arquivos:

- `src/graphrag/chains/graph_qa.py`
- `src/graphrag/prompts/cypher.py`

Nota prática no O&G:

- Se o subgrafo `Observation/State/Commodity/...` estiver carregado, o Cypher tende a responder perguntas quantitativas com precisão alta (p.ex., volumes exatos).

#### 2.4) Synthesize (resposta final)

Combina:

- `context_docs` (top‑N por cap) e
- `cypher_result` (quando existir)

…em um único “contexto” e pede ao LLM para responder **apenas com base nele** (sem inventar).

Arquivos:

- `src/graphrag/graph/nodes.py` (nó `synthesize_node`)
- `src/graphrag/prompts/synthesis.py`

---

### 3) Caminho global (Community Reports + map‑reduce)

Ordem (LangGraph):

1. `global_retrieve`
2. `global_synthesize`

#### 3.1) Global retrieve

Recupera um conjunto amplo de `CommunityReport` por similaridade vetorial (`CommunityReport.embedding`) e embaralha a lista para reduzir viés de ordem.

Arquivo:

- `src/graphrag/retrieval/global_search.py` (`fetch_global_community_reports`)

#### 3.2) Global synthesize (map‑reduce estilo GraphRAG)

- **Map**: por batch de relatórios, o LLM extrai “pontos factuais” com score de importância (0–100).
- **Reduce**: deduplica, ordena e usa o top‑N de pontos para gerar a resposta final.
- **Fallback**: se não houver pontos, faz uma síntese direta sobre uma janela dos relatórios.

Arquivos:

- `src/graphrag/retrieval/global_search.py` (`global_search_map_reduce`)
- `src/graphrag/prompts/global_search.py` (prompts map/reduce)

---

## Ontologias e tipos de relações

Nesta implementação, “ontologia” aparece em dois níveis:

1) **Ontologia do grafo base (GraphRAG)**: tipos de nós e relações persistidas pelo pipeline.
2) **Ontologia de domínio (Óleo & Gás)**: conceitos esperados na extração e, opcionalmente, no subgrafo estruturado.

### 1) Ontologia do grafo base (GraphRAG)

#### 1.1) Tipos de nós (labels)

- `Document`
- `TextUnit`
- `Entity`
- `Claim`
- `Covariate`
- `Community`
- `CommunityReport`

#### 1.2) Tipos de relações

- **Documentos e chunks**
  - `(:Document)-[:HAS_CHUNK]->(:TextUnit)`

- **Menções e evidências**
  - `(:TextUnit)-[:MENTIONS]->(:Entity)`
  - `(:TextUnit)-[:EVIDENCE_FOR]->(:Claim)`
  - `(:TextUnit)-[:EVIDENCE_FOR]->(:Covariate)`

- **Fatos/atributos ligados a entidades**
  - `(:Entity)-[:HAS_CLAIM]->(:Claim)`
  - `(:Entity)-[:HAS_COVARIATE]->(:Covariate)`

- **Relações semânticas entre entidades**
  - `(:Entity)-[:RELATES_TO {type, description}]->(:Entity)`
    - Observação: o “tipo semântico” (ex.: `LOCATED_IN`) é armazenado em `r.type`.

- **Comunidades e relatórios**
  - `(:Entity)-[:IN_COMMUNITY]->(:Community)` (nível 0)
  - `(:Community)-[:PART_OF]->(:Community)` (hierarquia)
  - `(:Community)-[:HAS_REPORT]->(:CommunityReport)`

- **Ancoragem em comunidades (claims/covariates)**
  - `(:Claim)-[:ANCHORED_IN_COMMUNITY]->(:Community)`
  - `(:Covariate)-[:ANCHORED_IN_COMMUNITY]->(:Community)`

### 2) Ontologia de domínio (Óleo & Gás)

#### 2.1) Conceitos priorizados na extração LLM

Quando `GRAPHRAG_EXTRACT_DOMAIN=oil_gas`, a extração dá preferência a entidades como:

- `State`
- `County`
- `OffshoreRegion`
- `Commodity`
- `DispositionType`
- `TimePeriod`
- `Measurement`

E incentiva relações como:

- `LOCATED_IN`
- `HAS_DISPOSITION`
- `HAS_COMMODITY`
- `IN_PERIOD`
- `MEASURED_VOLUME`

Na prática, essas entidades aparecem como `(:Entity {type: ...})` e as relações como `(:Entity)-[:RELATES_TO {type: "..."}]->(:Entity)`.

#### 2.2) Subgrafo estruturado (opcional)

Para consultas quantitativas exatas:

- Nós: `State`, `Commodity`, `Disposition`, `TimePeriod`, `Observation`
- Relações:
  - `FOR_STATE`, `FOR_COMMODITY`, `FOR_DISPOSITION`, `IN_YEAR`

---

## Perguntas avaliadas (Óleo & Gás)

O conjunto de avaliação está em:

- `examples/og_qa_eval_set.csv`

Ele contém 8 perguntas (com gabarito e pontos esperados) que cobrem:

- **Cobertura do recorte do dataset**:
  - estados presentes (NM, TX, LA) e faixas de anos por estado
- **Normalização de unidades**:
  - commodities e suas unidades (Gas em Mcf; Oil em bbl)
- **Sinais/anomalias de disposição**:
  - exemplos de disposições com volumes negativos (p.ex., Buy‑Back, Differences/Adjustments)
- **Consultas quantitativas específicas (exatas)**:
  - maior volume por county em TX para uma disposição e ano, e comparação com outro ano
  - comparação cross‑state (TX vs LA) para uma disposição em Oil em 2023
  - série temporal de um county/disposição (2020→2022)
  - volume exato em county/estado/disposição/ano (Eddy, NM, 2024)
  - identificação de county que teve um volume negativo específico (Denton, TX, 2022)

Em comum, as perguntas tendem a exigir:

- filtros por `state`, `year`, `commodity`, `disposition`, `county`
- recuperação de **números exatos** (volumes com precisão decimal)
- comparação temporal (variação entre anos)

Isso combina especialmente bem com o subgrafo estruturado de `Observation` quando disponível.

---

## Métricas de avaliação (Óleo & Gás)

O avaliador está em:

- `examples/og_graphrag_query_eval.py`

Ele avalia **somente a fase de consulta** (query) e exporta um CSV de resultados com:

- `question`, `reference_answer`, `expected_points`
- `generated_answer`
- `search_type` (local/global) e `subqueries` geradas
- métricas (0..1) e `overall_score` (média simples)

### Métricas (escala 0..1)

1. **answer_correctness**
   - Correção e completude da resposta em relação ao gabarito e aos “expected points”.

2. **context_comprehensiveness_recall**
   - Verifica se o **contexto recuperado** (text units, relatórios, cypher_result, etc.) contém os fatos essenciais necessários.
   - Penaliza quando o contexto não traz informações críticas para responder.

3. **faithfulness_groundedness**
   - Mede se a resposta está **totalmente suportada** pelo contexto recuperado / resultado do Cypher.
   - Penaliza alucinações e afirmações não fundamentadas.

4. **reasoning_path_evaluation**
   - Avalia a coerência do caminho de raciocínio observado:
     - decisão de roteamento (local/global)
     - qualidade das subqueries e decomposição
   - Penaliza decomposições irrelevantes ou insuficientes.

### Score agregado

- **overall_score** = média simples das 4 métricas:
  \[
  overall = \frac{AC + CCR + FG + RPE}{4}
  \]

---

## Referências de implementação (arquivos-chave)

- **Indexação**
  - `scripts/run_indexing.py` (entrada principal)
  - `src/graphrag/indexing/load_and_chunk.py`
  - `src/graphrag/indexing/extract_graph.py`
  - `src/graphrag/indexing/communities.py`
  - `src/graphrag/indexing/reports.py`
  - `src/graphrag/indexing/graph_links.py`
  - `src/graphrag/indexing/embed.py`

- **Query**
  - `src/graphrag/graph/query_graph.py`
  - `src/graphrag/graph/nodes.py`
  - `src/graphrag/retrieval/local_search.py`
  - `src/graphrag/retrieval/global_search.py`
  - `src/graphrag/chains/graph_qa.py`
  - `src/graphrag/prompts/cypher.py`
  - `src/graphrag/prompts/synthesis.py`

- **Óleo & Gás**
  - `examples/og_prepare_graphrag_input.py`
  - `docs/og_us_production/README.md`
  - `examples/og_qa_eval_set.csv`
  - `examples/og_graphrag_query_eval.py`

