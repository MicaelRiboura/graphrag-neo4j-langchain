# Dataset US Oil & Gas (ONRR/Kaggle) para GraphRAG

Este guia prepara o ficheiro `OGORBcsv_cleaned.csv` para o pipeline existente do projeto:

1. CSV bruto -> limpeza e agregacao
2. Narrativas em `.txt` -> `TextUnit` no GraphRAG
3. Extração LLM -> entidades/relacoes no Neo4j

## Preparacao rapida (demo)

No diretório raiz do projeto:

```bash
python examples/og_prepare_graphrag_input.py --demo
```

Isso gera ficheiros em `docs/og_us_production/` com um subconjunto menor:
- estados: TX, NM, LA
- commodities: oil, gas
- anos: 2020-2024

## Preparacao completa

```bash
python examples/og_prepare_graphrag_input.py --csv-path OGORBcsv_cleaned.csv
```

Filtros uteis:

```bash
python examples/og_prepare_graphrag_input.py --states TX,LA --commodities Oil,Gas --min-year 2018 --max-year 2024
```

## Indexacao GraphRAG

Depois de gerar os `.txt`:

```bash
python scripts/run_indexing.py --input-dir docs/og_us_production
```

Opcionalmente, para prompt de extração focado em petróleo e gas:

```bash
set GRAPHRAG_EXTRACT_DOMAIN=oil_gas
python scripts/run_indexing.py --input-dir docs/og_us_production
```

Exemplo completo (indexacao + consulta):

```bash
python examples/og_graphrag_example.py
```

Somente consulta no grafo ja indexado:

```bash
python examples/og_graphrag_example.py --skip-indexing --question "Quais estados e commodities aparecem com mais destaque?"
```

## Camada estruturada opcional no Neo4j

Para carregar observacoes agregadas (queries quantitativas exatas):

```bash
python examples/og_prepare_graphrag_input.py --demo --neo4j --no-txt
```

O script cria/relaciona:
- `State`
- `Commodity`
- `Disposition`
- `TimePeriod`
- `Observation` (com `volume`, `row_count`, periodo, county/offshore)

## Notas de escala

- Evite indexar linha-a-linha do CSV bruto.
- Sempre agregue antes de gerar texto.
- Comece por `--demo` para validar custo/tempo de extração LLM.
