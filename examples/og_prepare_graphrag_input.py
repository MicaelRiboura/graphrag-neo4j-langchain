"""Prepare US Oil & Gas CSV for GraphRAG text ingestion.

Reads OGORBcsv_cleaned.csv, normalizes volumes/dates, aggregates records and
generates narrative .txt files for the existing GraphRAG indexing pipeline.
Optionally writes an exact-queries subgraph to Neo4j.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RE_THOUSANDS = re.compile(r"[,\s]")
MISSING_VALUES = {"", "null", "none", "na", "n/a", "nan", "offshore", "-"}


@dataclass(frozen=True)
class GroupKey:
    state: str
    year: int
    commodity: str
    disposition_code: str
    disposition_description: str
    county: str
    offshore_region: str


@dataclass
class GroupAgg:
    total_volume: float = 0.0
    row_count: int = 0
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OGORB CSV as narrative docs for GraphRAG."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("OGORBcsv_cleaned.csv"),
        help="Path to OGORBcsv_cleaned.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/og_us_production"),
        help="Directory for generated .txt files",
    )
    parser.add_argument(
        "--states",
        type=str,
        default="",
        help="Comma-separated state filter (e.g. TX,NM,LA)",
    )
    parser.add_argument(
        "--commodities",
        type=str,
        default="",
        help="Comma-separated commodity filter",
    )
    parser.add_argument("--min-year", type=int, default=None, help="Min year filter")
    parser.add_argument("--max-year", type=int, default=None, help="Max year filter")
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Maximum number of aggregated groups to write",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Small subset for fast/cost-effective indexing",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Do not generate .txt files",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Also write structured aggregated nodes to Neo4j",
    )
    return parser.parse_args()


def norm_text(value: str, default: str = "Unknown") -> str:
    text = (value or "").strip()
    if text.lower() in MISSING_VALUES:
        return default
    return text


def parse_volume(raw_value: str) -> Optional[float]:
    raw = (raw_value or "").strip()
    if not raw:
        return None
    cleaned = RE_THOUSANDS.sub("", raw)
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_date(value: str) -> Optional[datetime]:
    raw = (value or "").strip()
    if not raw:
        return None
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m", "%Y/%m")
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                parsed = parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    return None


def build_filters(args: argparse.Namespace) -> Tuple[set[str], set[str]]:
    states = {
        s.strip().upper()
        for s in args.states.split(",")
        if s and s.strip()
    }
    commodities = {
        c.strip().lower()
        for c in args.commodities.split(",")
        if c and c.strip()
    }
    if args.demo:
        states.update({"TX", "NM", "LA"})
        if args.min_year is None:
            args.min_year = 2020
        if args.max_year is None:
            args.max_year = 2024
    return states, commodities


def should_keep_row(
    state: str,
    commodity: str,
    year: int,
    states_filter: set[str],
    commodities_filter: set[str],
    min_year: Optional[int],
    max_year: Optional[int],
) -> bool:
    if states_filter and state.upper() not in states_filter:
        return False
    if commodities_filter:
        commodity_norm = commodity.lower()
        if not any(token in commodity_norm for token in commodities_filter):
            return False
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def aggregate_csv(args: argparse.Namespace) -> Tuple[Dict[GroupKey, GroupAgg], dict]:
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    states_filter, commodities_filter = build_filters(args)
    groups: Dict[GroupKey, GroupAgg] = defaultdict(GroupAgg)

    stats = {
        "rows_read": 0,
        "rows_used": 0,
        "rows_invalid_date": 0,
        "rows_invalid_volume": 0,
        "rows_filtered": 0,
    }

    with args.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows_read"] += 1

            dt = parse_date(row.get("Production Date", ""))
            if not dt:
                stats["rows_invalid_date"] += 1
                continue

            volume = parse_volume(row.get("Volume", ""))
            if volume is None:
                stats["rows_invalid_volume"] += 1
                continue

            year = dt.year
            state = norm_text(row.get("State", ""), default="Offshore/Unknown")
            commodity = norm_text(row.get("Commodity", ""))
            if not should_keep_row(
                state=state,
                commodity=commodity,
                year=year,
                states_filter=states_filter,
                commodities_filter=commodities_filter,
                min_year=args.min_year,
                max_year=args.max_year,
            ):
                stats["rows_filtered"] += 1
                continue

            key = GroupKey(
                state=state,
                year=year,
                commodity=commodity,
                disposition_code=norm_text(row.get("Disposition Code", ""), "N/A"),
                disposition_description=norm_text(
                    row.get("Disposition Description", ""), "Unknown Disposition"
                ),
                county=norm_text(row.get("County", ""), "N/A"),
                offshore_region=norm_text(row.get("Offshore Region", ""), "N/A"),
            )
            agg = groups[key]
            agg.total_volume += volume
            agg.row_count += 1
            agg.min_date = dt if agg.min_date is None else min(agg.min_date, dt)
            agg.max_date = dt if agg.max_date is None else max(agg.max_date, dt)
            stats["rows_used"] += 1

    return groups, stats


def narrative_for_group(key: GroupKey, agg: GroupAgg) -> str:
    period_start = agg.min_date.strftime("%Y-%m-%d") if agg.min_date else f"{key.year}-01-01"
    period_end = agg.max_date.strftime("%Y-%m-%d") if agg.max_date else f"{key.year}-12-31"
    volume_str = f"{agg.total_volume:,.2f}"
    return (
        f"State: {key.state}. Year: {key.year}. Commodity: {key.commodity}. "
        f"Disposition: {key.disposition_description} (code: {key.disposition_code}).\n"
        f"Total reported volume: {volume_str}. Aggregated rows: {agg.row_count}. "
        f"Period covered: {period_start} to {period_end}. "
        f"County reference: {key.county}. Offshore region: {key.offshore_region}.\n"
        f"In {key.state} during {key.year}, {key.commodity} recorded {volume_str} volume units "
        f"for disposition '{key.disposition_description}'.\n"
    )


def write_txt_output(
    output_dir: Path,
    ordered_groups: List[Tuple[GroupKey, GroupAgg]],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_state_year: Dict[Tuple[str, int], List[Tuple[GroupKey, GroupAgg]]] = defaultdict(list)
    for key, agg in ordered_groups:
        by_state_year[(key.state, key.year)].append((key, agg))

    files_written = 0
    for (state, year), rows in sorted(by_state_year.items(), key=lambda x: (x[0][0], x[0][1])):
        safe_state = re.sub(r"[^A-Za-z0-9_-]+", "_", state).strip("_") or "unknown"
        target = output_dir / f"{safe_state}_{year}.txt"
        with target.open("w", encoding="utf-8") as f:
            f.write(f"US Oil & Gas Production and Disposition - {state} - {year}\n\n")
            for key, agg in sorted(rows, key=lambda item: item[1].total_volume, reverse=True):
                f.write(narrative_for_group(key, agg))
                f.write("\n")
        files_written += 1
    return files_written


def write_structured_to_neo4j(ordered_groups: List[Tuple[GroupKey, GroupAgg]]) -> int:
    from graphrag.store.neo4j_graph import get_neo4j_graph

    graph = get_neo4j_graph()
    driver = graph._driver
    query = """
    MERGE (s:State {name: $state})
    MERGE (c:Commodity {name: $commodity})
    MERGE (d:Disposition {code: $disp_code, description: $disp_desc})
    MERGE (t:TimePeriod {year: $year})
    CREATE (o:Observation {
      volume: $volume,
      row_count: $row_count,
      min_date: $min_date,
      max_date: $max_date,
      county: $county,
      offshore_region: $offshore_region
    })
    MERGE (o)-[:FOR_STATE]->(s)
    MERGE (o)-[:FOR_COMMODITY]->(c)
    MERGE (o)-[:FOR_DISPOSITION]->(d)
    MERGE (o)-[:IN_YEAR]->(t)
    """
    count = 0
    with driver.session() as session:
        for key, agg in ordered_groups:
            session.run(
                query,
                state=key.state,
                commodity=key.commodity,
                disp_code=key.disposition_code,
                disp_desc=key.disposition_description,
                year=key.year,
                volume=float(agg.total_volume),
                row_count=int(agg.row_count),
                min_date=agg.min_date.strftime("%Y-%m-%d") if agg.min_date else None,
                max_date=agg.max_date.strftime("%Y-%m-%d") if agg.max_date else None,
                county=key.county,
                offshore_region=key.offshore_region,
            )
            count += 1
    return count


def main() -> int:
    args = parse_args()
    groups, stats = aggregate_csv(args)

    ordered = sorted(
        groups.items(),
        key=lambda item: item[1].total_volume,
        reverse=True,
    )
    if args.max_groups is not None and args.max_groups >= 0:
        ordered = ordered[: args.max_groups]

    files_written = 0
    if not args.no_txt:
        files_written = write_txt_output(args.output_dir, ordered)

    observations_written = 0
    if args.neo4j:
        observations_written = write_structured_to_neo4j(ordered)

    print("=== OG Prepare GraphRAG Input ===")
    print(f"Rows read: {stats['rows_read']}")
    print(f"Rows used: {stats['rows_used']}")
    print(f"Rows filtered: {stats['rows_filtered']}")
    print(f"Rows invalid date: {stats['rows_invalid_date']}")
    print(f"Rows invalid volume: {stats['rows_invalid_volume']}")
    print(f"Groups produced: {len(groups)}")
    print(f"Groups exported: {len(ordered)}")
    if not args.no_txt:
        print(f"Text files written: {files_written} in {args.output_dir}")
    if args.neo4j:
        print(f"Structured observations written to Neo4j: {observations_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
