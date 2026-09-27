from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


COMMON_SUFFIX_REPLACEMENTS = {
    "pvt": "private",
    "private ltd": "private limited",
    "pvt ltd": "private limited",
    "private limited": "private limited",
    "llp": "llp",
    "ltd": "limited",
    "limited": "limited",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "llc": "llc",
    "co": "company",
    "company": "company",
    "plc": "plc",
}

ADDRESS_NORMALIZATION_REPLACEMENTS = {
    "st": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "ln": "lane",
    "lane": "lane",
    "dr": "drive",
    "drive": "drive",
    "ct": "court",
    "court": "court",
    "park": "park",
    "pl": "place",
    "place": "place",
    "way": "way",
    "suite": "suite",
    "ste": "suite",
    "apt": "apartment",
    "flat": "apartment",
    "no": "number",
    "num": "number",
    "#": "number",
}

DOMAIN_TLD_REPLACEMENTS = {
    "www": "",
    "com": "",
    "net": "",
    "org": "",
    "info": "",
    "io": "",
    "biz": "",
    "gov": "",
    "edu": "",
    "us": "",
    "uk": "",
    "in": "",
    "ca": "",
    "au": "",
}


def normalize_business_name(value: object) -> str:
    """Normalize noisy business names into a stable string for blocking.

    The objective is not perfect semantic normalization, but creating a consistent
    representation that preserves real-name equivalence while removing mostly
    cosmetic noise. This is intentionally separated from the full ML feature
    engineering pipeline.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = text.replace("+", " ")
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[\W_]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    for old, new in COMMON_SUFFIX_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    text = re.sub(r"\b(private limited|limited|incorporated|corporation|llc|company|llp|plc)\b", "", text)
    text = re.sub(rf"\b(?:{'|'.join(sorted(DOMAIN_TLD_REPLACEMENTS, key=len, reverse=True))})\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _business_name_variants(value: object) -> Set[str]:
    """Return canonical variants that capture compacted website-style names and spacing noise."""
    base = normalize_business_name(value)
    if not base:
        return set()

    variants: Set[str] = {base}
    compact = "".join(base.split())
    if compact:
        variants.add(compact)
    for token in base.split():
        if token in DOMAIN_TLD_REPLACEMENTS:
            variants.add(" ".join(part for part in base.split() if part not in DOMAIN_TLD_REPLACEMENTS))
    return {v for v in variants if v}


def normalize_address(value: object) -> str:
    """Normalize address strings to a stable, comparable key for blocking."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = text.replace("+", " ")
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[\W_]", " ", text, flags=re.UNICODE)

    tokens = text.split()
    normalized_tokens: List[str] = []
    prior_token = None
    for token in tokens:
        token = re.sub(r"^0+", "", token)
        token = ADDRESS_NORMALIZATION_REPLACEMENTS.get(token, token)
        if token in {"near", "beside", "opposite", "opp", "next", "to", "nr", "around"}:
            continue
        if token in {"india", "us", "usa", "united", "states"}:
            continue
        if token in {"unit", "apartment", "suite", "flat", "apt", "ste", "no", "num"} and prior_token == token:
            continue
        normalized_tokens.append(token)
        prior_token = token

    text = " ".join(normalized_tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"us", "usa", "united states", "united states of america"}:
        return "us"
    if text in {"in", "ind", "india"}:
        return "in"
    if text in {"fr", "fra", "france"}:
        return "fr"
    return text


def build_name_tokens(name: object) -> List[str]:
    normalized = normalize_business_name(name)
    return [token for token in normalized.split() if token]


def load_source_file(path: str | Path, chunksize: int | None = None) -> pd.DataFrame | Iterable[pd.DataFrame]:
    if chunksize is not None:
        return pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize)

    df = pd.read_csv(path, sep="\t", dtype=str)
    if "entity_id" not in df.columns:
        raise ValueError(f"Missing entity_id column in {path}")
    return df


def iter_source_chunks(path: str | Path, chunksize: int = 100_000) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
        if "entity_id" not in chunk.columns:
            raise ValueError(f"Missing entity_id column in {path}")
        yield chunk


@dataclass
class ExactNameBlocker:
    """Exact normalized-name blocking using a dictionary-backed inverted index."""

    def build_name_index(self, source_df: pd.DataFrame) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = defaultdict(list)
        for row in source_df.itertuples(index=False):
            entity_id = str(getattr(row, "entity_id", "")).strip()
            name = normalize_business_name(getattr(row, "business_name", ""))
            if not entity_id or not name:
                continue
            index[name].append(entity_id)
        return {k: sorted(set(v)) for k, v in index.items()}

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> List[Tuple[str, str, str, str]]:
        index = self.build_name_index(pd.concat([s2_df, s3_df], ignore_index=True, sort=False))

        rows: List[Tuple[str, str, str, str]] = []
        for row in s1_df.itertuples(index=False):
            s1_id = str(getattr(row, "entity_id", "")).strip()
            if not s1_id:
                continue

            norm_name = normalize_business_name(getattr(row, "business_name", ""))
            if not norm_name:
                continue

            matching_ids: Set[str] = set()
            for name_variant in _business_name_variants(getattr(row, "business_name", "")):
                matching_ids.update(index.get(name_variant, []))

            for cand_id in sorted(matching_ids):
                if cand_id.startswith("S2-"):
                    source = "source2"
                elif cand_id.startswith("S3-"):
                    source = "source3"
                else:
                    continue
                rows.append((s1_id, cand_id, source, "exact_name"))

        deduped: Set[Tuple[str, str, str, str]] = set()
        for row in rows:
            deduped.add(row)
        return sorted(deduped)


def _generate_candidates_for_chunk(
    s1_chunk: pd.DataFrame,
    name_index: Dict[str, List[str]],
    address_index: Dict[str, List[str]],
    entity_country: Dict[str, str],
    entity_source: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for row in s1_chunk.itertuples(index=False):
        s1_id = str(getattr(row, "entity_id", "")).strip()
        s1_country = normalize_country(getattr(row, "country", ""))
        if not s1_id:
            continue

        norm_name = normalize_business_name(getattr(row, "business_name", ""))
        norm_address = normalize_address(getattr(row, "business_address", ""))
        candidate_ids: Set[str] = set()

        if norm_name:
            for name_variant in _business_name_variants(getattr(row, "business_name", "")):
                candidate_ids.update(name_index.get(name_variant, []))
        if norm_address:
            candidate_ids.update(address_index.get(norm_address, []))

        for cand_id in sorted(candidate_ids):
            source = entity_source.get(cand_id)
            if source is None:
                continue

            cand_country = entity_country.get(cand_id, "")
            if s1_country and cand_country and s1_country != cand_country:
                continue

            pair = (s1_id, cand_id, source, "name_address")
            if pair not in seen:
                seen.add(pair)
                rows.append(pair)

    return rows


class EnhancedNameAddressBlocker:
    """A stronger blocking strategy that combines name and address evidence."""

    def build_name_index(self, source_df: pd.DataFrame) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = defaultdict(list)
        for row in source_df.itertuples(index=False):
            entity_id = str(getattr(row, "entity_id", "")).strip()
            raw_name = getattr(row, "business_name", "")
            for variant in _business_name_variants(raw_name):
                index[variant].append(entity_id)
        return {k: sorted(set(v)) for k, v in index.items()}

    def build_address_index(self, source_df: pd.DataFrame) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = defaultdict(list)
        for row in source_df.itertuples(index=False):
            entity_id = str(getattr(row, "entity_id", "")).strip()
            addr = normalize_address(getattr(row, "business_address", ""))
            if not entity_id or not addr:
                continue
            index[addr].append(entity_id)
        return {k: sorted(set(v)) for k, v in index.items()}

    def _build_combined_indexes(
        self,
        source_dfs: Iterable[pd.DataFrame],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
        name_index: Dict[str, List[str]] = defaultdict(list)
        address_index: Dict[str, List[str]] = defaultdict(list)
        entity_country: Dict[str, str] = {}
        entity_source: Dict[str, str] = {}

        for source_df in source_dfs:
            for row in source_df.itertuples(index=False):
                entity_id = str(getattr(row, "entity_id", "")).strip()
                if not entity_id:
                    continue
                country = normalize_country(getattr(row, "country", ""))
                entity_country[entity_id] = country
                if entity_id.startswith("S2-"):
                    entity_source[entity_id] = "source2"
                elif entity_id.startswith("S3-"):
                    entity_source[entity_id] = "source3"

                raw_name = getattr(row, "business_name", "")
                for variant in _business_name_variants(raw_name):
                    name_index[variant].append(entity_id)

                addr = normalize_address(getattr(row, "business_address", ""))
                if addr:
                    address_index[addr].append(entity_id)

        return (
            {k: sorted(set(v)) for k, v in name_index.items()},
            {k: sorted(set(v)) for k, v in address_index.items()},
            entity_country,
            entity_source,
        )

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> List[Tuple[str, str, str, str]]:
        name_index, address_index, entity_country, entity_source = self._build_combined_indexes([s2_df, s3_df])
        rows = _generate_candidates_for_chunk(s1_df, name_index, address_index, entity_country, entity_source)
        return sorted(set(rows))


def parse_ground_truth(path: str | Path) -> Dict[str, Set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt = gt[["source1_entity_id", "matched_entity_ids"]].dropna(subset=["source1_entity_id"])
    mapping: Dict[str, Set[str]] = {}
    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        raw = str(row.matched_entity_ids).strip() if not pd.isna(row.matched_entity_ids) else ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        mapping[s1_id] = matches
    return mapping
    gt = gt[["source1_entity_id", "matched_entity_ids"]].dropna(subset=["source1_entity_id"])
    mapping: Dict[str, Set[str]] = {}
    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        raw = str(row.matched_entity_ids).strip() if not pd.isna(row.matched_entity_ids) else ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        mapping[s1_id] = matches
    return mapping


def evaluate_exact_name_blocker(train_dir: str | Path) -> Dict[str, object]:
    """Evaluate the stronger name+address blocking on the training data using ground truth."""
    train_dir = Path(train_dir)
    s1_df = load_source_file(train_dir / "train_source1.tsv")
    s2_df = load_source_file(train_dir / "train_source2.tsv")
    s3_df = load_source_file(train_dir / "train_source3.tsv")
    gt = parse_ground_truth(train_dir / "train_ground_truth.tsv")

    start = time.perf_counter()
    blocker = EnhancedNameAddressBlocker()
    candidate_rows = blocker.generate_candidates(s1_df, s2_df, s3_df)
    elapsed = time.perf_counter() - start

    candidate_map: Dict[str, Set[str]] = defaultdict(set)
    for s1_id, cand_id, _, _ in candidate_rows:
        candidate_map[s1_id].add(cand_id)

    total_true_matches = 0
    total_recovered = 0
    missed_by_s1: Dict[str, List[str]] = {}

    for s1_id in sorted(gt):
        true_matches = gt[s1_id]
        total_true_matches += len(true_matches)
        recovered = true_matches & candidate_map.get(s1_id, set())
        total_recovered += len(recovered)
        missed = sorted(true_matches - candidate_map.get(s1_id, set()))
        if missed:
            missed_by_s1[s1_id] = missed

    counts = [len(candidate_map.get(s1_id, set())) for s1_id in s1_df["entity_id"].tolist()]
    total_s1 = len(s1_df)
    total_pairs = sum(len(v) for v in candidate_map.values())

    metrics = {
        "num_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "average_candidates_per_s1": total_pairs / total_s1 if total_s1 else 0.0,
        "median_candidates_per_s1": statistics.median(counts) if counts else 0.0,
        "max_candidates_per_s1": max(counts) if counts else 0.0,
        "candidate_recall": (total_recovered / total_true_matches) if total_true_matches else 1.0,
        "missed_true_matches": total_true_matches - total_recovered,
        "missed_by_s1": missed_by_s1,
        "runtime_seconds": elapsed,
        "candidate_count_distribution": counts,
        "candidate_rows": candidate_rows,
    }
    return metrics


def _print_metrics(metrics: Dict[str, object]) -> None:
    print("Exact Name Blocking Evaluation")
    print("=" * 60)
    print(f"Source 1 entities: {metrics['num_source1_entities']}")
    print(f"Total candidate pairs: {metrics['total_candidate_pairs']}")
    print(f"Average candidates / S1: {metrics['average_candidates_per_s1']:.3f}")
    print(f"Median candidates / S1: {metrics['median_candidates_per_s1']}")
    print(f"Max candidates / S1: {metrics['max_candidates_per_s1']}")
    print(f"Candidate recall: {metrics['candidate_recall']:.4f}")
    print(f"Missed true matches: {metrics['missed_true_matches']}")
    print(f"Runtime: {metrics['runtime_seconds']:.4f}s")

    missed = metrics["missed_by_s1"]
    if missed:
        print("\nMisses by S1 (showing up to 10):")
        for s1_id, ids in list(missed.items())[:10]:
            print(f"  {s1_id}: {', '.join(ids)}")
    else:
        print("\nNo missed true matches.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "student_resource" / "dataset" / "train"
    metrics = evaluate_exact_name_blocker(train_dir)
    _print_metrics(metrics)
