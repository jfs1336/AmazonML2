from __future__ import annotations

import random
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from src.blocking import (
    EnhancedNameAddressBlocker,
    load_source_file,
    normalize_address,
    normalize_business_name,
    parse_ground_truth,
)

DEFAULT_FEATURE_COLUMNS = [
    "name_jaccard",
    "address_jaccard",
    "name_edit_similarity",
    "address_edit_similarity",
    "country_match",
    "name_token_overlap",
    "address_token_overlap",
]


def _tokenize(value: str) -> set[str]:
    text = normalize_business_name(value)
    return {tok for tok in text.split() if tok}


def _address_tokens(value: str) -> set[str]:
    text = normalize_address(value)
    return {tok for tok in text.split() if tok}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0.0


def _token_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list and not b_list:
        return 0.0
    if not a_list or not b_list:
        return 0.0
    return len(set(a_list) & set(b_list)) / max(len(set(a_list)), len(set(b_list)))


def pair_features(s1_row: Dict[str, Any], cand_row: Dict[str, Any]) -> Dict[str, float | int]:
    s1_name = str(s1_row.get("business_name", "") or "")
    cand_name = str(cand_row.get("business_name", "") or "")
    s1_address = str(s1_row.get("business_address", "") or "")
    cand_address = str(cand_row.get("business_address", "") or "")

    s1_name_norm = normalize_business_name(s1_name)
    cand_name_norm = normalize_business_name(cand_name)
    s1_addr_norm = normalize_address(s1_address)
    cand_addr_norm = normalize_address(cand_address)

    name_tokens_1 = set(tok for tok in s1_name_norm.split() if tok)
    name_tokens_2 = set(tok for tok in cand_name_norm.split() if tok)
    address_tokens_1 = set(tok for tok in s1_addr_norm.split() if tok)
    address_tokens_2 = set(tok for tok in cand_addr_norm.split() if tok)

    country_match = int(
        str(s1_row.get("country", "")).strip().lower() == str(cand_row.get("country", "")).strip().lower()
    )

    name_edit = SequenceMatcher(None, s1_name_norm, cand_name_norm).ratio()
    address_edit = SequenceMatcher(None, s1_addr_norm, cand_addr_norm).ratio()

    return {
        "name_jaccard": _jaccard(name_tokens_1, name_tokens_2),
        "address_jaccard": _jaccard(address_tokens_1, address_tokens_2),
        "name_edit_similarity": float(name_edit),
        "address_edit_similarity": float(address_edit),
        "country_match": country_match,
        "name_token_overlap": _token_overlap(name_tokens_1, name_tokens_2),
        "address_token_overlap": _token_overlap(address_tokens_1, address_tokens_2),
    }


def build_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        entity_id = str(row.get("entity_id", "")).strip()
        if entity_id:
            lookup[entity_id] = row
    return lookup


def load_candidate_pairs(candidate_path: str | Path) -> Dict[str, List[str]]:
    candidate_path = Path(candidate_path)
    candidate_map: Dict[str, List[str]] = defaultdict(list)
    if not candidate_path.exists():
        return dict(candidate_map)

    df = pd.read_csv(candidate_path, sep="\t", dtype=str)
    for row in df.to_dict(orient="records"):
        s1_id = str(row.get("source1_entity_id", "")).strip()
        cand_text = str(row.get("candidate_entity_ids", "") or "")
        if not s1_id:
            continue
        if not cand_text:
            candidate_map[s1_id] = []
            continue
        candidate_map[s1_id] = [c.strip() for c in cand_text.split(",") if c.strip()]
    return dict(candidate_map)


def generate_training_examples(
    train_dir: str | Path,
    source1_ids: set[str] | None = None,
) -> pd.DataFrame:
    train_dir = Path(train_dir)
    s1_df = load_source_file(train_dir / "train_source1.tsv")
    s2_df = load_source_file(train_dir / "train_source2.tsv")
    s3_df = load_source_file(train_dir / "train_source3.tsv")
    gt_map = parse_ground_truth(train_dir / "train_ground_truth.tsv")

    blocker = EnhancedNameAddressBlocker()
    candidate_rows = blocker.generate_candidates(s1_df, s2_df, s3_df)

    s2_lookup = build_entity_lookup(s2_df)
    s3_lookup = build_entity_lookup(s3_df)
    s1_lookup = build_entity_lookup(s1_df)
    all_candidate_ids = sorted(set(s2_lookup) | set(s3_lookup))

    candidate_by_s1: Dict[str, set[str]] = defaultdict(set)
    for s1_id, cand_id, _, _ in candidate_rows:
        if cand_id.startswith("S2-") or cand_id.startswith("S3-"):
            candidate_by_s1[s1_id].add(cand_id)

    rng = random.Random(42)
    rows: List[Dict[str, Any]] = []
    for s1_id in s1_df["entity_id"].astype(str).tolist():
        if source1_ids is not None and s1_id not in source1_ids:
            continue
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        positives = sorted(candidate_by_s1.get(s1_id, set()))
        for cand_id in positives:
            cand_row = s2_lookup.get(cand_id) or s3_lookup.get(cand_id)
            if cand_row is None:
                continue
            feat = pair_features(s1_row, cand_row)
            feat["source1_entity_id"] = s1_id
            feat["candidate_entity_id"] = cand_id
            feat["label"] = int(cand_id in gt_map.get(s1_id, set()))
            rows.append(feat)

        true_ids = gt_map.get(s1_id, set())
        positives_set = set(positives)
        sample_size = min(max(len(positives) * 3, 3), 10)
        chosen_negatives: set[str] = set()
        attempts = 0
        all_cand_len = len(all_candidate_ids)
        while len(chosen_negatives) < sample_size and attempts < sample_size * 5 and all_cand_len > 0:
            rand_cand = all_candidate_ids[rng.randint(0, all_cand_len - 1)]
            if rand_cand not in positives_set and rand_cand not in true_ids:
                chosen_negatives.add(rand_cand)
            attempts += 1

        for cand_id in chosen_negatives:
            cand_row = s2_lookup.get(cand_id) or s3_lookup.get(cand_id)
            if cand_row is None:
                continue
            feat = pair_features(s1_row, cand_row)
            feat["source1_entity_id"] = s1_id
            feat["candidate_entity_id"] = cand_id
            feat["label"] = 0
            rows.append(feat)

    return pd.DataFrame(rows)


def train_model(train_dir: str | Path) -> Tuple[Any, List[str]]:
    from sklearn.linear_model import LogisticRegression

    df = generate_training_examples(train_dir)
    if df.empty:
        raise ValueError("No training examples were generated from the candidate set.")

    feature_cols = [col for col in DEFAULT_FEATURE_COLUMNS if col in df.columns]
    X = df[feature_cols].fillna(0.0).astype(float)
    y = df["label"].astype(int)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    return model, feature_cols


def predict_matches_for_candidate_map(
    model: Any,
    feature_cols: List[str],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_map: Dict[str, List[str]],
    threshold: float = 0.5,
) -> Dict[str, List[str]]:
    s2_lookup = build_entity_lookup(s2_df)
    s3_lookup = build_entity_lookup(s3_df)
    s1_lookup = build_entity_lookup(s1_df)

    result: Dict[str, List[str]] = {}
    for s1_id, candidate_ids in candidate_map.items():
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        final_candidates: List[str] = []
        for cand_id in candidate_ids:
            if cand_id.startswith("S2-"):
                cand_row = s2_lookup.get(cand_id)
            elif cand_id.startswith("S3-"):
                cand_row = s3_lookup.get(cand_id)
            else:
                continue
            if cand_row is None:
                continue

            feat = pd.DataFrame([pair_features(s1_row, cand_row)], columns=DEFAULT_FEATURE_COLUMNS)
            prob = model.predict_proba(feat[feature_cols].fillna(0.0).astype(float))[0, 1]
            if prob >= threshold:
                final_candidates.append(cand_id)

        result[s1_id] = sorted(set(final_candidates))
    return result
