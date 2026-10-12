"""
ber.retrieve.idf — Channel G3: Name-Led IDF Lexical Retrieval
=============================================================
A config-driven, reusable lexical retrieval module using a hashed-vocabulary
sparse inverted index over name tokens, built over the full record table.

Key components:
  1. Hashed-vocabulary sparse inverted index: maps token hash/ID -> doc IDs.
  2. Per-corpus DF table: document frequency df_t for every vocabulary token.
  3. Pre-registered DF cap: tokens with df_t > df_cap are skipped at query time
     to eliminate generic noise words (e.g. 'venue', 'local') and accelerate retrieval.
  4. IDF Scorer:
       idf(t) = ln(1 + N / df_t)
       score(q, c) = sum_{t in q cap c} idf(t) [optionally length-normalized]
     Exposes compute_idf_overlap(name_a, name_b) for downstream reranking (G2).
  5. Top-K retrieval excluding self, with optional score floor and country filter.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ber.canon.normalise import canonicalise


def infer_country_from_coords(lat: float, lon: float) -> str:
    """Infer coarse country code from geographic coordinates when unlabelled."""
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon):
        return ""
    if 24.0 <= lat <= 50.0 and -125.0 <= lon <= -65.0:
        return "US"
    elif 50.0 <= lat <= 60.0 and -11.0 <= lon <= 2.0:
        return "GB"
    elif 49.0 <= lat <= 52.0 and 2.5 <= lon <= 6.5:
        return "BE"
    elif 36.0 <= lat <= 42.5 and 26.0 <= lon <= 45.0:
        return "TR"
    elif 54.0 <= lat <= 60.0 and 35.0 <= lon <= 40.0:
        return "RU"
    elif 30.0 <= lat <= 45.0 and 128.0 <= lon <= 146.0:
        return "JP"
    elif 5.0 <= lat <= 21.0 and 97.0 <= lon <= 106.0:
        return "TH"
    elif 1.0 <= lat <= 7.0 and 99.0 <= lon <= 120.0:
        return "MY"
    elif -11.0 <= lat <= 6.0 and 95.0 <= lon <= 141.0:
        return "ID"
    elif -34.0 <= lat <= -19.0 and -55.0 <= lon <= -40.0:
        return "BR"
    return ""


@dataclass
class IDFConfig:
    """Configuration for Channel G3 IDF retriever.

    Parameters
    ----------
    k_candidates:
        Maximum number of candidate records to retrieve per query.
    df_cap:
        Maximum document frequency. If int, absolute count (e.g. 5000).
        If float in (0, 1), fraction of corpus size N.
        Tokens with df > df_cap are skipped at query time. None disables cap.
    min_score:
        Minimum score threshold required for a candidate to be retained.
    length_normalized:
        If True, normalizes the overlap score by sqrt(|q_tokens| * |c_tokens|).
    country_filter:
        If True, restricts candidates to those matching query country code.
    """
    k_candidates: int = 20
    df_cap: int | float | None = 5000
    min_score: float = 0.0
    length_normalized: bool = False
    country_filter: bool = False


class IDFRetriever:
    """Hashed-vocabulary sparse inverted index and IDF scorer."""

    def __init__(self, config: IDFConfig | None = None) -> None:
        self.config = config or IDFConfig()
        self.total_docs: int = 0
        self.doc_ids: list[str] = []
        self.doc_countries: list[str] = []
        self.doc_token_counts: list[int] = []
        self.id_to_idx: dict[str, int] = {}

        # Vocabulary and DF
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.df_table: list[int] = []
        self.idf_table: list[float] = []

        # Sparse inverted index: token_id -> array/list of doc indices
        self.postings: list[list[int]] = []

    def _tokenize(self, text: str | None) -> list[str]:
        canon = canonicalise(text)
        if not canon:
            return []
        return canon.split()

    def build_index(self, records: Sequence[dict[str, Any]]) -> None:
        """Build the inverted index and DF table over full records sequence.

        Expected record keys: 'id' (or 'record_id'), 'name', optional 'country'
        or 'raw_fields' (JSON containing country/lat/lon).
        """
        self.total_docs = len(records)
        self.doc_ids = []
        self.doc_countries = []
        self.doc_token_counts = []
        self.id_to_idx = {}

        self.token_to_id = {}
        self.id_to_token = []
        self.df_table = []
        self.idf_table = []
        self.postings = []

        # 1. First pass: extract record attributes and document-level token sets
        doc_tokens_list: list[list[int]] = []
        raw_token_df = Counter()

        for idx, rec in enumerate(records):
            rid = str(rec.get("id") or rec.get("record_id") or f"doc_{idx}")
            self.doc_ids.append(rid)
            self.id_to_idx[rid] = idx

            # Extract country
            country = rec.get("country", "")
            if not country:
                rf = rec.get("raw_fields")
                if isinstance(rf, str):
                    try:
                        fields = json.loads(rf)
                    except Exception:
                        fields = {}
                elif isinstance(rf, dict):
                    fields = rf
                else:
                    fields = {}
                country = fields.get("country", "")
                if not country:
                    try:
                        lat = float(rec.get("latitude", fields.get("latitude", "nan")))
                        lon = float(rec.get("longitude", fields.get("longitude", "nan")))
                        country = infer_country_from_coords(lat, lon)
                    except (ValueError, TypeError):
                        country = ""
            self.doc_countries.append(country.strip().upper())

            # Name tokens
            name = rec.get("name")
            if not name:
                rf = rec.get("raw_fields")
                if isinstance(rf, dict):
                    name = rf.get("name")
                elif isinstance(rf, str):
                    try:
                        name = json.loads(rf).get("name")
                    except Exception:
                        name = ""
            toks = self._tokenize(name)
            self.doc_token_counts.append(len(toks))
            unique_toks = set(toks)
            raw_token_df.update(unique_toks)

        # 2. Build vocabulary table
        for tok, df_val in raw_token_df.items():
            t_id = len(self.id_to_token)
            self.token_to_id[tok] = t_id
            self.id_to_token.append(tok)
            self.df_table.append(df_val)
            # Standard smoothed IDF: ln(1 + N / df)
            idf_val = math.log(1.0 + self.total_docs / max(1, df_val))
            self.idf_table.append(idf_val)
            self.postings.append([])

        # 3. Build postings lists
        for doc_idx, rec in enumerate(records):
            name = rec.get("name")
            if not name:
                rf = rec.get("raw_fields")
                if isinstance(rf, dict):
                    name = rf.get("name")
                elif isinstance(rf, str):
                    try:
                        name = json.loads(rf).get("name")
                    except Exception:
                        name = ""
            unique_toks = set(self._tokenize(name))
            for tok in unique_toks:
                t_id = self.token_to_id[tok]
                self.postings[t_id].append(doc_idx)

    def get_effective_df_cap(self) -> int:
        """Resolve config.df_cap into an absolute document frequency cutoff."""
        if self.config.df_cap is None:
            return self.total_docs + 1
        if isinstance(self.config.df_cap, float) and 0.0 < self.config.df_cap <= 1.0:
            return int(self.config.df_cap * self.total_docs)
        return int(self.config.df_cap)

    def score_pair(self, name_a: str | None, name_b: str | None) -> float:
        """Compute IDF name overlap score between two entity names.

        Exposed for downstream reranking (G2) and feature extraction.
        """
        toks_a = self._tokenize(name_a)
        toks_b = self._tokenize(name_b)
        if not toks_a or not toks_b:
            return 0.0

        shared = set(toks_a) & set(toks_b)
        if not shared:
            return 0.0

        score = 0.0
        for tok in shared:
            t_id = self.token_to_id.get(tok)
            if t_id is not None:
                score += self.idf_table[t_id]
            else:
                # OOV token gets theoretical maximum IDF
                score += math.log(1.0 + max(1, self.total_docs))

        if self.config.length_normalized:
            denom = math.sqrt(len(toks_a) * len(toks_b))
            return score / max(denom, 1e-6)
        return score

    def retrieve_single(
        self,
        query_record: dict[str, Any],
        k: int | None = None,
        country_filter: bool | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-K candidates for a single query record.

        Returns list of (candidate_id, score) tuples.
        """
        k_val = k if k is not None else self.config.k_candidates
        c_filter = country_filter if country_filter is not None else self.config.country_filter
        min_sc = self.config.min_score
        df_cap = self.get_effective_df_cap()

        q_id = str(query_record.get("id") or query_record.get("record_id") or "")
        q_idx = self.id_to_idx.get(q_id, -1)

        # Country
        q_country = query_record.get("country", "")
        if not q_country and q_idx >= 0:
            q_country = self.doc_countries[q_idx]
        elif not q_country:
            try:
                lat = float(query_record.get("latitude", "nan"))
                lon = float(query_record.get("longitude", "nan"))
                q_country = infer_country_from_coords(lat, lon)
            except (ValueError, TypeError):
                q_country = ""
        q_country = q_country.strip().upper()

        q_name = query_record.get("name")
        q_toks = self._tokenize(q_name)
        if not q_toks:
            return []

        # Identify active query tokens below df_cap
        active_t_ids: list[int] = []
        for tok in set(q_toks):
            t_id = self.token_to_id.get(tok)
            if t_id is not None and self.df_table[t_id] <= df_cap:
                active_t_ids.append(t_id)

        if not active_t_ids:
            return []

        # Accumulate candidate scores
        cand_scores: dict[int, float] = defaultdict(float)
        for t_id in active_t_ids:
            idf_val = self.idf_table[t_id]
            for doc_idx in self.postings[t_id]:
                if doc_idx == q_idx:
                    continue  # exclude self
                cand_scores[doc_idx] += idf_val

        if not cand_scores:
            return []

        # Apply filtering & length normalization
        valid_cands: list[tuple[float, int]] = []
        q_len = len(q_toks)
        norm = self.config.length_normalized

        for doc_idx, raw_score in cand_scores.items():
            if c_filter and q_country:
                c_country = self.doc_countries[doc_idx]
                if c_country and c_country != q_country:
                    continue

            final_score = raw_score
            if norm:
                c_len = self.doc_token_counts[doc_idx]
                denom = math.sqrt(q_len * c_len)
                final_score = raw_score / max(denom, 1e-6)

            if final_score >= min_sc:
                valid_cands.append((final_score, doc_idx))

        if not valid_cands:
            return []

        # Top-K
        if len(valid_cands) > k_val:
            top_k_items = heapq.nlargest(k_val, valid_cands, key=lambda x: x[0])
        else:
            top_k_items = sorted(valid_cands, key=lambda x: x[0], reverse=True)

        return [(self.doc_ids[doc_idx], sc) for sc, doc_idx in top_k_items]

    def retrieve_batch(
        self,
        query_records: Sequence[dict[str, Any]],
        k: int | None = None,
        country_filter: bool | None = None,
    ) -> tuple[dict[str, list[str]], set[tuple[str, str]], int]:
        """Retrieve candidates for a batch of query records.

        Returns
        -------
        candidate_map:
            dict mapping query_id -> list of candidate_ids.
        retrieved_undirected_pairs:
            set of (min_id, max_id) unique pairs.
        total_candidates:
            total count of candidate slots generated.
        """
        k_val = k if k is not None else self.config.k_candidates
        c_filter = country_filter if country_filter is not None else self.config.country_filter

        cand_map: dict[str, list[str]] = {}
        retrieved_undir: set[tuple[str, str]] = set()
        total_slots = 0

        for q_rec in query_records:
            q_id = str(q_rec.get("id") or q_rec.get("record_id") or "")
            cands_with_scores = self.retrieve_single(q_rec, k=k_val, country_filter=c_filter)
            c_ids = [c_id for c_id, _ in cands_with_scores]
            cand_map[q_id] = c_ids
            total_slots += len(c_ids)
            for c_id in c_ids:
                retrieved_undir.add((min(q_id, c_id), max(q_id, c_id)))

        return cand_map, retrieved_undir, total_slots

