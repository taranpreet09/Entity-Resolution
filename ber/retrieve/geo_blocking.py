"""
ber/retrieve/geo_blocking.py — Geo-Proximity Retrieval Channel (G1)
===================================================================
Config-driven spatial proximity blocker using BallTree with Haversine distance metric.
Indexes all records by (latitude, longitude) coordinates.

Given a query record with coordinates (lat, lon), retrieves up to K nearest
records within distance radius R (meters), excluding self.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_METERS = 6_371_008.8

@dataclass
class GeoConfig:
    k_neighbors: int = 20
    radius_meters: float = 1000.0  # 1 km default

class GeoBlocker:
    """Config-driven spatial proximity blocker."""

    def __init__(self, config: GeoConfig | None = None) -> None:
        self.config = config or GeoConfig()
        self.record_ids: list[str] = []
        self.coords_rad: np.ndarray | None = None
        self.tree: BallTree | None = None
        self.id_to_idx: dict[str, int] = {}

    def index_records(self, records: list[dict[str, Any]]) -> None:
        """Index records with valid lat/lon coordinates."""
        rids = []
        coords = []
        for idx, rec in enumerate(records):
            rid = rec["record_id"]
            rf = rec.get("raw_fields", "{}")
            if isinstance(rf, str):
                try: fields = json.loads(rf)
                except: fields = {}
            else:
                fields = rf or {}

            try:
                lat = float(fields.get("latitude", "nan"))
                lon = float(fields.get("longitude", "nan"))
                if not (math.isnan(lat) or math.isnan(lon)):
                    rids.append(rid)
                    coords.append([math.radians(lat), math.radians(lon)])
            except (ValueError, TypeError):
                continue

        self.record_ids = rids
        self.id_to_idx = {rid: i for i, rid in enumerate(rids)}
        if coords:
            self.coords_rad = np.array(coords, dtype=np.float64)
            self.tree = BallTree(self.coords_rad, metric="haversine")
        else:
            self.coords_rad = None
            self.tree = None

    def retrieve(self, query_records: list[dict[str, Any]],
                 k: int | None = None,
                 r_meters: float | None = None) -> tuple[list[dict], set[tuple[str, str]]]:
        """Retrieve up to K nearest neighbours within R meters. Excludes self-pairs."""
        if self.tree is None or not self.record_ids:
            return [], set()

        k_val = k if k is not None else self.config.k_neighbors
        r_val = r_meters if r_meters is not None else self.config.radius_meters
        r_rad = r_val / EARTH_RADIUS_METERS

        q_rids = []
        q_coords = []

        for rec in query_records:
            rid = rec["record_id"]
            rf = rec.get("raw_fields", "{}")
            if isinstance(rf, str):
                try: fields = json.loads(rf)
                except: fields = {}
            else:
                fields = rf or {}

            try:
                lat = float(fields.get("latitude", "nan"))
                lon = float(fields.get("longitude", "nan"))
                if not (math.isnan(lat) or math.isnan(lon)):
                    q_rids.append(rid)
                    q_coords.append([math.radians(lat), math.radians(lon)])
            except (ValueError, TypeError):
                continue

        if not q_coords:
            return [], set()

        q_arr = np.array(q_coords, dtype=np.float64)
        
        # Query BallTree for radius R
        ind_list, dist_list = self.tree.query_radius(q_arr, r=r_rad, return_distance=True)

        candidates = []
        retrieved_pairs: set[tuple[str, str]] = set()

        for q_id, indices, dists in zip(q_rids, ind_list, dist_list):
            if len(indices) == 0:
                continue
            
            # Sort by distance
            sort_order = np.argsort(dists)
            sorted_indices = indices[sort_order]
            sorted_dists = dists[sort_order]

            rank = 1
            for cand_idx, dist_rad in zip(sorted_indices, sorted_dists):
                cand_id = self.record_ids[cand_idx]
                if cand_id == q_id:
                    continue  # exclude self
                
                dist_m = dist_rad * EARTH_RADIUS_METERS
                candidates.append({
                    "query_id": q_id,
                    "cand_id": cand_id,
                    "rank": rank,
                    "score": -dist_m,  # closer = higher score
                    "distance_m": dist_m,
                    "blocking_key": "G1",
                })
                retrieved_pairs.add((q_id, cand_id))
                rank += 1
                if rank > k_val:
                    break

        return candidates, retrieved_pairs
