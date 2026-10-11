"""
tests/test_geo_blocking.py — Unit tests for GeoBlocker G1
"""

import math
import pytest
from ber.retrieve.geo_blocking import GeoBlocker, GeoConfig, EARTH_RADIUS_METERS

class TestGeoBlocker:
    def _make_rec(self, rid: str, lat: float, lon: float) -> dict:
        return {
            "record_id": rid,
            "raw_fields": f'{{"name": "test", "latitude": {lat}, "longitude": {lon}}}'
        }

    def test_indexing_and_self_exclusion(self):
        r1 = self._make_rec("R1", 37.7749, -122.4194)
        r2 = self._make_rec("R2", 37.7750, -122.4195)  # ~14 meters away
        
        blocker = GeoBlocker(GeoConfig(k_neighbors=5, radius_meters=100.0))
        blocker.index_records([r1, r2])

        cands, pairs = blocker.retrieve([r1])
        assert len(cands) == 1
        assert cands[0]["cand_id"] == "R2"
        assert cands[0]["query_id"] == "R1"
        assert cands[0]["rank"] == 1
        assert cands[0]["distance_m"] < 50.0

    def test_radius_cutoff(self):
        r1 = self._make_rec("R1", 37.7749, -122.4194)
        r2 = self._make_rec("R2", 37.8000, -122.4194)  # ~2.78 km away

        blocker = GeoBlocker(GeoConfig(k_neighbors=5, radius_meters=1000.0))  # 1 km limit
        blocker.index_records([r1, r2])

        cands, pairs = blocker.retrieve([r1])
        assert len(cands) == 0  # beyond 1 km radius

    def test_k_cutoff(self):
        r1 = self._make_rec("R1", 0.0, 0.0)
        recs = [r1] + [self._make_rec(f"R{i}", 0.0001 * i, 0.0) for i in range(2, 10)]

        blocker = GeoBlocker(GeoConfig(k_neighbors=3, radius_meters=10000.0))
        blocker.index_records(recs)

        cands, pairs = blocker.retrieve([r1])
        assert len(cands) == 3
