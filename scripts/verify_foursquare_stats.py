"""Verify null rates, country counts, lat/lon range, cluster sizes for the data_verification_foursquare.md."""
import sys, json
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DATA_DIR = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "preprocessing_data"
PROC_DIR = _REPO / "data" / "processed" / "foursquare"

# Read raw files
import csv

nulls = Counter()
country_counts = Counter()
lat_min, lat_max, lon_min, lon_max = float("inf"), float("-inf"), float("inf"), float("-inf")
total_rows = 0
non_latin = 0

for fname in ["train0.csv", "train1.csv"]:
    with open(DATA_DIR / fname, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            for col in ["name", "latitude", "longitude", "address", "city", "state",
                        "zip", "country", "url", "phone", "categories", "point_of_interest"]:
                v = row.get(col, "").strip()
                if not v or v.lower() in ("nan", "none", "null"):
                    nulls[col] += 1

            country_counts[row.get("country", "").strip()] += 1

            try:
                lat = float(row.get("latitude", "nan"))
                lon = float(row.get("longitude", "nan"))
                lat_min = min(lat_min, lat)
                lat_max = max(lat_max, lat)
                lon_min = min(lon_min, lon)
                lon_max = max(lon_max, lon)
            except ValueError:
                pass

print(f"Total rows: {total_rows:,}")
print("\nNull counts per column:")
cols_order = ["name","latitude","longitude","address","city","state","zip","country","url","phone","categories","point_of_interest"]
for col in cols_order:
    n = nulls[col]
    print(f"  {col:25s}: {n:7,}  ({100*n/total_rows:.2f}%)")

print(f"\nLat range: [{lat_min:.5f}, {lat_max:.5f}]")
print(f"Lon range: [{lon_min:.5f}, {lon_max:.5f}]")

print("\nTop 10 countries:")
for country, cnt in country_counts.most_common(10):
    print(f"  {country:5s}: {cnt:7,}  ({100*cnt/total_rows:.2f}%)")

# Cluster sizes from labels parquet
import pyarrow.parquet as pq
labels_t = pq.read_table(PROC_DIR / "labels.parquet", columns=["record_id", "entity_id"])
poi_groups = Counter(labels_t.column("entity_id").to_pylist())
sizes = list(poi_groups.values())
sizes.sort()

n_total = len(sizes)
n_singletons = sum(1 for s in sizes if s == 1)
n_non_singletons = sum(1 for s in sizes if s > 1)
max_size = max(sizes)
min_size = min(sizes)
median_size = sizes[n_total // 2]
import statistics
mean_size = statistics.mean(sizes)
p95_idx = int(n_total * 0.95)
p95_size = sizes[p95_idx]

print(f"\nCluster distribution:")
print(f"  Total clusters: {n_total:,}")
print(f"  Singletons: {n_singletons:,} ({100*n_singletons/n_total:.2f}%)")
print(f"  Non-singletons: {n_non_singletons:,} ({100*n_non_singletons/n_total:.2f}%)")
print(f"  Min/Median/P95/Max: {min_size} / {median_size} / {p95_size} / {max_size}")
print(f"  Mean: {mean_size:.2f}")
