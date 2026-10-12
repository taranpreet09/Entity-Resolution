import time
import duckdb
import random
from pathlib import Path
import sys
sys.path.insert(0, ".")

from ber.retrieve.idf import IDFRetriever, IDFConfig

p = Path("ber/data/raw/foursquare_kaggle/coordinates.parquet")
con = duckdb.connect()
df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{p.as_posix()}'").fetchdf()

records = df.to_dict(orient="records")
print(f"Total records in full index: {len(records):,}")

# Build index
t0 = time.time()
cfg = IDFConfig(k_candidates=20, df_cap=5000, country_filter=False)
retriever = IDFRetriever(cfg)
retriever.build_index(records)
print(f"Index build time: {time.time() - t0:.2f}s")
print(f"Vocab size: {len(retriever.token_to_id):,}")

# Test 1,000 queries
rng = random.Random(42)
sample_indices = sorted(rng.sample(range(len(records)), 1000))
sample_q = [records[i] for i in sample_indices]

t1 = time.time()
cand_map, undir, slots = retriever.retrieve_batch(sample_q)
dt = time.time() - t1
print(f"1,000 queries time: {dt:.2f}s ({dt/1000*1000:.2f} ms/query)")
print(f"Estimated 20k queries: {dt*20:.1f}s")
print(f"Estimated 100k queries: {dt*100:.1f}s")

