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

cfg = IDFConfig(k_candidates=50, df_cap=5000, country_filter=False)
retriever = IDFRetriever(cfg)
t0 = time.time()
retriever.build_index(records)
print(f"Index built in {time.time() - t0:.2f}s")

rng = random.Random(42)
sample_indices = sorted(rng.sample(range(len(records)), 20000))
sample_q = [records[i] for i in sample_indices]

t1 = time.time()
cand_map, undir, slots = retriever.retrieve_batch(sample_q, k=50)
dt = time.time() - t1
print(f"20,000 queries time at K=50: {dt:.2f}s ({dt/20000*1000:.2f} ms/query)")

