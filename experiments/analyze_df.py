import duckdb
from collections import Counter
from pathlib import Path
import math
import sys
sys.path.insert(0, ".")

from ber.canon.normalise import canonicalise

p = Path("ber/data/raw/foursquare_kaggle/coordinates.parquet")
con = duckdb.connect()
df = con.execute(f"SELECT id, name FROM '{p.as_posix()}'").fetchdf()

token_doc_counts = Counter()
total_docs = len(df)

for name in df["name"]:
    cn = canonicalise(name)
    if cn:
        toks = set(cn.split())
        token_doc_counts.update(toks)

print(f"Total documents: {total_docs:,}")
print(f"Total unique tokens: {len(token_doc_counts):,}")
print("\nTop 30 tokens by document frequency:")
for tok, cnt in token_doc_counts.most_common(30):
    pct = cnt / total_docs * 100.0
    idf = math.log(1.0 + total_docs / cnt)
    print(f"  {tok:20s}: {cnt:6d} ({pct:5.2f}%)  IDF={idf:5.2f}")

percentiles = [50, 90, 95, 99, 99.5, 99.9, 99.99]
import numpy as np
counts = np.array(list(token_doc_counts.values()))
print("\nToken DF percentiles:")
for p in percentiles:
    val = np.percentile(counts, p)
    print(f"  P{p:5.1f}: {val:6.1f} docs ({val/total_docs*100:5.3f}%)")
