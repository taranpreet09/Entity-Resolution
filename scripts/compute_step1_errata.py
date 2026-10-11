"""
scripts/compute_step1_errata.py — Compute G=None train residual on 100k sample
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pyarrow.parquet as pq

from ber.canon.normalise import canonicalise
from experiments.run_foursquare_control import FoursquareBlocker, FSQ_ALL_KEYS, _fsq_name_cat, _fsq_loc_cat

PROCESSED_DIR = _REPO / "data" / "processed" / "foursquare"

def main():
    print("Loading Foursquare records and splits...")
    records_t = pq.read_table(PROCESSED_DIR / "records.parquet")
    record_ids = records_t.column("record_id").to_pylist()
    raw_fields_list = records_t.column("raw_fields").to_pylist()
    record_map = dict(zip(record_ids, raw_fields_list))

    primary_splits = json.loads((PROCESSED_DIR / "primary_splits.json").read_text())
    
    # Train queries
    train_rids = [r for r in record_ids if primary_splits.get(r) == "train"]
    rng = random.Random(42)
    sample_100k_rids = set(rng.sample(train_rids, 100_000))
    print(f"Sampled 100k train queries out of {len(train_rids):,}")

    # Labels parquet
    pairs_t = pq.read_table(PROCESSED_DIR / "pairs_labels.parquet")
    q_ids = pairs_t.column("query_id").to_pylist()
    c_ids = pairs_t.column("cand_id").to_pylist()
    labels = pairs_t.column("label").to_pylist()

    train_sample_pairs = [
        (q, c) for q, c, l in zip(q_ids, c_ids, labels)
        if l == 1 and q in sample_100k_rids
    ]
    print(f"True positive pairs for 100k train sample: {len(train_sample_pairs):,}")

    # Build blocker index over full corpus
    print("Building full corpus FoursquareBlocker index...")
    blocker = FoursquareBlocker(gate_size=None, keys=FSQ_ALL_KEYS)
    all_records = [{"record_id": r, "raw_fields": record_map[r]} for r in record_ids]
    blocker.index_records(all_records)
    print("Full corpus index built.")

    # Retrieve for 100k train sample (G=None)
    sample_queries = [{"record_id": r, "raw_fields": record_map[r]} for r in sample_100k_rids]
    
    print("Retrieving G=None for 100k train sample...")
    BATCH = 10_000
    retrieved_norm = set()
    for i in range(0, len(sample_queries), BATCH):
        batch = sample_queries[i:i + BATCH]
        _, _, _, retr = blocker.retrieve(batch)
        retrieved_norm.update((min(a,b), max(a,b)) for a, b in retr)
        if (i + len(batch)) % 20_000 == 0:
            print(f"  Processed {i + len(batch):,}/{len(sample_queries):,} queries...")

    true_norm = {(min(a,b), max(a,b)) for a, b in train_sample_pairs}
    unrecovered_norm = true_norm - retrieved_norm
    print(f"\nUnrecovered true pairs at G=None on 100k train sample: {len(unrecovered_norm):,}/{len(true_norm):,} ({100*len(unrecovered_norm)/len(true_norm):.2f}%)")

    # Classify unrecovered pairs
    name_cats = Counter()
    for a, b in unrecovered_norm:
        ra = record_map.get(a, "{}"); rb = record_map.get(b, "{}")
        try: ra_f = json.loads(ra) if isinstance(ra, str) else ra
        except: ra_f = {}
        try: rb_f = json.loads(rb) if isinstance(rb, str) else rb
        except: rb_f = {}

        name_a = ra_f.get("name",""); name_b = rb_f.get("name","")
        name_cat = _fsq_name_cat(name_a, name_b)
        name_cats[name_cat] += 1

    print("\nResidual taxonomy at G=None (100k train sample):")
    for cat, count in name_cats.most_common():
        print(f"  {cat:25s}: {count:6,} ({100*count/len(unrecovered_norm):.2f}%)")

if __name__ == "__main__":
    main()
