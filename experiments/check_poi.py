import duckdb
import pyarrow.parquet as pq

p = "ber/data/raw/foursquare_kaggle/coordinates.parquet"
con = duckdb.connect()
df = con.execute(f"SELECT * FROM '{p}' LIMIT 10").fetchdf()
print("Columns:", df.columns.tolist())

# Check total POIs and cluster sizes
poi_counts = con.execute(f"SELECT point_of_interest, COUNT(*) as cnt FROM '{p}' GROUP BY point_of_interest").fetchdf()
print(f"Total POIs: {len(poi_counts):,}")
print("Cluster size stats:")
print(poi_counts["cnt"].describe())

