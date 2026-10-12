import duckdb
import pyarrow.parquet as pq

p = "ber/data/raw/foursquare_kaggle/coordinates.parquet"
con = duckdb.connect()
df = con.execute(f"SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude) FROM '{p}'").fetchall()
print("Lat/Lon bounding box:", df)

# Check some samples of lat/lon
samples = con.execute(f"SELECT latitude, longitude, name FROM '{p}' LIMIT 10").fetchall()
print("Samples:")
for s in samples:
    print(" ", s)

