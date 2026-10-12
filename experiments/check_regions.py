import duckdb

p = "ber/data/raw/foursquare_kaggle/coordinates.parquet"
con = duckdb.connect()
df = con.execute(f"SELECT latitude, longitude FROM '{p}'").fetchdf()

# Simple lat/lon clustering or bins
import numpy as np
lat_round = np.round(df["latitude"].values, 0)
lon_round = np.round(df["longitude"].values, 0)
from collections import Counter
regions = Counter(zip(lat_round, lon_round))
print("Top 15 geographic regions (1-deg cells):")
for (la, lo), cnt in regions.most_common(15):
    print(f"  lat={la:5.0f}, lon={lo:5.0f}: {cnt:6d} records")

