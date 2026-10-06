"""Print GLEIF column names at key indices and a few sample GB rows."""
import zlib, struct, csv

with open(r'ber/data/raw/gleif/gleif_level1.csv', 'rb') as f:
    raw = f.read()
fn_len = struct.unpack('<H', raw[26:28])[0]
extra_len = struct.unpack('<H', raw[28:30])[0]
data_start = 30 + fn_len + extra_len
compressed = raw[data_start:]

d = zlib.decompressobj(-15)
decompressed = d.decompress(compressed[:800_000])
lines = decompressed.split(b'\n')
header = lines[0].decode('utf-8', errors='replace')
cols = [c.strip('"').strip() for c in header.split(',')]

# Print address-related cols (indices 34-60)
print("=== Address & ID columns (34-60) ===")
for i in range(34, min(61, len(cols))):
    print(f"  [{i}] {cols[i]}")

print("\n=== Entity status, category cols (195-210) ===")
for i in range(195, min(215, len(cols))):
    print(f"  [{i}] {cols[i]}")

# Print first 2 data rows, showing address/ID fields
print("\n=== Sample row fields ===")
row = next(csv.reader([lines[1].decode('utf-8', errors='replace')]))
KEY_INDICES = [0, 1, 3, 34, 39, 40, 41, 42, 43, 44, 46, 53, 55, 56, 187, 189, 190, 199]
for i in KEY_INDICES:
    if i < len(row):
        print(f"  [{i}] {cols[i]!r:60s} = {row[i]!r}")
