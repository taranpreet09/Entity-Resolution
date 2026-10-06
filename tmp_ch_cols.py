import duckdb
con = duckdb.connect()
rows = con.execute(
    "SELECT * FROM read_csv_auto('ber/data/raw/companies_house/BasicCompanyDataAsOneFile-2026-10-01.csv', sample_size=500) LIMIT 1"
).description
for i, r in enumerate(rows):
    print(i, repr(r[0]))
