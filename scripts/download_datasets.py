"""
scripts/download_datasets.py — Download raw datasets for Companies House and GLEIF
"""
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "ber" / "data" / "raw"

CH_DIR = RAW_DIR / "companies_house"
GLEIF_DIR = RAW_DIR / "gleif"
FODORS_DIR = RAW_DIR / "Fodors-Zagats"

CH_URL = "http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-10-01.zip"
CH_ZIP = CH_DIR / "BasicCompanyDataAsOneFile-2026-10-01.zip"
CH_CSV = CH_DIR / "BasicCompanyDataAsOneFile-2026-10-01.csv"

GLEIF_URL = "https://goldencopy.gleif.org/storage/golden-copy-files/2026/10/12/1287425/20261012-0000-gleif-goldencopy-lei2-golden-copy.csv.zip"
GLEIF_TARGET = GLEIF_DIR / "gleif_level1.csv"


def download_with_progress(url: str, dest_path: Path, desc: str):
    print(f"\n[Download] {desc}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        last_print = t0

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 5.0 or downloaded == total_size:
                elapsed = now - t0
                speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                pct = (downloaded / total_size * 100.0) if total_size else 0.0
                print(f"  Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.2f} MB/s", flush=True)
                last_print = now

    if tmp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        tmp_path.rename(dest_path)
    print(f"  Finished {desc} in {time.time() - t0:.1f}s ({dest_path.stat().st_size / (1024*1024):.1f} MB).")


def setup_companies_house():
    if CH_CSV.exists() and CH_CSV.stat().st_size > 100_000_000:
        print(f"Companies House CSV already exists at {CH_CSV} ({CH_CSV.stat().st_size / (1024*1024):.1f} MB). Skipping.")
        return

    CH_DIR.mkdir(parents=True, exist_ok=True)
    download_with_progress(CH_URL, CH_ZIP, "Companies House BasicCompanyDataAsOneFile (ZIP)")

    print(f"Extracting {CH_ZIP} to {CH_DIR}...")
    t0 = time.time()
    with zipfile.ZipFile(CH_ZIP, "r") as zf:
        zf.extractall(CH_DIR)
    print(f"Extracted in {time.time() - t0:.1f}s.")

    # Remove zip to save disk space
    if CH_ZIP.exists():
        CH_ZIP.unlink()
        print(f"Cleaned up {CH_ZIP}.")

    if CH_CSV.exists():
        print(f"Companies House CSV ready: {CH_CSV} ({CH_CSV.stat().st_size / (1024*1024):.1f} MB).")
    else:
        # Check what was extracted
        csvs = list(CH_DIR.glob("*.csv"))
        print(f"Extracted CSVs: {csvs}")
        if csvs:
            if csvs[0] != CH_CSV:
                csvs[0].rename(CH_CSV)
                print(f"Renamed {csvs[0]} -> {CH_CSV}")


def setup_gleif():
    if GLEIF_TARGET.exists() and GLEIF_TARGET.stat().st_size > 100_000_000:
        print(f"GLEIF file already exists at {GLEIF_TARGET} ({GLEIF_TARGET.stat().st_size / (1024*1024):.1f} MB). Skipping.")
        return

    GLEIF_DIR.mkdir(parents=True, exist_ok=True)
    # The adapter expects the zip file directly at gleif_level1.csv
    download_with_progress(GLEIF_URL, GLEIF_TARGET, "GLEIF Golden Copy Level 1 (Zip payload)")
    print(f"GLEIF Level 1 ready: {GLEIF_TARGET} ({GLEIF_TARGET.stat().st_size / (1024*1024):.1f} MB).")


def check_fodors_zagats():
    print(f"\nChecking Fodors-Zagats harness set at {FODORS_DIR}...")
    if FODORS_DIR.exists():
        files = list(FODORS_DIR.glob("*.csv"))
        print(f"Found {len(files)} files in Fodors-Zagats:")
        for f in files:
            print(f"  {f.name}: {f.stat().st_size:,} bytes")
    else:
        print("Fodors-Zagats folder missing!")


def main():
    print("=== Starting Dataset Downloads ===")
    check_fodors_zagats()
    setup_companies_house()
    setup_gleif()
    print("\n=== Dataset Downloads Complete ===")


if __name__ == "__main__":
    main()

