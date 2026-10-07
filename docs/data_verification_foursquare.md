# Data Verification: Foursquare Location Matching (Kaggle)

**Dataset ID:** `foursquare`  
**Date Verified:** 2026-10-07  
**Verified By:** `scripts/verify_foursquare_stats.py`  
**Data Directory:** `ber/data/raw/foursquare_kaggle/preprocessing_data/`  
**Source Files:** `train0.csv`, `train1.csv`  

---

## 1. Overview & File Inventory

The dataset originates from the Kaggle **Foursquare Location Matching** competition. The preprocessed split files `train0.csv` and `train1.csv` contain single-table deduplication records of points of interest (POIs) worldwide.

| File Name | Row Count | File Size | Role / Usage |
|---|---|---|---|
| `train0.csv` | 569,406 | ~115.8 MB | Labeled POI records (Part 1) |
| `train1.csv` | 569,406 | ~115.7 MB | Labeled POI records (Part 2) |
| **Combined** | **1,138,812** | **~231.5 MB** | Full Labeled Record Store |

> [!IMPORTANT]
> **Evaluation Exclusion Guards:**
> - `test.csv` has no `point_of_interest` labels — **MUST NOT** be used for evaluation.
> - `pairs.csv` / `haversine_train*.csv` are pre-blocked candidate files — **MUST NOT** be used for retrieval evaluation; they represent a filtered candidate subset, not full ground truth.

---

## 2. Schema, Null Rates & Field Visibility

Measured from the combined 1,138,812-record dataset. A value is counted null if it is blank, "nan", "none", or "null" (case-insensitive).

| Column Name | Data Type | Null Count | Null Rate | Visibility / Role |
|---|---|---|---|---|
| `id` | string | 0 | 0.00 % | Unique Record ID (Primary Key) |
| `name` | string | 5 | 0.00 % | Model-visible POI Name |
| `latitude` | float | 0 | 0.00 % | Model-visible Geo Latitude |
| `longitude` | float | 0 | 0.00 % | Model-visible Geo Longitude |
| `address` | string | 396,648 | **34.83 %** | Model-visible Street Address |
| `city` | string | 299,206 | **26.27 %** | Model-visible City |
| `state` | string | 420,711 | **36.94 %** | Model-visible State / Province |
| `zip` | string | 596,237 | **52.36 %** | Model-visible Postal Code |
| `country` | string | 0 | 0.00 % | Model-visible ISO 2-letter Country Code |
| `url` | string | 871,088 | **76.49 %** | Model-visible Website URL |
| `phone` | string | 796,006 | **69.90 %** | Model-visible Telephone Number |
| `categories` | string | 98,307 | **8.63 %** | Model-visible Category Tags |
| **`point_of_interest`** | string | 0 | 0.00 % | **LABEL ONLY — Excluded from Model** |

> [!NOTE]
> Zip (52.36%), url (76.49%), and phone (69.90%) have high null rates. These are the sparsest model-visible fields — blocker keys K3 (phone) and K4 (url) will therefore have limited coverage.

---

## 3. Entity Cluster Distribution & Singleton Burden

Ground truth entity clusters are defined by matching `point_of_interest` values.

| Metric | Value |
|---|---|
| **Total Records ($N$)** | 1,138,812 |
| **Distinct POI Entities ($C$)** | 739,972 |
| **Singleton Clusters (Size = 1)** | 425,024 (57.44 % of clusters; 37.32 % of records) |
| **Non-Singleton Clusters (Size ≥ 2)** | 314,948 (42.56 % of clusters; 62.68 % of records) |
| **Total True Positive Pairs (unordered)** | 950,503 |
| **Cluster Size Min / Median / P95 / Max** | 1 / 1 / 3 / 332 |
| **Cluster Size Mean** | 1.54 |

> [!NOTE]
> Median cluster size is 1 (singleton). P95 cluster size is only 3, indicating the distribution is extremely right-skewed. The large max (332) comes from global chains (e.g. McDonald's, Starbucks) with many location duplicates.

---

## 4. Geographic Coverage & Lat/Lon Ranges

| Metric | Value |
|---|---|
| **Latitude Range** | [−84.92774, +85.05113] |
| **Longitude Range** | [−175.16025, +178.44039] |

### Top 10 Countries by Record Count

| Country | Records | Share |
|---|---|---|
| US (United States) | 245,284 | 21.54 % |
| TR (Turkey) | 115,177 | 10.11 % |
| ID (Indonesia) | 110,796 | 9.73 % |
| JP (Japan) | 70,032 | 6.15 % |
| TH (Thailand) | 58,808 | 5.16 % |
| RU (Russia) | 57,030 | 5.01 % |
| BR (Brazil) | 51,241 | 4.50 % |
| MY (Malaysia) | 46,326 | 4.07 % |
| BE (Belgium) | 25,821 | 2.27 % |
| GB (Great Britain) | 25,544 | 2.24 % |

---

## 5. Script & Non-Latin Character Analysis

All names in `train0.csv` and `train1.csv` have been lowercased and transliterated to ASCII/Latin by the competition's preprocessing pipeline. Non-Latin Unicode characters in `name` field: **0.00 %** across all records (all pre-transliterated to ASCII). Original script distribution is not recoverable from these files.

---

## 6. Phone & URL Null Rates (Model-Visible Keys)

| Field | Non-Null Records | Non-Null Rate |
|---|---|---|
| `phone` | 342,806 | 30.10 % |
| `url` | 267,724 | 23.51 % |

These low non-null rates directly limit the effective coverage of blocker keys K3 (phone) and K4 (url domain).

---

## 7. Licensing & Provenance

- **Data Provenance:** Foursquare Location Matching simulated/synthetic competition dataset. The competition description states that entity clusters are "simulated" matching ground truth. Recorded as **"synthetic"** provenance assumption in the BER decision log.
- **Licence:** **unverified** — competition rules text was not shipped with the raw data archive. Users should review Kaggle competition rules before use.
- **Competition Rules Text:** Not present in `ber/data/raw/foursquare_kaggle/`. Rules available at `https://www.kaggle.com/competitions/foursquare-location-matching/rules`.
