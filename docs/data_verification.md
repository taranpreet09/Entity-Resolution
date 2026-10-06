# Data Verification Report

Generated: 2026-10-06 16:05 UTC

> All measurements are from the actual files in `data/raw/`.  No figures are estimated or invented.  Where measurement was not possible the field shows "unmeasured".

## 1. File Inventory

| File | Size |
|---|---|
| `gleif_level1.csv` | 222.58 MB |
| `BasicCompanyDataAsOneFile-2026-10-01.csv` | 2.62 GB |
| `Fodors-Zagats/tableA.csv` | 316,937 bytes |
| `Fodors-Zagats/tableB.csv` | 2.82 MB |
| `Fodors-Zagats/test_set.csv` | 2.20 MB |
| `Fodors-Zagats/train_set.csv` | 6.63 MB |
| `Fodors-Zagats/valid_set.csv` | 2.20 MB |

## 2. GLEIF Level-1 Golden Copy

**File status:** Truncated Zip64 archive — the end-of-central-directory record is absent.  The compressed payload decompresses successfully via raw-deflate (`zlib.decompressobj(-15)`).  Rows available in this file: **1,524,435** out of 3,451,660 announced in `latest_lei.json` (~44% of full golden copy).  All GLEIF counts are **lower bounds**.

- Total columns: 338
- Total rows in file: 1,524,435
- Records with a RegistrationAuthorityID: 1,524,435
- Records with a RegistrationAuthorityEntityID: 1,311,356
- Records with BOTH RA code AND entity ID: 1,311,356

### Null rates (first 20 columns sampled)

| Column | Null rate |
|---|---|
| `LEI` | 0.00% |
| `Entity.LegalName` | 0.00% |
| `Entity.LegalName.xmllang` | 1.36% |
| `Entity.OtherEntityNames.OtherEntityName.1` | 86.94% |
| `Entity.OtherEntityNames.OtherEntityName.1.xmllang` | 87.19% |
| `Entity.OtherEntityNames.OtherEntityName.1.type` | 86.94% |
| `Entity.OtherEntityNames.OtherEntityName.2` | 98.14% |
| `Entity.OtherEntityNames.OtherEntityName.2.xmllang` | 98.17% |
| `Entity.OtherEntityNames.OtherEntityName.2.type` | 98.14% |
| `Entity.OtherEntityNames.OtherEntityName.3` | 99.55% |
| `Entity.OtherEntityNames.OtherEntityName.3.xmllang` | 99.55% |
| `Entity.OtherEntityNames.OtherEntityName.3.type` | 99.55% |
| `Entity.OtherEntityNames.OtherEntityName.4` | 99.89% |
| `Entity.OtherEntityNames.OtherEntityName.4.xmllang` | 99.89% |
| `Entity.OtherEntityNames.OtherEntityName.4.type` | 99.89% |
| `Entity.OtherEntityNames.OtherEntityName.5` | 99.97% |
| `Entity.OtherEntityNames.OtherEntityName.5.xmllang` | 99.97% |
| `Entity.OtherEntityNames.OtherEntityName.5.type` | 99.97% |
| `Entity.TransliteratedOtherEntityNames.TransliteratedOtherEntityName.1` | 92.27% |
| `Entity.TransliteratedOtherEntityNames.TransliteratedOtherEntityName.1.xmllang` | 92.27% |

## 3. GLEIF — GB Entity Analysis

GB entities are identified by `Entity.RegistrationAuthority.RegistrationAuthorityID` ∈ {RA000585, RA000586, RA000587}.  These codes were confirmed from the actual data file (see ADR-003).

| RA Code | Authority | GB LEI Records (in file) |
|---|---|---|
| `RA000585` | Companies House — England & Wales | 72,294 |
| `RA000586` | Companies House — Northern Ireland | 925 |
| `RA000587` | Companies House — Scotland | 4,593 |
| **Total** | All GB CH authorities | **77,812** |

## 4. GLEIF × Companies House ID Matching

Normalisation applied to both sides before matching:
1. Strip leading/trailing whitespace
2. Uppercase
3. Zero-pad purely-numeric strings to 8 characters (e.g. `1234567` → `01234567`)

| Metric | Value |
|---|---|
| GB LEI records in file (RA000585/586/587) | 77,812 |
| Exactly matched a Companies House CompanyNumber | 65,388 |
| No match found in CH data | 12,424 |
| Match rate | 84.03% |

**Note on unmatched records:** Non-matches likely include:
- Companies dissolved before the CH extract date and removed from the basic data file
- Registered societies, LLPs and other entity types where the registration number format differs
- GLEIF records where the EntityID was mis-entered or uses an alternate format
- Scottish limited partnerships (SL prefix) which may not appear in the basic company data

## 5. Companies House Basic Data

- Total rows: 5,704,711
- Total columns: 55
- CompanyNumber column detected: `CompanyNumber`
- Distinct normalised CompanyNumbers loaded: 5,704,711

### Null rates (first 10 columns, full scan)

| Column | Null rate |
|---|---|
| `CompanyName` | 0.00% |
| `CompanyNumber` | 0.00% |
| `RegAddress.CareOf` | 99.57% |
| `RegAddress.POBox` | 99.44% |
| `RegAddress.AddressLine1` | 1.11% |
| `RegAddress.AddressLine2` | 40.56% |
| `RegAddress.PostTown` | 1.66% |
| `RegAddress.County` | 71.28% |
| `RegAddress.Country` | 15.72% |
| `RegAddress.PostCode` | 1.60% |

## 6. Fodors-Zagats Folder (Products Dataset)

**Note:** Despite the folder name, this dataset contains product matching records (columns: `id, title, modelno, price, shipweight, brand, dimensions`), not restaurant records.  The dataset ID used throughout the system is `fodors_zagats` to preserve the supplied name.

The train/valid/test split labels were generated from a blocking step.  They are **not** a random sample of all positive pairs.  Retrieval recall **cannot** be measured from this data; any recall figures computed on this dataset reflect post-blocking recall only.

### `tableA.csv`

- Rows: 2,554
- Columns: `id`, `title`, `modelno`, `price`, `shipweight`, `brand`, `dimensions`
- Null rates:
  - `id`: 0.00%
  - `title`: 0.00%
  - `modelno`: 0.00%
  - `price`: 0.00%
  - `shipweight`: 7.28%
  - `brand`: 0.00%
  - `dimensions`: 27.33%

### `tableB.csv`

- Rows: 22,074
- Columns: `id`, `title`, `modelno`, `price`, `shipweight`, `brand`, `dimensions`
- Null rates:
  - `id`: 0.00%
  - `title`: 0.00%
  - `modelno`: 0.00%
  - `price`: 12.96%
  - `shipweight`: 2.04%
  - `brand`: 0.00%
  - `dimensions`: 38.58%

### `test_set.csv`

- Rows: 8,684
- Columns: `id`, `left_id`, `right_id`, `label`, `left_title`, `left_modelno`, `left_price`, `left_shipweight`, `left_brand`, `left_dimensions`, `right_title`, `right_modelno`, `right_price`, `right_shipweight`, `right_brand`, `right_dimensions`
- Null rates:
  - `id`: 0.00%
  - `left_id`: 0.00%
  - `right_id`: 0.00%
  - `label`: 0.00%
  - `left_title`: 0.00%
  - `left_modelno`: 0.00%
  - `left_price`: 0.00%
  - `left_shipweight`: 7.40%
  - `left_brand`: 0.00%
  - `left_dimensions`: 26.95%
  - `right_title`: 0.00%
  - `right_modelno`: 0.00%
  - `right_price`: 11.27%
  - `right_shipweight`: 0.90%
  - `right_brand`: 0.00%
  - `right_dimensions`: 20.29%

### `train_set.csv`

- Rows: 26,050
- Columns: `id`, `left_id`, `right_id`, `label`, `left_title`, `left_modelno`, `left_price`, `left_shipweight`, `left_brand`, `left_dimensions`, `right_title`, `right_modelno`, `right_price`, `right_shipweight`, `right_brand`, `right_dimensions`
- Null rates:
  - `id`: 0.00%
  - `left_id`: 0.00%
  - `right_id`: 0.00%
  - `label`: 0.00%
  - `left_title`: 0.00%
  - `left_modelno`: 0.00%
  - `left_price`: 0.00%
  - `left_shipweight`: 7.34%
  - `left_brand`: 0.00%
  - `left_dimensions`: 27.29%
  - `right_title`: 0.00%
  - `right_modelno`: 0.00%
  - `right_price`: 11.50%
  - `right_shipweight`: 0.92%
  - `right_brand`: 0.00%
  - `right_dimensions`: 20.02%

### `valid_set.csv`

- Rows: 8,684
- Columns: `id`, `left_id`, `right_id`, `label`, `left_title`, `left_modelno`, `left_price`, `left_shipweight`, `left_brand`, `left_dimensions`, `right_title`, `right_modelno`, `right_price`, `right_shipweight`, `right_brand`, `right_dimensions`
- Null rates:
  - `id`: 0.00%
  - `left_id`: 0.00%
  - `right_id`: 0.00%
  - `label`: 0.00%
  - `left_title`: 0.00%
  - `left_modelno`: 0.00%
  - `left_price`: 0.00%
  - `left_shipweight`: 6.98%
  - `left_brand`: 0.00%
  - `left_dimensions`: 27.82%
  - `right_title`: 0.00%
  - `right_modelno`: 0.00%
  - `right_price`: 10.97%
  - `right_shipweight`: 0.83%
  - `right_brand`: 0.00%
  - `right_dimensions`: 19.81%

## 7. Licenses and Terms of Use

| Dataset | License / Terms |
|---|---|
| GLEIF Level-1 Golden Copy | [CC0 1.0 Public Domain Dedication](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy#/) (confirmed on GLEIF website) |
| Companies House Basic Data | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) (confirmed on Companies House website) |
| Fodors-Zagats / Products dataset | **Unverified** — no license file shipped with the data; origin and terms of use not confirmed from shipped files |
