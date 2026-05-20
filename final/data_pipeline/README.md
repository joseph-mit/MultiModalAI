# Data Pipeline

This folder contains the complete data collection, cleaning, maintenance, and embedding pipeline used to construct the multimodal real estate dataset for this project. Running these scripts in order will reproduce the exact dataset used in our experiments.

---

## Files

| File | Description |
|------|-------------|
| `data_collection.py` | Scrapes Zillow listings and fetches Google Street View, satellite, and Census data |
| `data_maintenance.py` | Fills missing census data, re-downloads missing images, downloads listing photos, fetches listing descriptions |
| `data_cleaning.py` | Cleans the master CSV, remaps image paths, computes price gap, outputs `boston_cleaned.csv` |
| `clip_script.py` | Generates CLIP ViT-B/32 embeddings for all four visual/text modalities and saves to `clip_embeddings.npz` |

---

## The Six Modalities

### Strategic Modalities (agent-curated)
These are produced by the seller's agent and are designed to present the property favorably.

**1. Listing Text**
- What: The property description written by the listing agent
- Source: Zillow property detail API (via Hasdata)
- Format: Free text string, stored in `listing_description` column of the master CSV
- Coverage: ~99.8% of properties

**2. Listing Photos (Interior)**
- What: 15–37 professionally staged listing photos selected by the agent
- Source: Zillow CDN (photo URLs returned by the listing API)
- Format: JPEG files saved as `listing_photos/{property_id}/{property_id}_interior_{n}.jpg`
- Coverage: ~92% of properties

---

### Objective Modalities (independent sources)
These exist independently of the transaction and are not curated by the seller.

**3. Google Street View**
- What: Four street-level images at 0°, 90°, 180°, 270° headings centered on the property's coordinates
- Source: Google Street View Static API
- Format: JPEG files saved as `street_view/{property_id}_h{heading}.jpg`
- Coverage: 100% of properties

**4. Satellite Imagery**
- What: One top-down satellite tile at zoom level 18 centered on the property's coordinates
- Source: Google Maps Static API (satellite map type)
- Format: JPEG files saved as `satellite/{property_id}_sat.jpg`
- Coverage: 100% of properties

**5. Tabular Attributes**
- What: Structured property attributes including beds, baths, square footage, home type, sale price, listing price, days on market
- Source: Zillow listing API (via Hasdata)
- Format: Columns in the master CSV (`boston_listings_with_census.csv`)
- Coverage: 100% of properties

**6. Census Demographics**
- What: Tract-level demographic data including median household income, median home value, median rent, educational attainment, total population, and racial/ethnic composition
- Source: U.S. Census Bureau ACS 5-Year Estimates (2023), retrieved via the Census geocoding and data APIs
- Format: Columns in the master CSV prefixed with `census_`
- Coverage: ~99% of properties (a small number of tracts have suppressed data)

---

## Dataset Overview

| Statistic | Value |
|-----------|-------|
| Total properties | 23,624 |
| Cities/towns | 15 (Greater Boston metropolitan area) |
| Price range | ~$50K – $28M |
| Satellite images | 25,439 |
| Street view images | 101,571 |
| Listing photo folders | 22,965 |
| Listing photos total | 572,413 |
| Properties with all 4 CLIP modalities | 21,769 (92.1%) |

**Cities included:** Boston, Brookline, Medford, Newton, Quincy, Lowell, Arlington, Brockton, Cambridge, Waltham, Framingham, Lynn, Malden, Braintree, Somerville

---

## Setup

### Prerequisites

```bash
pip install requests pandas python-dotenv
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install numpy pillow pyarrow
```

### API Keys

Create a `.env` file in the same directory as the scripts:

```
HASDATA_KEY=your_hasdata_key_here
GOOGLE_API_KEY=your_google_key_here
CENSUS_KEY=your_census_key_here
```

- **Hasdata API key**: Sign up at [hasdata.com](https://hasdata.com) — used to scrape Zillow listings and property details
- **Google API key**: Create at [console.cloud.google.com](https://console.cloud.google.com) — enable **Street View Static API** and **Maps Static API**
- **Census API key**: Free, register at [api.census.gov](https://api.census.gov/data/key_signup.html)

### Directory Structure

All data should live in a single base directory. Set `BASE` at the top of each script to point to your local data folder:

```
mmai_midterm_report/
├── boston_listings_with_census.csv   ← master raw CSV
├── boston_cleaned.csv                ← cleaned CSV for modeling
├── clip_embeddings.npz               ← CLIP embeddings
├── listing_photos/
│   └── {property_id}/
│       └── {property_id}_interior_{n}.jpg
├── street_view/
│   └── {property_id}_h{heading}.jpg
└── satellite/
    └── {property_id}_sat.jpg
```

---

## Reproduction Steps

Run the scripts in this exact order to reproduce the dataset from scratch.

### Step 1 — Collect listings (`data_collection.py`)

Scrapes Zillow for sold properties, downloads Street View and satellite images, and fetches Census demographics. Edit the `keyword` and `max_pages` in the `__main__` block to specify the city and how many pages to collect.

```python
# In data_collection.py, edit the main block:
new_props = fetch_all_new_listings(
    keyword='Boston, MA',
    listing_type='sold',
    start_page=1,
    max_pages=15
)
```

```bash
python data_collection.py
```

Repeat for each city. All results append to `boston_listings_with_census.csv` and deduplicate by property ID automatically.

**Cities collected for this dataset:**
Boston (including all neighborhoods), Cambridge, Somerville, Brookline, Newton, Arlington, Medford, Quincy, Braintree, Malden, Lowell, Lynn, Framingham, Waltham, Brockton

### Step 2 — Fill gaps (`data_maintenance.py`)

Downloads any missing Street View or satellite images, downloads listing photos from Zillow CDN, fills missing Census data, and fetches listing descriptions via Hasdata property detail API.

```bash
python data_maintenance.py
```

Comment out steps you don't need in the `__main__` block:

```python
if __name__ == '__main__':
    fill_missing_census()       # Step 1: fill missing census data
    fix_missing_images()        # Step 2: download missing GSV/satellite
    download_all_photos()       # Step 3: download listing photos
    fetch_property_details()    # Step 4: fetch listing descriptions
```

### Step 3 — Clean the dataset (`data_cleaning.py`)

Reads `boston_listings_with_census.csv`, parses photo URL lists, coerces numeric columns, computes the price gap variable, remaps image file paths by checking disk, and outputs `boston_cleaned.csv`.

```bash
python data_cleaning.py
```

### Step 4 — Generate CLIP embeddings (`clip_script.py`)

Encodes all four CLIP-encodable modalities (listing text, listing photos, Street View, satellite) using CLIP ViT-B/32. Saves results to `clip_embeddings.npz`. The script is resumable (if interrupted, re-running picks up from the last checkpoint)

```bash
python clip_script.py
```

**Expected output:**
```
Text:      23,580/23,624 (99.8%)
Photos:    21,811/23,624 (92.3%)
GSV:       23,624/23,624 (100.0%)
Satellite: 23,624/23,624 (100.0%)
```

---

## Resumability

All scripts are designed to be safely interrupted and restarted:
- `data_collection.py` deduplicates by property ID on every save
- `data_maintenance.py` checks for existing files before downloading and saves after each property
- `clip_script.py` checkpoints atomically every 100 properties using a write-to-temp-then-rename strategy

---

## Notes on Data Quality

- Properties outside the Greater Boston metropolitan area were dropped
- Boston neighborhood names (Allston, Brighton, Roxbury, etc.) were normalized to `Boston`
- Newton sub-neighborhoods (Newtonville, Auburndale, etc.) were normalized to `Newton`
- Census tracts with suppressed data are flagged with `census_unavailable = True`
