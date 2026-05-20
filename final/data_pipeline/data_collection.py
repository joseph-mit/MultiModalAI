# Data Collection
#
# scrapes Zillow listings and fetches Google Street View,
# satellite images, and Census demographics for each property.
# all results append to the master CSV and deduplicate by property ID.
#
# dependencies:
# python -m pip install requests pandas python-dotenv
#
# setup: create a file called .env in the same folder as this script:
#     HASDATA_KEY=your_key_here
#     GOOGLE_API_KEY=your_key_here
#     CENSUS_KEY=your_key_here
#
# run:
#   python data_collection.py
#

import time
import os
import requests
import urllib.parse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API keys
HASDATA_KEY = os.getenv('HASDATA_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_KEY = os.getenv('CENSUS_KEY')

for name, val in [('HASDATA_KEY', HASDATA_KEY), ('GOOGLE_API_KEY', GOOGLE_API_KEY), ('CENSUS_KEY', CENSUS_KEY)]:
    if not val:
        raise RuntimeError(f"Missing API key: {name}. Add it to your .env file.")

# paths
BASE = Path.home() / 'mmai' / 'mmai_midterm_report'
MASTER_CSV = BASE / 'boston_listings_with_census.csv'
GSV_DIR = BASE / 'street_view'
SAT_DIR = BASE / 'satellite'

GSV_DIR.mkdir(parents=True, exist_ok=True)
SAT_DIR.mkdir(parents=True, exist_ok=True)

if MASTER_CSV.exists():
    existing_df = pd.read_csv(MASTER_CSV, dtype={'id': str})
    seen_ids = set(existing_df['id'].astype(str))
    print(f"Loaded {len(seen_ids)} existing properties")
else:
    existing_df = pd.DataFrame()
    seen_ids = set()
    print("No existing data found")


# fetching zillow listings w hasdata zillow api

def fetch_zillow_listings(keyword, listing_type, days_on_zillow='36m', page=1):
    encoded_keyword = urllib.parse.quote(keyword)
    url = (
        f"https://api.hasdata.com/scrape/zillow/listing"
        f"?keyword={encoded_keyword}&type={listing_type}"
        f"&daysOnZillow={days_on_zillow}&page={page}"
    )
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': HASDATA_KEY
    }
    r = requests.get(url, headers=headers, timeout=30)
    print(f"  Page {page} status: {r.status_code}")
    if r.status_code == 200:
        return r.json()
    return {}


def fetch_all_new_listings(keyword='Boston, MA', listing_type='sold', days_on_zillow='36m', start_page=1, max_pages=5):
    if MASTER_CSV.exists():
        seen_ids = set(pd.read_csv(MASTER_CSV, dtype={'id': str})['id'].astype(str))
    else:
        seen_ids = set()

    all_new_props = []
    for page in range(start_page, start_page + max_pages):
        result = fetch_zillow_listings(keyword, listing_type, days_on_zillow, page=page)
        props = result.get('properties', [])
        if not props:
            print(f"No results on page {page}")
            break
        new_props = [p for p in props if str(p['id']) not in seen_ids]
        print(f"Page {page}: {len(props)} total, {len(new_props)} new")
        all_new_props.extend(new_props)
        seen_ids.update(str(p['id']) for p in new_props)
        time.sleep(1.0)
    return all_new_props


def parse_properties(props):
    rows = []
    for p in props:
        rows.append({
            'id': str(p['id']),
            'url': p['url'],
            'home_type': p['homeType'],
            'status': p['status'],
            'price': p.get('price'),
            'zestimate': p.get('zestimate'),
            'days_on_market': p.get('daysOnZillow'),
            'area_sqft': p.get('area'),
            'beds': p.get('beds'),
            'baths': p.get('baths'),
            'lat': p.get('latitude'),
            'lon': p.get('longitude'),
            'street': p['address']['street'],
            'city': p['address']['city'],
            'zipcode': p['address']['zipcode'],
            'broker': p.get('brokerName'),
            'photo_urls': p.get('photos', []),
            'first_photo': p['photos'][0] if p.get('photos') else None,
        })
    return rows


# street view

def fetch_street_view(lat, lon, prop_id, heading=0, size='640x640', fov=90):
    path = GSV_DIR / f"{prop_id}_h{heading}.jpg"
    if path.exists():
        return str(path)
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lon}"
        f"&heading={heading}&fov={fov}&pitch=0&source=outdoor&key={GOOGLE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
        with open(path, 'wb') as f:
            f.write(r.content)
        return str(path)
    return None


def fetch_all_street_views(df):
    gsv_paths = {}
    for _, row in df.iterrows():
        paths = {}
        for heading in [0, 90, 180, 270]:
            p = fetch_street_view(row['lat'], row['lon'], row['id'], heading=heading)
            paths[heading] = p
            time.sleep(0.1)
        gsv_paths[row['id']] = paths
        print(f"GSV done: {row['street']}")
    return gsv_paths


# satellite

def fetch_satellite(lat, lon, prop_id, zoom=18, size='640x640'):
    path = SAT_DIR / f"{prop_id}_sat.jpg"
    if path.exists():
        return str(path)
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}"
        f"&size={size}&maptype=satellite&key={GOOGLE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
        with open(path, 'wb') as f:
            f.write(r.content)
        return str(path)
    return None


def fetch_all_satellite(df):
    sat_paths = {}
    for _, row in df.iterrows():
        p = fetch_satellite(row['lat'], row['lon'], row['id'])
        sat_paths[row['id']] = p
        time.sleep(0.1)
        print(f"Satellite done: {row['street']}")
    return sat_paths


# census

def census_get(url, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("  All retries exhausted, skipping.")
                return None
    return None


def get_census_tract(lat, lon):
    url = (
        f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Census2020_Current"
        f"&layers=Census+Tracts&format=json"
    )
    r = census_get(url)
    if r is None or r.status_code != 200:
        print("  Tract lookup failed")
        return None
    try:
        tract = r.json()['result']['geographies']['Census Tracts'][0]
        return {
            'state':  tract['STATE'],
            'county': tract['COUNTY'],
            'tract':  tract['TRACT']
        }
    except (KeyError, IndexError) as e:
        print(f"  Tract parse error: {e}")
        return None


def get_census_demographics(state, county, tract):
    variables = ','.join([
        'B19013_001E', 'B25077_001E', 'B25064_001E',
        'B15003_022E', 'B15003_001E', 'B01003_001E',
        'B02001_002E', 'B02001_003E', 'B02001_005E', 'B03003_003E',
    ])
    url = (
        f"https://api.census.gov/data/2023/acs/acs5"
        f"?get={variables}"
        f"&for=tract:{tract}&in=state:{state}+county:{county}"
        f"&key={CENSUS_KEY}"
    )
    r = census_get(url)
    if r is None or r.status_code != 200:
        print("  Demographics failed")
        return None
    data   = r.json()
    result = dict(zip(data[0], data[1]))

    total_pop = int(result.get('B01003_001E') or 0)
    pop_25plus = max(int(result.get('B15003_001E') or 0), 1)

    return {
        'census_median_income': result.get('B19013_001E'),
        'census_median_home_value': result.get('B25077_001E'),
        'census_median_rent': result.get('B25064_001E'),
        'census_pct_educated': round(int(result.get('B15003_022E') or 0) / pop_25plus, 4),
        'census_total_population': total_pop,
        'census_pct_white': round(int(result.get('B02001_002E') or 0) / max(total_pop, 1), 4),
        'census_pct_black': round(int(result.get('B02001_003E') or 0) / max(total_pop, 1), 4),
        'census_pct_asian': round(int(result.get('B02001_005E') or 0) / max(total_pop, 1), 4),
        'census_pct_hispanic': round(int(result.get('B03003_003E') or 0) / max(total_pop, 1), 4),
    }


def fetch_all_census(df):
    census_data = {}
    for _, row in df.iterrows():
        prop_id = str(row['id'])
        tract_info = get_census_tract(row['lat'], row['lon'])
        if tract_info:
            demo = get_census_demographics(
                tract_info['state'],
                tract_info['county'],
                tract_info['tract']
            )
            census_data[prop_id] = {**(tract_info or {}), **(demo or {})}
        else:
            census_data[prop_id] = {}
        time.sleep(1.0)
        print(f"  Census done: {row['street']}")
    return census_data


# save

def save(new_df):
    if MASTER_CSV.exists():
        current_df = pd.read_csv(MASTER_CSV, dtype={'id': str})
    else:
        current_df = pd.DataFrame()

    combined = pd.concat([current_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset='id', keep='first', inplace=True)
    combined.to_csv(MASTER_CSV, index=False)
    print(f"Saved {len(combined)} total properties to {MASTER_CSV}")


# main block
# change keyword to the city you want to collect
# change max_pages to control how many pages of results to fetch (each page ~40 properties)

if __name__ == '__main__':

    print("\n1. Fetching new listings...")
    new_props = fetch_all_new_listings(
        keyword='Boston, MA',
        listing_type='sold',
        start_page=1,
        max_pages=15
    )

    if not new_props:
        print("No new properties found. Exiting.")
    else:
        print(f"\nFound {len(new_props)} new properties")

        print("\n2. Parsing properties:")
        rows = parse_properties(new_props)
        new_df = pd.DataFrame(rows)
        new_df['id'] = new_df['id'].astype(str)
        save(new_df)
        print("Tabular data saved.")

        print("\n3. Getting street view images:")
        gsv_paths = fetch_all_street_views(new_df)
        for prop_id, paths in gsv_paths.items():
            for heading, path in paths.items():
                col = f'gsv_path_{heading}'
                new_df.loc[new_df['id'] == str(prop_id), col] = path
        save(new_df)
        print("GSV paths saved.")

        print("\n4. Getting satellite images:")
        sat_paths = fetch_all_satellite(new_df)
        for prop_id, path in sat_paths.items():
            new_df.loc[new_df['id'] == str(prop_id), 'sat_path'] = path
        save(new_df)
        print("Satellite paths saved.")

        print("\n5. Getting census data:")
        census_data = fetch_all_census(new_df)
        census_df = pd.DataFrame.from_dict(census_data, orient='index')
        census_df.index.name = 'id'
        census_df = census_df.reset_index()
        census_df['id'] = census_df['id'].astype(str)
        new_df      = new_df.merge(census_df, on='id', how='left')
        save(new_df)
        print(f"\nDone. Final shape: {new_df.shape}")
