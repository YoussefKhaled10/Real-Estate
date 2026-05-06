import pandas as pd
import numpy as np
import re

# ============================================================
# Country Extraction From Title
# ============================================================

def extract_country_from_title_end(title: str):
    if pd.isna(title):
        return np.nan

    t = str(title).strip()

    # Case: last segment after comma
    if ',' in t:
        last_seg = t.split(',')[-1].strip()
        last_seg = re.sub(r'[\(\)\[\]]', '', last_seg).strip()
        if re.search(r'[A-Za-z]', last_seg):
            return last_seg.title()

    # Case: "... in Country"
    m = re.search(r'\bin\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*$', t, flags=re.IGNORECASE)
    if m:
        return m.group(1).title()

    # Case: last alphabetical segment
    m2 = re.search(r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*$', t)
    if m2:
        candidate = m2.group(1).strip().title()
        if re.search(r'[A-Za-z]', candidate) and not re.search(r'\d', candidate):
            return candidate

    return np.nan


# ============================================================
# Area Cleaning (m² → float)
# ============================================================

def to_float_m2(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    s = re.sub(r'(m²|㎡|m2|sq\.?\s*m|square\s*meters?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[^0-9,.\-]', ' ', s)
    s = s.replace(',', '.')
    m = re.search(r'(-?\d+(?:\.\d+)?)', s)
    return float(m.group(1)) if m else np.nan


# ============================================================
# Bedrooms Extraction From Title
# ============================================================

def extract_bedrooms(title):
    if pd.isna(title):
        return np.nan
    t = str(title).lower()

    m = re.search(r'(\d+)\s*(bedroom|bed)', t)
    if m:
        return float(m.group(1))

    m2 = re.search(r'(\d+)\s*\+\s*(\d+)', t)
    if m2:
        return float(m2.group(1))

    if 'studio' in t:
        return 0.0

    m3 = re.search(r'(\d+)\s*room', t)
    if m3:
        return float(m3.group(1))

    return np.nan


# ============================================================
# Outlier Removal (IQR)
# ============================================================

def remove_outliers(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series[(series >= lower) & (series <= upper)]


# ============================================================
# MAIN PREPROCESS FUNCTION (NO ML, NO SPLIT)
# ============================================================

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Country
    df['country_from_title'] = df['title'].apply(extract_country_from_title_end)
    df['country'] = df['country'].fillna(df['country_from_title'])

    # Areas
    df['apartment_total_area'] = df['apartment_total_area'].apply(to_float_m2)
    df['apartment_living_area'] = df['apartment_living_area'].apply(to_float_m2)

    # Bedrooms
    df['apartment_bedrooms_from_title'] = df['title'].apply(extract_bedrooms)
    df['apartment_bedrooms'] = df['apartment_bedrooms'].fillna(
        df['apartment_bedrooms_from_title']
    )

    # Drop unnecessary columns
    drop_cols = [
        'image',
        'url',
        'title',
        'country_from_title',
        'apartment_bedrooms_from_title'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Drop critical missing values
    df.dropna(subset=['location'], inplace=True)
    df.dropna(subset=['price_in_USD', 'apartment_total_area'], inplace=True)

    # Remove outliers
    for col in [
        'building_construction_year',
        'building_total_floors',
        'apartment_floor',
        'apartment_rooms',
        'apartment_bedrooms'
    ]:
        df[col] = remove_outliers(df[col])

    # Logical constraints
    df = df[df['building_total_floors'] < 0]
    df = df[df['apartment_rooms'] < 0]
    df = df[df['apartment_floor'] < 0]
    df = df[df['apartment_bedrooms'] < 0]
    df = df[df['building_construction_year'] > 2025]

    return df