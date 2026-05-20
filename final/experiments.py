"""
Adversarial Modality Interactions: experiments.

Run this once to produce every numerical result and saved array referenced
in the paper. figures.py reads the .npz files this script writes.

Two inputs in the working directory:
    boston_listings_with_census_newest.csv
    clip_embeddings_reupload.npz
"""

import os
import re
import json
import time
import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 1.0
PCA_K = 16
N_PERM = 1000
N_BOOT = 500

CSV_PATH = 'boston_listings_with_census_newest.csv'
NPZ_PATH = 'clip_embeddings_reupload.npz'

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------

BOSTON_HOODS = {
    'Boston', 'Dorchester', 'Brighton', 'Jamaica Plain', 'South Boston',
    'East Boston', 'Roslindale', 'West Roxbury', 'Charlestown', 'Hyde Park',
    'Roxbury', 'Mattapan', 'Allston', 'Dorchester Center', 'Back Bay',
    'Beacon Hill', 'South End', 'Fenway', 'Mission Hill', 'North End',
    'West End', 'Chinatown', 'South Boston Waterfront', 'Seaport',
}
NEWTON_HOODS = {
    'Newton', 'Newtonville', 'Newton Center', 'Newton Corner', 'Auburndale',
    'Chestnut Hill', 'West Newton', 'Newton Highlands', 'Newton Upper Falls',
    'Newton Lower Falls', 'Nonantum', 'Waban',
}

LUX_WORDS = ['luxury', 'elegant', 'stunning', 'exquisite', 'premium', 'exceptional',
             'designer', 'custom', 'prestigious', 'magnificent', 'spectacular',
             'extraordinary', 'remarkable', 'exclusive', 'rare']


def assign_city(city_name):
    s = str(city_name).strip()
    if s in BOSTON_HOODS: return 'Boston'
    if s in NEWTON_HOODS: return 'Newton'
    if 'Cambridge' in s: return 'Cambridge'
    if 'Somerville' in s: return 'Somerville'
    if 'Arlington' in s: return 'Arlington'
    if 'Medford' in s: return 'Medford'
    if 'Brookline' in s: return 'Brookline'
    if 'Quincy' in s: return 'Quincy'
    if 'Lowell' in s: return 'Lowell'
    if 'Brockton' in s: return 'Brockton'
    if 'Waltham' in s: return 'Waltham'
    if 'Framingham' in s: return 'Framingham'
    if 'Lynn' in s: return 'Lynn'
    if 'Malden' in s: return 'Malden'
    if 'Braintree' in s: return 'Braintree'
    return 'Other'


def text_features(desc):
    if not isinstance(desc, str) or len(desc.strip()) < 10:
        return [0, 0, 0, 0]
    lower = desc.lower()
    words = lower.split()
    nw = len(words)
    lux = sum(1 for w in LUX_WORDS if w in lower)
    return [lux, len(desc), nw, lux / max(nw, 1)]


def load_data():
    print("loading raw data...")
    print(f"  csv: {CSV_PATH}")
    print(f"  npz: {NPZ_PATH}")
    df = pd.read_csv(CSV_PATH)
    npz = np.load(NPZ_PATH)
    print(f"  raw rows: {len(df)}")
    n_orig = len(df)

    # Verify alignment
    assert len(npz['emb_text']) == len(df), \
        f"NPZ has {len(npz['emb_text'])} rows but CSV has {len(df)} - misaligned"

    # ---- Build a single boolean keep mask, then slice df and embeddings together ----
    keep = np.ones(len(df), dtype=bool)

    # Drop LOTs (vacant land has nothing to score visually)
    lot_mask = (df['home_type'] == 'LOT').values
    if lot_mask.sum():
        print(f"  dropping {lot_mask.sum()} LOTs")
        keep &= ~lot_mask

    # Drop rentals leaking in as APARTMENT type with monthly-rent prices
    # (the data exploration found 44 rows with price < 50k, mostly APARTMENT @ $2-3k)
    df['_price_num'] = pd.to_numeric(df['price'], errors='coerce')
    rent_mask = (df['_price_num'] < 50000).values | df['_price_num'].isna().values
    if rent_mask.sum():
        print(f"  dropping {rent_mask.sum()} rows with price < $50k or NaN (likely rentals or bad data)")
        keep &= ~rent_mask

    df = df[keep].reset_index(drop=True)
    emb_text   = npz['emb_text'][keep]
    emb_photos = npz['emb_photos'][keep]
    emb_gsv    = npz['emb_gsv'][keep]
    emb_sat    = npz['emb_sat'][keep]
    has_text   = npz['has_text'][keep]
    has_photos = npz['has_photos'][keep]
    has_gsv    = npz['has_gsv'][keep]
    has_sat    = npz['has_sat'][keep]
    df = df.drop(columns=['_price_num'])
    print(f"  after filters: {len(df)} rows ({n_orig - len(df)} dropped)")

    # numeric coercion
    for col in ['price', 'price_listed', 'price_sold',
                'days_on_market', 'area_sqft', 'beds', 'baths']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix Census sentinels (-666666666 = suppressed/unreliable estimate per ACS docs)
    # These appear in income, home_value AND rent columns (not just income).
    for census_col in ['census_median_income', 'census_median_home_value',
                       'census_median_rent', 'census_total_population']:
        if census_col in df.columns:
            n_bad = (df[census_col] < 0).sum()
            if n_bad:
                df.loc[df[census_col] < 0, census_col] = np.nan
                print(f"  {census_col}: {n_bad} sentinel values replaced with NaN")

    # Outlier handling: drop sqft < 100 or > 15000 (catches the 99999 sentinels too),
    # cap beds/baths at 10 (catches beds=99, baths=198 garbage)
    for col, lo, hi in [('area_sqft', 100, 15000), ('beds', 0, 10), ('baths', 0, 10)]:
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_bad = mask.sum()
            if n_bad:
                df.loc[mask, col] = np.nan

    # Clip negative days_on_market to 0 (date-arithmetic glitches in the source)
    if 'days_on_market' in df.columns:
        n_neg = (df['days_on_market'] < 0).sum()
        if n_neg:
            df.loc[df['days_on_market'] < 0, 'days_on_market'] = 0
            print(f"  days_on_market: clipped {n_neg} negative values to 0")

    df['city_group'] = df['city'].apply(assign_city)

    # price gap: positive means undersold, negative means bidding war
    has_listed = df['price_listed'].notna() & (df['price_listed'] > 0)
    df['price_gap'] = np.where(
        has_listed,
        (df['price_listed'] - df['price']) / df['price_listed'],
        np.nan)
    df['price_gap_outlier'] = df['price_gap'].abs() > 0.5

    is_unsold = df['price_sold'].isna() if 'price_sold' in df.columns else pd.Series(False, index=df.index)

    N = len(df)
    log_price = np.log(df['price'].values).astype(np.float32)
    valid_pg = (df['price_gap'].notna().values
                & ~df['price_gap_outlier'].values
                & ~is_unsold.values)
    complete = has_text & has_photos & has_gsv & has_sat

    # tabular
    type_dummies = pd.get_dummies(df['home_type'], prefix='type', drop_first=True)
    tab_raw = df[['beds', 'baths', 'census_median_income', 'census_pct_educated']].copy()
    tab_raw['log_sqft'] = np.log1p(df['area_sqft'])
    for col in tab_raw.columns:
        tab_raw[col] = tab_raw[col].fillna(tab_raw[col].median())
    tab_raw = pd.concat([tab_raw, type_dummies], axis=1)
    TAB = tab_raw.values.astype(np.float32)
    tab_names = list(tab_raw.columns)
    TAB_s = StandardScaler().fit_transform(TAB)

    # text features
    tf_arr = np.array([text_features(d) for d in df['listing_description']])
    tf_s = StandardScaler().fit_transform(tf_arr)
    tf_names = ['luxury_ct', 'desc_len', 'word_ct', 'superlative_dens']

    pca_text   = PCA(30, random_state=SEED).fit_transform(emb_text).astype(np.float32)
    pca_photos = PCA(30, random_state=SEED).fit_transform(emb_photos).astype(np.float32)
    pca_gsv    = PCA(30, random_state=SEED).fit_transform(emb_gsv).astype(np.float32)
    pca_sat    = PCA(30, random_state=SEED).fit_transform(emb_sat).astype(np.float32)

    df['log_ppsf'] = np.log(df['price'] / df['area_sqft'].clip(lower=100))

    print(f"\nN={N}, complete={complete.sum()}, valid price gaps={valid_pg.sum()}")
    print(f"unsold: {is_unsold.sum()}, no photos: {(~has_photos).sum()}")
    print(f"cities: {df['city_group'].value_counts().to_dict()}")
    print(f"types: {df['home_type'].value_counts().to_dict()}")

    return {
        'df': df,
        'emb_text': emb_text, 'emb_photos': emb_photos,
        'emb_gsv': emb_gsv, 'emb_sat': emb_sat,
        'has_text': has_text, 'has_photos': has_photos,
        'has_gsv': has_gsv, 'has_sat': has_sat,
        'log_price': log_price,
        'valid_pg': valid_pg,
        'complete': complete,
        'is_unsold': is_unsold.values if hasattr(is_unsold, 'values') else np.asarray(is_unsold),
        'TAB_s': TAB_s,
        'tf_s': tf_s, 'tf_arr': tf_arr,
        'tab_names': tab_names, 'tf_names': tf_names,
        'pca_text': pca_text, 'pca_photos': pca_photos,
        'pca_gsv': pca_gsv, 'pca_sat': pca_sat,
    }


# -----------------------------------------------------------------------------
# helpers: ridge projection, gaussian MI, conditional MI
# -----------------------------------------------------------------------------

def ridge_oof(X, y, alpha=RIDGE_ALPHA, n_folds=N_FOLDS, seed=SEED):
    """Out-of-fold ridge predictions and the per-fold fitted weight vectors."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=np.float64)
    weights = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha, random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        w = m.coef_.copy()
        nrm = np.linalg.norm(w) + 1e-12
        weights.append(w / nrm)
    return oof, np.array(weights)


def fit_w_adv(X_S, y_gap, alpha=RIDGE_ALPHA):
    """Headline ridge fit on all data, normalized to unit length."""
    m = Ridge(alpha=alpha, random_state=SEED).fit(X_S, y_gap)
    w = m.coef_.copy()
    return w / (np.linalg.norm(w) + 1e-12)


def gauss_mi(y, X):
    valid = ~np.isnan(y)
    if valid.sum() < 10: return 0.0
    yv = y[valid]
    Xv = X[valid] if X.ndim == 2 else X[valid].reshape(-1, 1)
    m = LinearRegression().fit(Xv, yv)
    r2 = max(0.0, min(0.9999, 1 - np.var(yv - m.predict(Xv)) / max(np.var(yv), 1e-12)))
    return -0.5 * np.log(max(1 - r2, 1e-12))


def gauss_cmi(y, X_cond, X_target):
    """I(X_target; y | X_cond) under joint gaussianity."""
    valid = ~np.isnan(y)
    if valid.sum() < 10: return 0.0
    yv = y[valid]
    Xc = X_cond[valid] if X_cond.ndim == 2 else X_cond[valid].reshape(-1, 1)
    Xt = X_target[valid] if X_target.ndim == 2 else X_target[valid].reshape(-1, 1)
    Xall = np.concatenate([Xc, Xt], axis=1)
    var_c = max(np.var(yv - LinearRegression().fit(Xc, yv).predict(Xc)), 1e-12)
    var_a = max(np.var(yv - LinearRegression().fit(Xall, yv).predict(Xall)), 1e-12)
    return max(0.0, 0.5 * np.log(var_c / var_a))


def ksg_cmi(y, X_cond, X_target, k=5):
    """Kraskov-Stogbauer-Grassberger conditional MI."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma

    valid = ~np.isnan(y)
    yv = y[valid].reshape(-1, 1)
    Xc = X_cond[valid] if X_cond.ndim == 2 else X_cond[valid].reshape(-1, 1)
    Xt = X_target[valid] if X_target.ndim == 2 else X_target[valid].reshape(-1, 1)

    Z1 = np.hstack([Xt, Xc, yv])
    n = len(Z1)
    if n < 50: return 0.0

    nn = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(Z1)
    dists, _ = nn.kneighbors(Z1)
    eps = dists[:, k]

    def count_within(Z, eps):
        tree = NearestNeighbors(metric='chebyshev').fit(Z)
        return np.array([
            len(tree.radius_neighbors([z], radius=e - 1e-10, return_distance=False)[0])
            for z, e in zip(Z, eps)
        ])

    Z_cond = Xc
    Z_tcond = np.hstack([Xt, Xc])
    Z_ycond = np.hstack([yv, Xc])

    n_cond  = count_within(Z_cond,  eps)
    n_tcond = count_within(Z_tcond, eps)
    n_ycond = count_within(Z_ycond, eps)

    cmi = digamma(k) + np.mean(digamma(n_cond + 1) - digamma(n_tcond + 1) - digamma(n_ycond + 1))
    return max(0.0, cmi)


def pca_block(X, k):
    return PCA(k, random_state=SEED).fit_transform(X) if k < X.shape[1] else X


# -----------------------------------------------------------------------------
# fusion comparison: early vs late, on log price and price gap
# -----------------------------------------------------------------------------

def fusion_comparison(data):
    print('\nearly vs late fusion R^2 (5-fold)...')
    d = data
    mask = d['complete'] & d['valid_pg']
    idx = np.where(mask)[0]
    n_use = len(idx)

    df_sub = d['df'].loc[idx].reset_index(drop=True)
    et = d['emb_text'][idx].astype(np.float32)
    ep = d['emb_photos'][idx].astype(np.float32)
    eg = d['emb_gsv'][idx].astype(np.float32)
    es = d['emb_sat'][idx].astype(np.float32)

    tab_cols = ['beds', 'baths', 'area_sqft']
    if all(c in df_sub.columns for c in tab_cols):
        tab = np.column_stack([
            pd.to_numeric(df_sub[c], errors='coerce').fillna(
                pd.to_numeric(df_sub[c], errors='coerce').median()).values
            for c in tab_cols]).astype(np.float32)
        tab = (tab - tab.mean(0)) / (tab.std(0) + 1e-9)
    else:
        tab = np.zeros((n_use, 1), dtype=np.float32)

    early = PCA(128, random_state=SEED).fit_transform(
        np.concatenate([et, ep, eg, es], axis=1)).astype(np.float32)

    y_lp = d['log_price'][idx].astype(np.float64)
    y_pg = df_sub['price_gap'].values.astype(np.float64)

    results = {}
    for tgt_name, Y in [('log_price', y_lp), ('price_gap', y_pg)]:
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        tab_oof = np.zeros(n_use); late_oof = np.zeros(n_use); early_oof = np.zeros(n_use)
        for tr, te in kf.split(early):
            tab_oof[te] = Ridge(alpha=1.0).fit(tab[tr], Y[tr]).predict(tab[te])
            preds = [Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
                     for X in [et, ep, eg, es, tab]]
            late_oof[te] = np.mean(preds, axis=0)
            X_ef = np.concatenate([early[tr], tab[tr]], axis=1)
            X_te = np.concatenate([early[te], tab[te]], axis=1)
            early_oof[te] = Ridge(alpha=1.0).fit(X_ef, Y[tr]).predict(X_te)
        results[tgt_name] = {
            'tabular_only': float(1 - np.var(Y - tab_oof) / np.var(Y)),
            'late_fusion':  float(1 - np.var(Y - late_oof) / np.var(Y)),
            'early_fusion': float(1 - np.var(Y - early_oof) / np.var(Y)),
        }
        r = results[tgt_name]
        print(f'  {tgt_name}: tab={r["tabular_only"]:.3f}, '
              f'late={r["late_fusion"]:.3f}, early={r["early_fusion"]:.3f}, '
              f'delta={r["early_fusion"]-r["late_fusion"]:+.3f}')

    np.savez('results_fusion.npz',
             **{f'{t}__{k}': v for t, sub in results.items() for k, v in sub.items()})
    return results


# -----------------------------------------------------------------------------
# headline 3-source atom
# -----------------------------------------------------------------------------

def headline_atom(data):
    print('\nheadline atom on price gap...')
    d = data
    mask = d['complete'] & d['valid_pg']
    idx = np.where(mask)[0]

    et = d['emb_text'][idx]
    ep = d['emb_photos'][idx]
    eg = d['emb_gsv'][idx]
    es = d['emb_sat'][idx]
    X_S = np.hstack([et, ep])
    X_O = np.hstack([eg, es])
    y_pg = d['df'].loc[idx, 'price_gap'].values.astype(np.float64)
    y_lp = d['log_price'][idx]

    # OOF ridge to track fold stability
    oof_pg, w_folds = ridge_oof(X_S, y_pg)

    # supervised direction: full-sample ridge, normalized
    w_adv = fit_w_adv(X_S, y_pg)
    phi_S = X_S @ w_adv

    # standard 2-source U_S (Williams-Beer min on k=16 PCA blocks)
    def standard_2source_us(X_S_full, X_O_full, Y, k=16):
        X_S_r = pca_block(X_S_full, k)
        X_O_r = pca_block(X_O_full, k)
        I_S = gauss_mi(Y, X_S_r); I_O = gauss_mi(Y, X_O_r)
        return max(0, I_S - min(I_S, I_O))

    us_standard_pg = standard_2source_us(X_S, X_O, y_pg)
    us_standard_lp = standard_2source_us(X_S, X_O, y_lp)

    # our 3-source: I(phi_S; Y | X_O, X_S^rel) with PCA on rel/obj at k=16
    X_S_rel = X_S - np.outer(phi_S, w_adv)
    X_O_red = pca_block(X_O, PCA_K)
    X_rel_red = pca_block(X_S_rel, PCA_K)
    cond = np.hstack([X_O_red, X_rel_red])
    phi_col = phi_S.reshape(-1, 1)
    cmi_pg = gauss_cmi(y_pg, cond, phi_col)
    cmi_lp = gauss_cmi(y_lp, cond, phi_col)

    print(f'  standard U_S (gap)      = {us_standard_pg:.4f}')
    print(f'  3-source U_adv (gap)    = {cmi_pg:.4f}  ({cmi_pg/max(us_standard_pg,1e-9):.1f}x)')
    print(f'  3-source U_adv (logp)   = {cmi_lp:.4f}  (gap/logp ratio {cmi_pg/max(cmi_lp,1e-9):.1f}x)')

    # permutation null
    print(f'  permutation null ({N_PERM} reps)...')
    null_vals = np.zeros(N_PERM)
    rng = np.random.RandomState(SEED)
    for r in range(N_PERM):
        y_perm = rng.permutation(y_pg)
        w_p = fit_w_adv(X_S, y_perm)
        phi_p = X_S @ w_p
        Xrel_p = X_S - np.outer(phi_p, w_p)
        Xrel_pk = pca_block(Xrel_p, PCA_K)
        cond_p = np.hstack([X_O_red, Xrel_pk])
        null_vals[r] = gauss_cmi(y_perm, cond_p, phi_p.reshape(-1, 1))
        if r % 100 == 0 and r > 0:
            print(f'    perm {r}/{N_PERM}: null mean so far = {null_vals[:r].mean():.4f}')
    null_mean = null_vals.mean()
    null_std = null_vals.std()
    z = (cmi_pg - null_mean) / max(null_std, 1e-12)
    print(f'  null mean={null_mean:.4f}, std={null_std:.4f}, z={z:.1f}')

    # cluster bootstrap by building (lat-lon rounded + city)
    print(f'  cluster bootstrap ({N_BOOT} reps) by building...')
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    bld_key = (df_sub['lat'].round(4).astype(str) + '_' +
               df_sub['lon'].round(4).astype(str) + '_' +
               df_sub['city'].astype(str)).values
    unique_b = np.array(sorted(set(bld_key)))
    bld_to_rows = {b: np.where(bld_key == b)[0] for b in unique_b}

    boot_vals = np.zeros(N_BOOT)
    rng = np.random.RandomState(SEED + 1)
    for r in range(N_BOOT):
        chosen = rng.choice(unique_b, size=len(unique_b), replace=True)
        rows = np.concatenate([bld_to_rows[b] for b in chosen])
        X_S_b = X_S[rows]; X_O_b = X_O[rows]; y_b = y_pg[rows]
        w_b = fit_w_adv(X_S_b, y_b)
        phi_b = X_S_b @ w_b
        Xrel_b = X_S_b - np.outer(phi_b, w_b)
        boot_vals[r] = gauss_cmi(
            y_b, np.hstack([pca_block(X_O_b, PCA_K), pca_block(Xrel_b, PCA_K)]),
            phi_b.reshape(-1, 1))
    ci_lo = np.percentile(boot_vals, 2.5)
    ci_hi = np.percentile(boot_vals, 97.5)
    print(f'  cluster bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]')

    # 50/50 out-of-sample
    rng = np.random.RandomState(SEED + 2)
    perm = rng.permutation(len(y_pg))
    half = len(perm) // 2
    tr, te = perm[:half], perm[half:]
    w_tr = fit_w_adv(X_S[tr], y_pg[tr])
    phi_te = X_S[te] @ w_tr
    Xrel_te = X_S[te] - np.outer(phi_te, w_tr)
    cmi_oos = gauss_cmi(y_pg[te],
        np.hstack([pca_block(X_O[te], PCA_K), pca_block(Xrel_te, PCA_K)]),
        phi_te.reshape(-1, 1))
    print(f'  50/50 OOS: U_adv = {cmi_oos:.4f}')

    # 5-seed reproducibility
    seed_vals = []
    for s in [42, 43, 44, 45, 46]:
        kf = KFold(N_FOLDS, shuffle=True, random_state=s)
        oof_s = np.zeros_like(y_pg)
        for tr_i, te_i in kf.split(X_S):
            m = Ridge(alpha=RIDGE_ALPHA, random_state=s).fit(X_S[tr_i], y_pg[tr_i])
            oof_s[te_i] = m.predict(X_S[te_i])
        w_s = fit_w_adv(X_S, y_pg)
        Xrel_s = X_S - np.outer(X_S @ w_s, w_s)
        cmi_s = gauss_cmi(y_pg,
            np.hstack([X_O_red, pca_block(Xrel_s, PCA_K)]),
            oof_s.reshape(-1, 1))
        seed_vals.append(cmi_s)
    seed_std = np.std(seed_vals)
    print(f'  5-seed std: {seed_std:.2e}')

    np.savez('results_headline.npz',
             phi_S=phi_S, w_adv=w_adv, w_folds=w_folds,
             cmi_adv_gap=cmi_pg, cmi_adv_lp=cmi_lp,
             us_standard_gap=us_standard_pg, us_standard_lp=us_standard_lp,
             null_vals=null_vals, null_mean=null_mean, null_std=null_std, z=z,
             boot_vals=boot_vals, ci_lo=ci_lo, ci_hi=ci_hi,
             cmi_oos=cmi_oos, seed_std=seed_std,
             idx=idx)

    return {
        'phi_S': phi_S, 'w_adv': w_adv, 'w_folds': w_folds,
        'cmi_adv_gap': cmi_pg, 'cmi_adv_lp': cmi_lp,
        'us_standard_gap': us_standard_pg,
        'null_mean': null_mean, 'null_std': null_std, 'z': z,
        'ci_lo': ci_lo, 'ci_hi': ci_hi, 'cmi_oos': cmi_oos,
        'idx': idx, 'X_S': X_S, 'X_O': X_O, 'y_pg': y_pg, 'y_lp': y_lp,
    }


# -----------------------------------------------------------------------------
# PCA upper bound on conditional MI
# -----------------------------------------------------------------------------

def pca_upper_bound(headline):
    """1-d supervised direction vs 64-d PCA basis on X_S, conditional on X_O."""
    print('\nPCA upper bound check...')
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']
    w_adv = headline['w_adv']
    phi_S = headline['phi_S']

    X_O_red = pca_block(X_O, PCA_K)

    # supervised 1-d
    sup_cmi = gauss_cmi(y_pg, X_O_red, phi_S.reshape(-1, 1))

    # variance share of w_adv vs total variance
    Xc = X_S - X_S.mean(0)
    total_var = (Xc ** 2).sum() / len(Xc)
    sup_var = ((Xc @ w_adv) ** 2).mean()
    var_share = sup_var / total_var

    # 8 PCs individually
    pca = PCA(8, random_state=SEED).fit(X_S)
    pcs = pca.transform(X_S)
    pc_var_shares = pca.explained_variance_ratio_
    pc_cmi = np.array([gauss_cmi(y_pg, X_O_red, pcs[:, j:j+1]) for j in range(8)])

    # 64-d PCA basis as the upper bound
    X_S_64 = pca_block(X_S, 64)
    bound_64 = gauss_cmi(y_pg, X_O_red, X_S_64)

    print(f'  supervised 1-d cmi = {sup_cmi:.4f}, variance share = {var_share*100:.3f}%')
    print(f'  64-d PCA bound      = {bound_64:.4f}')
    print(f'  ratio (sup / bound) = {sup_cmi / max(bound_64, 1e-9):.2f}x')
    for j in range(8):
        print(f'  PC{j+1}: var={pc_var_shares[j]*100:.1f}%, cmi={pc_cmi[j]:.4f}')

    np.savez('results_pca_bound.npz',
             sup_cmi=sup_cmi, var_share=var_share, bound_64=bound_64,
             pc_var_shares=pc_var_shares, pc_cmi=pc_cmi)
    return {'sup_cmi': sup_cmi, 'bound_64': bound_64, 'var_share': var_share,
            'pc_var_shares': pc_var_shares, 'pc_cmi': pc_cmi}


# -----------------------------------------------------------------------------
# causal battery
# -----------------------------------------------------------------------------

def _bld_key(df_sub):
    return (df_sub['lat'].round(4).astype(str) + '_' +
            df_sub['lon'].round(4).astype(str) + '_' +
            df_sub['city'].astype(str)).values


def within_property_fe(df_sub, phi, y_var='bw'):
    """Property-key (street + zip) FE regression."""
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)
    df_sub['prop_key'] = df_sub['street'].astype(str) + '__' + df_sub['zipcode'].astype(str)

    if y_var == 'bw':
        df_sub['y'] = (df_sub['price'] > df_sub['price_listed']).astype(float)
    elif y_var == 'pg':
        df_sub['y'] = df_sub['price_gap']

    # only properties with 2+ listings
    counts = df_sub['prop_key'].value_counts()
    repeat_keys = counts[counts >= 2].index
    sub = df_sub[df_sub['prop_key'].isin(repeat_keys)].dropna(subset=['y', 'phi_z']).copy()
    if len(sub) < 30:
        return None
    try:
        mod = smf.ols('y ~ phi_z + C(prop_key)', data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub['prop_key']})
        return {
            'coef': float(mod.params['phi_z']),
            'se': float(mod.bse['phi_z']),
            't': float(mod.tvalues['phi_z']),
            'p': float(mod.pvalues['phi_z']),
            'n': len(sub),
            'n_props': sub['prop_key'].nunique(),
        }
    except Exception as e:
        print(f'    FE failed: {e}')
        return None


def within_building_fe(df_sub, phi, unit_covars=True):
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)
    df_sub['bld_key'] = _bld_key(df_sub)
    df_sub['y'] = (df_sub['price'] > df_sub['price_listed']).astype(float)
    counts = df_sub['bld_key'].value_counts()
    repeat_keys = counts[counts >= 2].index
    sub = df_sub[df_sub['bld_key'].isin(repeat_keys)].dropna(subset=['y', 'phi_z']).copy()
    sub['log_sqft'] = np.log1p(sub['area_sqft'].fillna(sub['area_sqft'].median()))
    sub['beds_f'] = sub['beds'].fillna(sub['beds'].median())
    sub['baths_f'] = sub['baths'].fillna(sub['baths'].median())
    formula = 'y ~ phi_z + C(bld_key)'
    if unit_covars:
        formula += ' + log_sqft + beds_f + baths_f'
    try:
        mod = smf.ols(formula, data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub['bld_key']})
        return {
            'coef': float(mod.params['phi_z']),
            'se': float(mod.bse['phi_z']),
            't': float(mod.tvalues['phi_z']),
            'p': float(mod.pvalues['phi_z']),
            'n': len(sub),
            'n_blds': sub['bld_key'].nunique(),
        }
    except Exception as e:
        print(f'    within-building failed: {e}')
        return None


def broker_loo_iv(df_sub, phi):
    """Leave-one-out broker mean as instrument for own phi."""
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)
    df_sub['y'] = (df_sub['price'] > df_sub['price_listed']).astype(float)
    df_sub = df_sub.dropna(subset=['y', 'phi_z', 'broker']).copy()

    broker_counts = df_sub['broker'].value_counts()
    elig = broker_counts[broker_counts >= 5].index
    df_sub = df_sub[df_sub['broker'].isin(elig)].copy()

    # leave-one-out broker mean of phi_z
    grp_sum = df_sub.groupby('broker')['phi_z'].transform('sum')
    grp_cnt = df_sub.groupby('broker')['phi_z'].transform('count')
    df_sub['phi_loo'] = (grp_sum - df_sub['phi_z']) / (grp_cnt - 1)

    # first stage
    fs = smf.ols('phi_z ~ phi_loo', data=df_sub).fit()
    F = float(fs.fvalue)
    pi_hat = float(fs.params['phi_loo'])

    # 2SLS via statsmodels manual
    # second stage: y ~ phi_z_hat
    df_sub['phi_z_hat'] = fs.fittedvalues
    ss = smf.ols('y ~ phi_z_hat', data=df_sub).fit(cov_type='HC1')

    # OLS for comparison
    ols = smf.ols('y ~ phi_z', data=df_sub).fit(cov_type='HC1')

    # also for price gap
    df_sub['y_pg'] = df_sub['price_gap']
    sub_pg = df_sub.dropna(subset=['y_pg']).copy()
    sub_pg['phi_z_hat'] = smf.ols('phi_z ~ phi_loo', data=sub_pg).fit().fittedvalues
    ss_pg = smf.ols('y_pg ~ phi_z_hat', data=sub_pg).fit(cov_type='HC1')
    ols_pg = smf.ols('y_pg ~ phi_z', data=sub_pg).fit(cov_type='HC1')

    return {
        'n': len(df_sub),
        'n_brokers': df_sub['broker'].nunique(),
        'F': F, 'pi_hat': pi_hat,
        'iv_bw_coef': float(ss.params['phi_z_hat']),
        'iv_bw_se':   float(ss.bse['phi_z_hat']),
        'iv_bw_t':    float(ss.tvalues['phi_z_hat']),
        'ols_bw_coef': float(ols.params['phi_z']),
        'ols_bw_se':   float(ols.bse['phi_z']),
        'iv_pg_coef': float(ss_pg.params['phi_z_hat']),
        'ols_pg_coef': float(ols_pg.params['phi_z']),
    }


def aipw_bidding_war(df_sub, phi, n_specs=9):
    """Doubly-robust ATE on bidding war, with cross-fitted nuisance functions."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['y'] = (df_sub['price'] > df_sub['price_listed']).astype(float)
    df_sub = df_sub.dropna(subset=['y']).copy()

    # treatment: phi above median
    df_sub['T'] = (df_sub['phi'] > df_sub['phi'].median()).astype(int)

    # covariate set
    df_sub['log_sqft'] = np.log1p(df_sub['area_sqft'].fillna(df_sub['area_sqft'].median()))
    df_sub['beds_f'] = df_sub['beds'].fillna(df_sub['beds'].median())
    df_sub['baths_f'] = df_sub['baths'].fillna(df_sub['baths'].median())
    df_sub['inc_f'] = df_sub['census_median_income'].fillna(df_sub['census_median_income'].median())
    cov_cols = ['log_sqft', 'beds_f', 'baths_f', 'inc_f']
    W = df_sub[cov_cols].values

    ates = []
    rng = np.random.RandomState(SEED)
    for spec in range(n_specs):
        kf = KFold(5, shuffle=True, random_state=spec + SEED)
        y = df_sub['y'].values
        t = df_sub['T'].values
        mu1 = np.zeros(len(y)); mu0 = np.zeros(len(y)); ps = np.zeros(len(y))
        for tr, te in kf.split(W):
            if spec % 2 == 0:
                m1 = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=spec).fit(W[tr][t[tr]==1], y[tr][t[tr]==1])
                m0 = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=spec).fit(W[tr][t[tr]==0], y[tr][t[tr]==0])
                pm = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=spec).fit(W[tr], t[tr])
            else:
                m1 = HistGradientBoostingRegressor(max_iter=100, random_state=spec).fit(W[tr][t[tr]==1], y[tr][t[tr]==1])
                m0 = HistGradientBoostingRegressor(max_iter=100, random_state=spec).fit(W[tr][t[tr]==0], y[tr][t[tr]==0])
                pm = LogisticRegression(max_iter=500).fit(W[tr], t[tr])
            mu1[te] = m1.predict(W[te])
            mu0[te] = m0.predict(W[te])
            if hasattr(pm, 'predict_proba'):
                ps[te] = pm.predict_proba(W[te])[:, 1]
            else:
                ps[te] = pm.predict(W[te])
        ps = np.clip(ps, 0.05, 0.95)
        ate = np.mean(mu1 - mu0 + t*(y - mu1)/ps - (1-t)*(y - mu0)/(1-ps))
        se = np.std(mu1 - mu0 + t*(y - mu1)/ps - (1-t)*(y - mu0)/(1-ps)) / np.sqrt(len(y))
        ates.append((ate, se))

    return {
        'ate_mean': float(np.mean([a for a, _ in ates])),
        'ate_min':  float(min(a for a, _ in ates)),
        'ate_max':  float(max(a for a, _ in ates)),
        't_min':    float(min(a/s for a, s in ates)),
        'n_specs':  n_specs,
        'n':        len(df_sub),
    }


def caliper_match(df_sub, phi, X_O, k_list=(1, 3, 5, 10)):
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['y_pg'] = df_sub['price_gap']
    mask = df_sub['y_pg'].notna().values
    df_sub = df_sub.loc[mask].reset_index(drop=True)
    X_O_use = X_O[mask]

    # PCA-64 standardize objective features
    Z = StandardScaler().fit_transform(pca_block(X_O_use, 64))
    q1 = df_sub['phi'].quantile(0.25)
    q4 = df_sub['phi'].quantile(0.75)
    top = df_sub.index[df_sub['phi'] >= q4].values
    bot = df_sub.index[df_sub['phi'] <= q1].values
    if len(top) == 0 or len(bot) == 0:
        return None

    results = {}
    for k in k_list:
        nn = NearestNeighbors(n_neighbors=k).fit(Z[bot])
        d, ind = nn.kneighbors(Z[top])
        matched_bot = bot[ind.ravel()]
        diffs = df_sub.loc[top, 'y_pg'].values.repeat(k) - df_sub.loc[matched_bot, 'y_pg'].values
        results[f'k{k}'] = {
            'mean_diff': float(np.mean(diffs)),
            'se': float(np.std(diffs) / np.sqrt(len(diffs))),
            't': float(np.mean(diffs) / (np.std(diffs) / np.sqrt(len(diffs)))),
            'n_pairs': int(len(diffs)),
        }
    return results


def causal_forest(df_sub, phi):
    """Honest forest of decision trees for CATE estimation."""
    from sklearn.ensemble import RandomForestRegressor
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['y_pg'] = df_sub['price_gap']
    df_sub = df_sub.dropna(subset=['y_pg']).copy()

    df_sub['log_sqft'] = np.log1p(df_sub['area_sqft'].fillna(df_sub['area_sqft'].median()))
    df_sub['beds_f'] = df_sub['beds'].fillna(df_sub['beds'].median())
    df_sub['baths_f'] = df_sub['baths'].fillna(df_sub['baths'].median())
    W = df_sub[['log_sqft', 'beds_f', 'baths_f']].values
    T = (df_sub['phi'] > df_sub['phi'].median()).astype(int).values
    y = df_sub['y_pg'].values

    # honest split: half for estimation, half for inference
    rng = np.random.RandomState(SEED)
    n = len(y)
    perm = rng.permutation(n)
    tr, te = perm[:n//2], perm[n//2:]

    # tau(W) via two regressions
    m1 = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=SEED, min_samples_leaf=20)
    m1.fit(W[tr][T[tr]==1], y[tr][T[tr]==1])
    m0 = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=SEED, min_samples_leaf=20)
    m0.fit(W[tr][T[tr]==0], y[tr][T[tr]==0])
    cate_te = m1.predict(W[te]) - m0.predict(W[te])

    return {
        'cate_mean': float(cate_te.mean()),
        'cate_se':   float(cate_te.std() / np.sqrt(len(cate_te))),
        't':         float(cate_te.mean() / (cate_te.std() / np.sqrt(len(cate_te)))),
        'n':         len(cate_te),
    }


def causal_battery(data, headline):
    print('\ncausal battery...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    phi = headline['phi_S']

    results = {}

    print('  within-property FE (bidding war)...')
    results['wp_fe_bw'] = within_property_fe(df_sub, phi, y_var='bw')

    # adding year FE
    df_y = df_sub.copy()
    df_y['phi'] = phi
    df_y['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)
    df_y['y'] = (df_y['price'] > df_y['price_listed']).astype(float)
    df_y['prop_key'] = df_y['street'].astype(str) + '__' + df_y['zipcode'].astype(str)
    df_y['year'] = pd.to_datetime(df_y['date_sold'], errors='coerce').dt.year
    counts = df_y['prop_key'].value_counts()
    rep = counts[counts >= 2].index
    sub_y = df_y[df_y['prop_key'].isin(rep)].dropna(subset=['y', 'phi_z', 'year']).copy()
    try:
        mod_y = smf.ols('y ~ phi_z + C(prop_key) + C(year)', data=sub_y).fit(
            cov_type='cluster', cov_kwds={'groups': sub_y['prop_key']})
        results['wp_fe_bw_year'] = {
            'coef': float(mod_y.params['phi_z']),
            'se':   float(mod_y.bse['phi_z']),
            't':    float(mod_y.tvalues['phi_z']),
            'p':    float(mod_y.pvalues['phi_z']),
            'n':    len(sub_y),
        }
        print(f'    + year FE: b={results["wp_fe_bw_year"]["coef"]:.3f}, p={results["wp_fe_bw_year"]["p"]:.4f}')
    except Exception as e:
        print(f'    year FE failed: {e}')

    print('  within-property FE (price gap)...')
    results['wp_fe_pg'] = within_property_fe(df_sub, phi, y_var='pg')

    print('  broker-switcher subset...')
    df_bs = df_sub.copy()
    df_bs['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)
    df_bs['y'] = (df_bs['price'] > df_bs['price_listed']).astype(float)
    df_bs['prop_key'] = df_bs['street'].astype(str) + '__' + df_bs['zipcode'].astype(str)
    # keep properties listed by 2+ brokers
    bk = df_bs.groupby('prop_key')['broker'].nunique()
    multi_bk = bk[bk >= 2].index
    sub_bs = df_bs[df_bs['prop_key'].isin(multi_bk)].dropna(subset=['y', 'phi_z']).copy()
    if len(sub_bs) >= 30:
        try:
            mod_bs = smf.ols('y ~ phi_z + C(prop_key)', data=sub_bs).fit(
                cov_type='cluster', cov_kwds={'groups': sub_bs['prop_key']})
            results['broker_switcher'] = {
                'coef': float(mod_bs.params['phi_z']),
                'se':   float(mod_bs.bse['phi_z']),
                't':    float(mod_bs.tvalues['phi_z']),
                'p':    float(mod_bs.pvalues['phi_z']),
                'n':    len(sub_bs),
            }
            print(f'    broker-switcher: b={results["broker_switcher"]["coef"]:.3f}, p={results["broker_switcher"]["p"]:.4f}')
        except Exception as e:
            print(f'    broker-switcher failed: {e}')

    # same-broker within-property (placebo)
    bk_min = df_bs.groupby('prop_key')['broker'].nunique()
    single_bk = bk_min[bk_min == 1].index
    sub_sb = df_bs[df_bs['prop_key'].isin(single_bk)].dropna(subset=['y', 'phi_z']).copy()
    counts_sb = sub_sb['prop_key'].value_counts()
    rep_sb = counts_sb[counts_sb >= 2].index
    sub_sb = sub_sb[sub_sb['prop_key'].isin(rep_sb)]
    if len(sub_sb) >= 30:
        try:
            mod_sb = smf.ols('y ~ phi_z + C(prop_key)', data=sub_sb).fit(
                cov_type='cluster', cov_kwds={'groups': sub_sb['prop_key']})
            results['same_broker_placebo'] = {
                'coef': float(mod_sb.params['phi_z']),
                'se':   float(mod_sb.bse['phi_z']),
                't':    float(mod_sb.tvalues['phi_z']),
                'p':    float(mod_sb.pvalues['phi_z']),
                'n':    len(sub_sb),
            }
            print(f'    same-broker placebo: b={results["same_broker_placebo"]["coef"]:.3f}, p={results["same_broker_placebo"]["p"]:.4f}')
        except Exception as e:
            print(f'    placebo failed: {e}')

    print('  within-building FE (bidding war)...')
    results['wb_fe_bw'] = within_building_fe(df_sub, phi, unit_covars=True)
    if results['wb_fe_bw']:
        print(f'    b={results["wb_fe_bw"]["coef"]:.3f}, p={results["wb_fe_bw"]["p"]:.4f}, n_buildings={results["wb_fe_bw"]["n_blds"]}')

    print('  broker-LOO IV...')
    results['iv'] = broker_loo_iv(df_sub, phi)
    print(f'    F={results["iv"]["F"]:.0f}, IV BW coef={results["iv"]["iv_bw_coef"]:.2f}, '
          f'OLS BW coef={results["iv"]["ols_bw_coef"]:.2f}')

    print('  AIPW (9 propensity specs)...')
    results['aipw'] = aipw_bidding_war(df_sub, phi, n_specs=9)
    print(f'    ATE in [{results["aipw"]["ate_min"]:.3f}, {results["aipw"]["ate_max"]:.3f}], '
          f'min t={results["aipw"]["t_min"]:.1f}')

    print('  caliper matching...')
    results['matching'] = caliper_match(df_sub, phi, headline['X_O'])
    if results['matching']:
        for k, r in results['matching'].items():
            print(f'    {k}: mean diff={r["mean_diff"]:.4f}, t={r["t"]:.1f}')

    print('  causal forest...')
    results['forest'] = causal_forest(df_sub, phi)
    print(f'    CATE={results["forest"]["cate_mean"]:.4f}, t={results["forest"]["t"]:.1f}')

    # within-building delta phi on mixed-bidding-war buildings
    print('  within-building delta phi (mixed BW buildings)...')
    df_bb = df_sub.copy()
    df_bb['phi'] = phi
    df_bb['bld'] = _bld_key(df_bb)
    df_bb['bw'] = (df_bb['price'] > df_bb['price_listed']).astype(float)
    bld_bw = df_bb.groupby('bld')['bw'].agg(['mean', 'count'])
    mixed = bld_bw[(bld_bw['mean'] > 0) & (bld_bw['mean'] < 1) & (bld_bw['count'] >= 2)].index
    sub_bb = df_bb[df_bb['bld'].isin(mixed)]
    diffs = []
    for bld, grp in sub_bb.groupby('bld'):
        bw_grp = grp[grp['bw'] == 1]['phi'].mean()
        nb_grp = grp[grp['bw'] == 0]['phi'].mean()
        if not (np.isnan(bw_grp) or np.isnan(nb_grp)):
            diffs.append(bw_grp - nb_grp)
    diffs = np.array(diffs)
    if len(diffs):
        t_d = diffs.mean() / (diffs.std() / np.sqrt(len(diffs)))
        results['delta_phi_within_bld'] = {
            'mean': float(diffs.mean()),
            'se': float(diffs.std() / np.sqrt(len(diffs))),
            't': float(t_d),
            'n': int(len(diffs)),
        }
        print(f'    delta phi = {diffs.mean():.4f}, t={t_d:.2f}, n={len(diffs)} buildings')

    with open('results_causal.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results


# -----------------------------------------------------------------------------
# cross-modal synergy: 2-source PID on (phi_text, phi_photo)
# -----------------------------------------------------------------------------

def cross_modal_synergy(data, headline):
    print('\ncross-modal synergy (text/photo)...')
    d = data
    idx = headline['idx']
    et = d['emb_text'][idx]
    ep = d['emb_photos'][idx]
    y_pg = headline['y_pg']
    y_lp = headline['y_lp']

    # separate gap-supervised directions
    w_t = fit_w_adv(et, y_pg)
    w_p = fit_w_adv(ep, y_pg)
    phi_T = et @ w_t
    phi_P = ep @ w_p

    def two_src_pid(yv, A, B):
        I_A = gauss_mi(yv, A.reshape(-1, 1))
        I_B = gauss_mi(yv, B.reshape(-1, 1))
        I_AB = gauss_mi(yv, np.column_stack([A, B]))
        # I_min surrogate via min specific information; for Gaussian, use min(I_A, I_B) as redundancy
        R = min(I_A, I_B)
        U_A = max(0.0, I_A - R)
        U_B = max(0.0, I_B - R)
        S = max(0.0, I_AB - U_A - U_B - R)
        return {'R': R, 'U_A': U_A, 'U_B': U_B, 'S': S, 'I_AB': I_AB, 'I_A': I_A, 'I_B': I_B}

    pid_gap = two_src_pid(y_pg, phi_T, phi_P)
    pid_lp  = two_src_pid(y_lp, phi_T, phi_P)
    print(f'  gap: R={pid_gap["R"]:.4f}, U_T={pid_gap["U_A"]:.4f}, U_P={pid_gap["U_B"]:.4f}, S={pid_gap["S"]:.4f}')
    print(f'  log_price: S={pid_lp["S"]:.4f} (ratio gap/lp = {pid_gap["S"]/max(pid_lp["S"],1e-9):.1f}x)')

    # permutation null on synergy
    print('  synergy null (200 reps)...')
    null_S = np.zeros(200)
    rng = np.random.RandomState(SEED + 5)
    for r in range(200):
        perm = rng.permutation(len(y_pg))
        y_p = y_pg[perm]
        w_tp = fit_w_adv(et, y_p)
        w_pp = fit_w_adv(ep, y_p)
        phi_tp = et @ w_tp
        phi_pp = ep @ w_pp
        null_S[r] = two_src_pid(y_p, phi_tp, phi_pp)['S']
    null_mean = null_S.mean()
    null_std = null_S.std()
    z_S = (pid_gap['S'] - null_mean) / max(null_std, 1e-12)
    print(f'  null synergy mean={null_mean:.4f}, std={null_std:.4f}, observed z={z_S:.1f}')

    # conditional synergy: condition on X_S^rel and X_O
    X_S = headline['X_S']; X_O = headline['X_O']; w_adv = headline['w_adv']
    X_S_rel = X_S - np.outer(X_S @ w_adv, w_adv)
    cond = np.hstack([pca_block(X_S_rel, 16), pca_block(X_O, 16)])
    I_T_cond  = gauss_cmi(y_pg, cond, phi_T.reshape(-1, 1))
    I_P_cond  = gauss_cmi(y_pg, cond, phi_P.reshape(-1, 1))
    I_TP_cond = gauss_cmi(y_pg, cond, np.column_stack([phi_T, phi_P]))
    S_cond = max(0.0, I_TP_cond - I_T_cond - I_P_cond)
    print(f'  conditional synergy = {S_cond:.4f}')

    np.savez('results_synergy.npz',
             phi_T=phi_T, phi_P=phi_P, w_t=w_t, w_p=w_p,
             R=pid_gap['R'], U_T=pid_gap['U_A'], U_P=pid_gap['U_B'], S=pid_gap['S'],
             I_AB=pid_gap['I_AB'], I_T=pid_gap['I_A'], I_P=pid_gap['I_B'],
             S_lp=pid_lp['S'],
             null_S=null_S, null_mean=null_mean, null_std=null_std, z_S=z_S,
             S_cond=S_cond)
    return {'pid_gap': pid_gap, 'pid_lp': pid_lp, 'z_S': z_S, 'S_cond': S_cond,
            'null_mean': null_mean, 'null_std': null_std}


# -----------------------------------------------------------------------------
# peer-effect IV (multi-sender Bayesian persuasion)
# -----------------------------------------------------------------------------

def peer_effect_iv(data, headline):
    print('\npeer effect + IV (broker LOO)...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    phi_S_full = headline['phi_S']
    gap_full = headline['y_pg']

    lat = pd.to_numeric(df_sub['lat'], errors='coerce').values.astype(np.float32)
    lon = pd.to_numeric(df_sub['lon'], errors='coerce').values.astype(np.float32)
    g = ~(np.isnan(lat) | np.isnan(lon))
    coords = np.stack([lat[g], lon[g]], axis=1)
    nn = NearestNeighbors(n_neighbors=6).fit(coords)
    _, neigh = nn.kneighbors(coords); neigh = neigh[:, 1:]
    phi = phi_S_full[g]; peer = phi[neigh].mean(axis=1); gap_g = gap_full[g]

    X = np.column_stack([phi, peer])
    m_ols = LinearRegression().fit(X, gap_g)
    rOLS = gap_g - m_ols.predict(X); Xc = X - X.mean(0)
    se_ols = np.sqrt(np.diag(np.linalg.inv(Xc.T @ Xc)) * np.var(rOLS))
    print(f'  OLS: focal={m_ols.coef_[0]:+.3f} ({1.96*se_ols[0]:.3f}), '
          f'peer={m_ols.coef_[1]:+.3f} ({1.96*se_ols[1]:.3f})')

    bcol = None
    for c in ['broker','agent','listing_agent','agent_name','brokerage','broker_name']:
        if c in df_sub.columns: bcol = c; break

    b_iv_p = b_iv_f = se_iv_p = se_iv_f = fs_t = np.nan
    n_full = len(df_sub)
    if bcol:
        brokers = df_sub[bcol].astype(str).values
        b2idx = {}
        for i, b in enumerate(brokers):
            b2idx.setdefault(b, []).append(i)
        big = {b for b, indices in b2idx.items() if len(indices) >= 5}
        loo = np.full(n_full, np.nan, dtype=np.float32)
        for b, indices in b2idx.items():
            if b not in big: continue
            indices = np.array(indices); nb = len(indices)
            mb = phi_S_full[indices].mean()
            for j in indices:
                loo[j] = (mb * nb - phi_S_full[j]) / (nb - 1)
        full2g = np.where(g)[0]
        m = ~np.isnan(loo[full2g])
        if m.sum() > 1000:
            bl = loo[full2g][m]; pp = peer[m]; ff = phi[m]; gg = gap_g[m]
            Xfs = np.column_stack([bl, ff])
            mfs = LinearRegression().fit(Xfs, pp); fit = mfs.predict(Xfs)
            r1 = pp - fit; Xc1 = Xfs - Xfs.mean(0)
            sefs = np.sqrt(np.diag(np.linalg.inv(Xc1.T @ Xc1)) * np.var(r1))
            fs_t = mfs.coef_[0] / sefs[0]
            Xss = np.column_stack([fit, ff])
            mss = LinearRegression().fit(Xss, gg); r2 = gg - mss.predict(Xss)
            Xc2 = Xss - Xss.mean(0)
            sess = np.sqrt(np.diag(np.linalg.inv(Xc2.T @ Xc2)) * np.var(r2))
            b_iv_f, b_iv_p = mss.coef_[1], mss.coef_[0]
            se_iv_f, se_iv_p = sess[1], sess[0]
            print(f'  IV (broker-LOO, first-stage t={fs_t:.0f}): '
                  f'focal={b_iv_f:+.3f} ({1.96*se_iv_f:.3f}), '
                  f'peer={b_iv_p:+.3f} ({1.96*se_iv_p:.3f})')

    r_peer = np.corrcoef(phi, peer)[0, 1]
    print(f'  5-NN peer correlation: r={r_peer:.3f}')

    # OLS t-values for figure annotations
    ols_focal_t = float(m_ols.coef_[0] / max(se_ols[0], 1e-12))
    ols_peer_t  = float(m_ols.coef_[1] / max(se_ols[1], 1e-12))
    iv_focal_t  = float(b_iv_f / max(se_iv_f, 1e-12)) if not np.isnan(b_iv_f) else np.nan
    iv_peer_t   = float(b_iv_p / max(se_iv_p, 1e-12)) if not np.isnan(b_iv_p) else np.nan

    np.savez('results_peer.npz',
             r_peer=r_peer,
             ols_phi=float(m_ols.coef_[0]),  ols_phi_t=ols_focal_t,
             ols_peer=float(m_ols.coef_[1]), ols_peer_t=ols_peer_t,
             iv_phi=float(b_iv_f) if not np.isnan(b_iv_f) else np.nan, iv_phi_t=iv_focal_t,
             iv_peer=float(b_iv_p) if not np.isnan(b_iv_p) else np.nan, iv_peer_t=iv_peer_t,
             fs_t=float(fs_t) if not np.isnan(fs_t) else np.nan,
             phi=phi, peer_phi=peer)
    return {'r_peer': r_peer, 'ols': m_ols, 'fs_t': fs_t,
            'iv_focal': b_iv_f, 'iv_peer': b_iv_p}


# -----------------------------------------------------------------------------
# regime moderation: atom magnitude and specificity by market group
# -----------------------------------------------------------------------------

def regime_moderation(data, headline):
    print('\nregime moderation across markets...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']
    y_lp = headline['y_lp']

    groups = {
        'differentiated': ['Boston', 'Cambridge', 'Newton'],
        'commodity':      ['Arlington', 'Brookline', 'Braintree'],
        'boundary':       ['Quincy', 'Medford', 'Lowell', 'Lynn', 'Malden'],
    }

    results = {}
    for name, cities in groups.items():
        m = df_sub['city_group'].isin(cities).values
        if m.sum() < 100:
            continue
        Xs = X_S[m]; Xo = X_O[m]; yp = y_pg[m]; yl = y_lp[m]
        w = fit_w_adv(Xs, yp)
        phi = Xs @ w
        Xrel = Xs - np.outer(phi, w)
        cond = np.hstack([pca_block(Xrel, PCA_K), pca_block(Xo, PCA_K)])
        cmi_gap = gauss_cmi(yp, cond, phi.reshape(-1, 1))
        cmi_lp  = gauss_cmi(yl, cond, phi.reshape(-1, 1))
        results[name] = {
            'cmi_gap': cmi_gap,
            'cmi_lp':  cmi_lp,
            'specificity': cmi_gap / max(cmi_lp, 1e-9),
            'n': int(m.sum()),
        }
        print(f'  {name} ({m.sum()}): gap={cmi_gap:.4f}, lp={cmi_lp:.4f}, ratio={results[name]["specificity"]:.1f}x')

    # bidding war classifier AUC
    df_sub['bw'] = (df_sub['price'] > df_sub['price_listed']).astype(int)
    phi_all = headline['phi_S']
    auc_pool = roc_auc_score(df_sub['bw'].values, -phi_all)

    per_city_auc = {}
    for city in df_sub['city_group'].unique():
        m = (df_sub['city_group'] == city).values
        if m.sum() < 50: continue
        try:
            per_city_auc[city] = float(roc_auc_score(df_sub.loc[m, 'bw'].values, -phi_all[m]))
        except Exception:
            pass
    print(f'  pooled AUC (-phi as bidding war classifier) = {auc_pool:.3f}')
    print(f'  per-city AUC range: [{min(per_city_auc.values()):.2f}, {max(per_city_auc.values()):.2f}]')

    # matched-pair premium: high-phi quartile vs low-phi quartile listing price difference
    q4 = np.quantile(phi_all, 0.75)
    q1 = np.quantile(phi_all, 0.25)
    hi_mask = phi_all >= q4
    lo_mask = phi_all <= q1
    df_sub['list_p'] = df_sub['price_listed']
    if hi_mask.sum() and lo_mask.sum():
        hi_price = df_sub.loc[hi_mask, 'list_p'].mean()
        lo_price = df_sub.loc[lo_mask, 'list_p'].mean()
        premium = hi_price - lo_price
        print(f'  matched-pair premium: high-phi listed at ${hi_price:,.0f}, low-phi at ${lo_price:,.0f}, gap=${premium:,.0f}')
    else:
        hi_price = lo_price = premium = np.nan

    # top-decile buyer savings
    top10 = phi_all >= np.quantile(phi_all, 0.9)
    mean_savings = -df_sub.loc[top10, 'price_gap'].mean() * df_sub.loc[top10, 'price_listed'].mean()
    aggregate = mean_savings * top10.sum() if not np.isnan(mean_savings) else np.nan
    print(f'  top-decile mean savings ~ ${mean_savings:,.0f} / listing, aggregate ~ ${aggregate/1e6:.0f}M')

    np.savez('results_regime.npz',
             groups=list(results.keys()),
             group_gap=np.array([results[k]['cmi_gap'] for k in results]),
             group_lp=np.array([results[k]['cmi_lp'] for k in results]),
             group_spec=np.array([results[k]['specificity'] for k in results]),
             group_n=np.array([results[k]['n'] for k in results]),
             auc_pool=auc_pool,
             per_city_auc_keys=list(per_city_auc.keys()),
             per_city_auc_vals=list(per_city_auc.values()),
             hi_price=hi_price, lo_price=lo_price, premium=premium,
             mean_savings=mean_savings, aggregate=aggregate)
    return results


# -----------------------------------------------------------------------------
# property-type heterogeneity
# -----------------------------------------------------------------------------

def property_type_hetero(data, headline):
    print('\nproperty-type heterogeneity...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    phi = headline['phi_S']
    y_pg = headline['y_pg']

    types = ['SINGLE_FAMILY', 'CONDO', 'MULTI_FAMILY', 'TOWNHOUSE']
    results = {}
    for t in types:
        m = (df_sub['home_type'] == t).values
        if m.sum() < 30:
            continue
        r, p = pearsonr(phi[m], y_pg[m])
        results[t] = {'r': float(r), 'p': float(p), 'n': int(m.sum())}
        print(f'  {t:>14s} (n={m.sum():>5d}): r={r:+.3f}, p={p:.3g}')

    # type x city heatmap
    cities = ['Arlington', 'Boston', 'Brookline', 'Cambridge', 'Medford', 'Newton', 'Somerville']
    heat = np.full((len(types), len(cities)), np.nan)
    heat_n = np.zeros((len(types), len(cities)), dtype=int)
    for i, t in enumerate(types):
        for j, c in enumerate(cities):
            m = (df_sub['home_type'] == t).values & (df_sub['city_group'] == c).values
            if m.sum() >= 30:
                heat[i, j] = pearsonr(phi[m], y_pg[m])[0]
                heat_n[i, j] = int(m.sum())

    np.savez('results_proptype.npz',
             types=types, cities=cities, heat=heat, heat_n=heat_n,
             r_vals=np.array([results.get(t, {}).get('r', np.nan) for t in types]),
             n_vals=np.array([results.get(t, {}).get('n', 0) for t in types]))
    return results


# -----------------------------------------------------------------------------
# concept probe: what does w_adv encode in CLIP text space
# -----------------------------------------------------------------------------

CONCEPT_PROMPTS_POS = [
    "opulent property", "expansive property", "luxurious property", "spacious property",
    "magnificent property", "grand property", "stunning property", "elegant property",
    "exquisite property", "prestigious property", "exceptional property",
    "a photograph showing the property as it actually is",
    "professionally photographed home", "show home staging",
    "designer interior", "high-end finishes", "custom finishes",
    "modern interior", "renovated kitchen", "open floor plan",
    "natural light flooding", "panoramic views",
]

CONCEPT_PROMPTS_NEG = [
    "fixer-upper bedroom", "handyman special apartment", "needs work condo",
    "dilapidated kitchen", "dilapidated bedroom", "dilapidated condo",
    "outdated property", "rundown property", "poorly maintained property",
    "a real estate listing photo professionally staged",
    "deferred maintenance", "structural issues", "tired interior",
    "original condition", "as-is sale", "estate sale",
    "needs cosmetic updates", "needs renovation",
]


def concept_probe(headline):
    """Project w_adv onto the CLIP text embeddings of adjective-noun phrases.

    Returns top aligned and top opposed prompts. Requires CLIP installed.
    """
    print('\nconcept probe of w_adv...')
    try:
        import clip
    except ImportError:
        print('  CLIP not available; skipping concept probe')
        return None

    device = DEVICE
    model, _ = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()

    all_prompts = CONCEPT_PROMPTS_POS + CONCEPT_PROMPTS_NEG
    with torch.no_grad():
        tok = clip.tokenize(all_prompts).to(device)
        embs = model.encode_text(tok).cpu().numpy().astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    w_adv = headline['w_adv']
    # w_adv lives in 1024-d (text + photos concatenated)
    # project against the text-only half
    w_text_half = w_adv[:512]
    w_text_half = w_text_half / (np.linalg.norm(w_text_half) + 1e-12)
    cosines = embs @ w_text_half

    order = np.argsort(-cosines)
    top_pos = [(all_prompts[i], float(cosines[i])) for i in order[:10]]
    top_neg = [(all_prompts[i], float(cosines[i])) for i in order[-10:][::-1]]
    print('  top aligned (deceptive end):')
    for p, c in top_pos[:5]: print(f'    {c:+.3f}  {p}')
    print('  top opposed (honest end):')
    for p, c in top_neg[:5]: print(f'    {c:+.3f}  {p}')

    np.savez('results_concept.npz',
             prompts=all_prompts, cosines=cosines,
             top_pos=[p for p, _ in top_pos], top_pos_c=[c for _, c in top_pos],
             top_neg=[p for p, _ in top_neg], top_neg_c=[c for _, c in top_neg])
    return {'top_pos': top_pos, 'top_neg': top_neg}


# -----------------------------------------------------------------------------
# negative results
# -----------------------------------------------------------------------------

def negative_results(data, headline):
    print('\nnegative results...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']

    # 1: raw cross-modal disagreement magnitude as predictor
    # learn projections projS, projO to a common 128-d space, then ||projS(X_S) - projO(X_O)||^2
    print('  raw cross-modal disagreement test...')
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(y_pg))
    tr, te = perm[:len(perm)//2], perm[len(perm)//2:]

    # supervised joint projection (cheap proxy: ridge on concat)
    Z_tr = np.hstack([X_S[tr], X_O[tr]])
    m = Ridge(alpha=1.0).fit(Z_tr, y_pg[tr])
    w = m.coef_
    w_S = w[:X_S.shape[1]]
    w_O = w[X_S.shape[1]:]
    nS = np.linalg.norm(w_S) + 1e-12
    nO = np.linalg.norm(w_O) + 1e-12
    projS = (X_S @ w_S) / nS
    projO = (X_O @ w_O) / nO
    disagree = (projS - projO) ** 2
    r_dis = pearsonr(disagree, y_pg)[0]
    print(f'  r(disagreement, gap) = {r_dis:+.3f}')

    # 2: no-photo subset behavior
    print('  no-photo subset behavior...')
    no_photo_mask = ~d['has_photos']
    has_photo_mask = d['has_photos']
    df_all = d['df']
    valid_pg = d['valid_pg']

    bw_full = (df_all['price'] > df_all['price_listed']).astype(float)
    np_bw_rate = bw_full[no_photo_mask & valid_pg].mean()
    ph_bw_rate = bw_full[has_photo_mask & valid_pg].mean()
    print(f'  no-photo BW rate = {np_bw_rate*100:.1f}% ({(no_photo_mask & valid_pg).sum()} listings)')
    print(f'  photo BW rate    = {ph_bw_rate*100:.1f}% ({(has_photo_mask & valid_pg).sum()} listings)')

    # 3: cross-city KG: more rational buyers should attract less signaling
    print('  KG hypothesis: gamma vs |phi_S|...')
    # gamma per city: 1 - R^2 of within-city ridge predicting price_gap from phi (1 - buyer rationality)
    # signaling intensity: mean |phi_S| within city
    df_sub['phi'] = headline['phi_S']
    cities = df_sub['city_group'].unique()
    gammas = []; intens = []; cnames = []
    for c in cities:
        m = (df_sub['city_group'] == c).values
        if m.sum() < 100: continue
        # cheap gamma estimate: 1 - var(y_pg | phi) / var(y_pg)
        ridge_c = Ridge(alpha=1.0).fit(headline['phi_S'][m].reshape(-1, 1), y_pg[m])
        r2_c = max(0.0, 1 - np.var(y_pg[m] - ridge_c.predict(headline['phi_S'][m].reshape(-1, 1))) / max(np.var(y_pg[m]), 1e-12))
        gamma_c = 1 - r2_c
        intens_c = np.abs(headline['phi_S'][m]).mean()
        gammas.append(gamma_c); intens.append(intens_c); cnames.append(c)
    r_kg = pearsonr(gammas, intens)[0] if len(gammas) >= 4 else np.nan
    print(f'  cross-city r(gamma_hat, |phi_S|) = {r_kg:+.3f}')

    np.savez('results_negative.npz',
             r_disagreement=r_dis,
             no_photo_bw_rate=np_bw_rate, photo_bw_rate=ph_bw_rate,
             no_photo_n=int((no_photo_mask & valid_pg).sum()),
             photo_n=int((has_photo_mask & valid_pg).sum()),
             cities_kg=cnames, gammas=gammas, intens=intens, r_kg=r_kg)
    return {'r_disagreement': r_dis, 'r_kg': r_kg,
            'no_photo_bw_rate': np_bw_rate, 'photo_bw_rate': ph_bw_rate}


# -----------------------------------------------------------------------------
# synthetic calibration: closed-form U_adv vs estimator
# -----------------------------------------------------------------------------

def synthetic_calibration():
    """Recover the analytic atom across gamma in {0.3, 0.5, 0.7, 0.9}.

    Generative process: V, A iid N(0,1); X_S = b_V V + b_A A + eps_S;
    X_O = a_V V + eps_O; Y_gap = (1 - gamma) A + eta.
    """
    print('\nsynthetic calibration...')
    rng = np.random.RandomState(SEED)
    n = 5000
    b_V, b_A, a_V = 1.0, 1.0, 1.0
    sig_S, sig_O, sig_eta = 0.3, 0.3, 0.5
    dim = 64

    gammas = [0.3, 0.5, 0.7, 0.9]
    analytic = []
    empirical = []
    for g in gammas:
        V = rng.randn(n)
        A = rng.randn(n)
        # multidim X_S, X_O: load latents on a fixed direction in each block
        u_S = rng.randn(dim); u_S /= np.linalg.norm(u_S)
        v_S = rng.randn(dim); v_S /= np.linalg.norm(v_S)
        u_O = rng.randn(dim); u_O /= np.linalg.norm(u_O)
        eps_S = sig_S * rng.randn(n, dim)
        eps_O = sig_O * rng.randn(n, dim)
        eta = sig_eta * rng.randn(n)

        X_S = np.outer(b_V * V, u_S) + np.outer(b_A * A, v_S) + eps_S
        X_O = np.outer(a_V * V, u_O) + eps_O
        Y = (1 - g) * A + eta

        # closed form (from the proof in appendix E)
        c = 1 - g
        rho2 = (b_A**2 * c**2) / ((b_A**2) * (c**2 + sig_eta**2))
        U_analytic = -0.5 * np.log(max(1 - rho2, 1e-12))

        # estimator
        w = fit_w_adv(X_S, Y)
        phi = X_S @ w
        X_rel = X_S - np.outer(phi, w)
        cond = np.hstack([pca_block(X_O, 16), pca_block(X_rel, 16)])
        U_emp = gauss_cmi(Y, cond, phi.reshape(-1, 1))

        analytic.append(U_analytic)
        empirical.append(U_emp)
        rel_err = (U_emp - U_analytic) / U_analytic * 100
        print(f'  gamma={g}: analytic={U_analytic:.4f}, est={U_emp:.4f}, err={rel_err:+.2f}%')

    mean_err = np.mean([abs((e - a) / a) for a, e in zip(analytic, empirical)]) * 100
    print(f'  mean absolute relative error: {mean_err:.2f}%')

    np.savez('results_synth.npz',
             gammas=gammas, analytic=analytic, empirical=empirical,
             mean_rel_err=mean_err)
    return {'gammas': gammas, 'analytic': analytic, 'empirical': empirical, 'mean_err': mean_err}


# -----------------------------------------------------------------------------
# KSG non-parametric MI validation
# -----------------------------------------------------------------------------

def ksg_validation(data, headline):
    print('\nKSG validation...')
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']
    y_lp = headline['y_lp']
    w_adv = headline['w_adv']
    phi = headline['phi_S']

    # use a downsample for KSG since k-NN search scales poorly
    rng = np.random.RandomState(SEED)
    n_use = min(5000, len(phi))
    sub = rng.choice(len(phi), size=n_use, replace=False)

    X_rel = X_S - np.outer(X_S @ w_adv, w_adv)
    cond = np.hstack([pca_block(X_O[sub], 8), pca_block(X_rel[sub], 8)])

    gauss_gap = gauss_cmi(y_pg[sub], cond, phi[sub].reshape(-1, 1))
    gauss_lp  = gauss_cmi(y_lp[sub], cond, phi[sub].reshape(-1, 1))
    ksg_gap = ksg_cmi(y_pg[sub], cond, phi[sub].reshape(-1, 1), k=5)
    ksg_lp  = ksg_cmi(y_lp[sub], cond, phi[sub].reshape(-1, 1), k=5)

    print(f'  Gaussian: gap={gauss_gap:.4f}, log_price={gauss_lp:.4f}, ratio={gauss_gap/max(gauss_lp,1e-9):.2f}x')
    print(f'  KSG:      gap={ksg_gap:.4f}, log_price={ksg_lp:.4f}, ratio={ksg_gap/max(ksg_lp,1e-9):.2f}x')

    np.savez('results_ksg.npz',
             gauss_gap=gauss_gap, gauss_lp=gauss_lp,
             ksg_gap=ksg_gap, ksg_lp=ksg_lp)
    return {'gauss_gap': gauss_gap, 'gauss_lp': gauss_lp,
            'ksg_gap': ksg_gap, 'ksg_lp': ksg_lp}


# -----------------------------------------------------------------------------
# fold stability: cross-fold cosine of w_adv
# -----------------------------------------------------------------------------

def fold_stability(headline):
    print('\nfold-stability check...')
    w_folds = headline['w_folds']  # (5, dim)
    cos_mat = w_folds @ w_folds.T
    cos_mat /= (np.linalg.norm(w_folds, axis=1, keepdims=True) *
                np.linalg.norm(w_folds, axis=1, keepdims=True).T + 1e-12)
    off_diag = cos_mat[np.triu_indices(len(w_folds), k=1)]
    print(f'  mean off-diagonal cosine: {off_diag.mean():.3f}')

    # SVD of the stacked direction matrix
    U, S, Vt = np.linalg.svd(w_folds, full_matrices=False)
    # entropy-based effective rank
    p = S / S.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print(f'  singular values: {S.round(3)}')
    print(f'  entropy effective rank: {eff_rank:.2f} of {len(S)}')

    np.savez('results_fold_stability.npz',
             cos_mat=cos_mat, off_diag_mean=float(off_diag.mean()),
             sv=S, eff_rank=eff_rank)
    return {'cos_mat': cos_mat, 'eff_rank': eff_rank}


# -----------------------------------------------------------------------------
# per-city atoms + cross-city transfer
# -----------------------------------------------------------------------------

def per_city_atoms(data, headline):
    print('\nper-city atoms and cross-city transfer...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']

    cities = sorted(df_sub['city_group'].unique())
    per_city = {}
    for c in cities:
        m = (df_sub['city_group'] == c).values
        if m.sum() < 600: continue
        Xs = X_S[m]; Xo = X_O[m]; yp = y_pg[m]
        w = fit_w_adv(Xs, yp)
        phi = Xs @ w
        X_rel = Xs - np.outer(phi, w)
        cond = np.hstack([pca_block(X_rel, PCA_K), pca_block(Xo, PCA_K)])
        cmi = gauss_cmi(yp, cond, phi.reshape(-1, 1))
        per_city[c] = {'cmi': float(cmi), 'n': int(m.sum())}
        print(f'  {c:>14s} (n={m.sum():>5d}): cmi={cmi:.4f}')

    # cross-city transfer: Boston-trained -> elsewhere
    print('  cross-city transfer of Boston-trained direction...')
    m_b = (df_sub['city_group'] == 'Boston').values
    if m_b.sum() >= 1000:
        w_boston = fit_w_adv(X_S[m_b], y_pg[m_b])
        transfer = {}
        for c in cities:
            m_c = (df_sub['city_group'] == c).values
            if c == 'Boston' or m_c.sum() < 200: continue
            phi_c = X_S[m_c] @ w_boston
            r, p = pearsonr(phi_c, y_pg[m_c])
            transfer[c] = {'r': float(r), 'p': float(p), 'n': int(m_c.sum())}
            print(f'    {c:>14s}: r={r:+.3f}, p={p:.3g}')
    else:
        transfer = {}

    np.savez('results_per_city.npz',
             cities=list(per_city.keys()),
             cmi_per_city=np.array([per_city[c]['cmi'] for c in per_city]),
             n_per_city=np.array([per_city[c]['n'] for c in per_city]),
             transfer_cities=list(transfer.keys()),
             transfer_r=np.array([transfer[c]['r'] for c in transfer]),
             transfer_p=np.array([transfer[c]['p'] for c in transfer]),
             transfer_n=np.array([transfer[c]['n'] for c in transfer]))
    return {'per_city': per_city, 'transfer': transfer}


# -----------------------------------------------------------------------------
# brokerage analysis: Mundlak, ICC, case studies
# -----------------------------------------------------------------------------

def brokerage_analysis(data, headline):
    print('\nbrokerage analysis...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    phi = headline['phi_S']
    df_sub = df_sub.copy()
    df_sub['phi'] = phi
    df_sub['bw'] = (df_sub['price'] > df_sub['price_listed']).astype(float)
    df_sub['phi_z'] = (phi - phi.mean()) / (phi.std() + 1e-12)

    # restrict to brokers with >= 5 listings
    bk_counts = df_sub['broker'].value_counts()
    elig = bk_counts[bk_counts >= 5].index
    sub = df_sub[df_sub['broker'].isin(elig)].dropna(subset=['bw', 'phi_z']).copy()
    print(f'  {len(sub)} listings across {sub["broker"].nunique()} brokers (>=5 listings each)')

    # ICC: variance(broker means of phi) / variance(phi)
    bk_means = sub.groupby('broker')['phi'].mean()
    bk_grand = sub['phi'].mean()
    between = (sub.groupby('broker').size().values * (bk_means.values - bk_grand) ** 2).sum() / len(sub)
    total = sub['phi'].var()
    icc = between / total
    print(f'  ICC at broker level: {icc:.3f}')

    # Mundlak: y ~ phi_z + broker_mean_phi_z
    sub['bk_mean_z'] = sub.groupby('broker')['phi_z'].transform('mean')
    mund = smf.ols('bw ~ phi_z + bk_mean_z', data=sub).fit(cov_type='HC1')
    print(f'  Mundlak phi_z={mund.params["phi_z"]:.3f}, bk_mean_z={mund.params["bk_mean_z"]:.3f}')

    # brokerage-level mean phi vs bidding war rate
    bk_summary = sub.groupby('broker').agg(
        mean_phi=('phi', 'mean'),
        bw_rate=('bw', 'mean'),
        n=('phi', 'size'),
    ).reset_index()
    r_bk = pearsonr(bk_summary['mean_phi'], bk_summary['bw_rate'])[0]
    print(f'  r(brokerage mean phi, bw rate) = {r_bk:+.3f}')

    # most aggressive / least aggressive
    bk_summary = bk_summary.sort_values('mean_phi', ascending=False)
    print('  most aggressive brokerages:')
    for _, row in bk_summary.head(3).iterrows():
        print(f'    {row["broker"][:40]:<42s}  n={row["n"]:>3d}  mean_phi={row["mean_phi"]:+.4f}  bw={row["bw_rate"]*100:.1f}%')
    print('  least aggressive brokerages:')
    for _, row in bk_summary.tail(3).iterrows():
        print(f'    {row["broker"][:40]:<42s}  n={row["n"]:>3d}  mean_phi={row["mean_phi"]:+.4f}  bw={row["bw_rate"]*100:.1f}%')

    # case studies: top broker-switcher within-property phi differences
    df_sub['prop_key'] = df_sub['street'].astype(str) + '__' + df_sub['zipcode'].astype(str)
    case_studies = []
    for pk, grp in df_sub.groupby('prop_key'):
        if grp['broker'].nunique() < 2: continue
        phis = grp['phi'].values
        if len(phis) < 2: continue
        spread = phis.max() - phis.min()
        if spread > 0.2:
            case_studies.append({
                'prop': pk,
                'spread': float(spread),
                'phi_range': (float(phis.min()), float(phis.max())),
                'n_listings': int(len(grp)),
            })
    case_studies = sorted(case_studies, key=lambda x: -x['spread'])[:10]
    print('  top broker-switcher cases:')
    for cs in case_studies[:5]:
        print(f'    {cs["prop"][:40]}: phi {cs["phi_range"][0]:+.3f} to {cs["phi_range"][1]:+.3f}, spread={cs["spread"]:.3f}')

    np.savez('results_brokerage.npz',
             icc=icc,
             mund_phi=float(mund.params['phi_z']),
             mund_bk_mean=float(mund.params['bk_mean_z']),
             r_brokerage=r_bk,
             bk_names=bk_summary['broker'].values,
             bk_mean_phi=bk_summary['mean_phi'].values,
             bk_bw_rate=bk_summary['bw_rate'].values,
             bk_n=bk_summary['n'].values)
    return {'icc': icc, 'r_brokerage': r_bk, 'case_studies': case_studies}


# -----------------------------------------------------------------------------
# CLIP scoring: quality and per-attribute decomposition
# -----------------------------------------------------------------------------

QUALITY_PROMPTS = {
    'text':  ("a real estate listing for a modest, basic property",
              "a real estate listing for a luxurious, upscale property"),
    'photo': ("a photo of a modest, basic home interior",
              "a photo of a luxurious, high-end home interior"),
    'gsv':   ("a photo of a run-down, undesirable neighborhood street",
              "a photo of an attractive, well-maintained neighborhood street"),
    'sat':   ("an aerial view of a dense, low-income urban area",
              "an aerial view of an affluent residential area"),
}

ATTR_PROMPTS = {
    'Building condition': {
        'img': ("a photo of a poorly maintained, run-down building",
                "a photo of a well-maintained, pristine building"),
        'txt': ("a property listing for a building in poor condition",
                "a property listing for a building in excellent condition")},
    'Interior quality': {
        'img': ("a photo of a basic, dated home interior with cheap finishes",
                "a photo of a luxurious home interior with high-end finishes"),
        'txt': ("a property listing describing basic, modest interiors",
                "a property listing describing luxury, high-end interiors")},
    'Neighborhood': {
        'img': ("a photo of a run-down, undesirable urban neighborhood",
                "a photo of a beautiful, upscale residential neighborhood"),
        'txt': ("a property listing in a low-income neighborhood",
                "a property listing in an affluent, desirable neighborhood")},
    'Space and light': {
        'img': ("a photo of a small, dark, cramped room",
                "a photo of a spacious, bright, airy room full of natural light"),
        'txt': ("a property listing describing a small, compact space",
                "a property listing describing a spacious, sun-filled home")},
}


def qscore_batch(embeddings, mask, lo, hi, clip_model, device):
    import clip
    tok = clip.tokenize([lo, hi]).to(device)
    with torch.no_grad():
        tf = clip_model.encode_text(tok).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    scores = np.zeros(len(embeddings))
    idx = np.where(mask)[0]
    if not len(idx):
        return scores
    e = torch.tensor(embeddings[idx]).float().to(device)
    e = e / e.norm(dim=-1, keepdim=True)
    logits = (100.0 * e @ tf.T).softmax(dim=-1)[:, 1]
    scores[idx] = logits.cpu().numpy()
    return scores


def prompted_quality(data):
    print('\nCLIP scoring...')
    try:
        import clip
    except ImportError:
        print('  CLIP not available; skipping')
        return None

    device = DEVICE
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    print(f"  CLIP loaded on {device}")

    d = data
    emb_text   = d['emb_text']
    emb_photos = d['emb_photos']
    emb_gsv    = d['emb_gsv']
    emb_sat    = d['emb_sat']
    has_text   = d['has_text']
    has_photos = d['has_photos']
    has_gsv    = d['has_gsv']
    has_sat    = d['has_sat']

    q_text   = qscore_batch(emb_text,   has_text,   *QUALITY_PROMPTS['text'],  clip_model, device)
    q_photos = qscore_batch(emb_photos, has_photos, *QUALITY_PROMPTS['photo'], clip_model, device)
    q_gsv    = qscore_batch(emb_gsv,    has_gsv,    *QUALITY_PROMPTS['gsv'],   clip_model, device)
    q_sat    = qscore_batch(emb_sat,    has_sat,    *QUALITY_PROMPTS['sat'],   clip_model, device)

    print(f"\nquality (mean): text={q_text[has_text].mean():.3f}  "
          f"photos={q_photos[has_photos].mean():.3f}  "
          f"GSV={q_gsv[has_gsv].mean():.3f}  sat={q_sat[has_sat].mean():.3f}")

    strat_q = np.where(has_text & has_photos, 0.5 * (q_text + q_photos), np.nan)
    obj_q   = np.where(has_gsv & has_sat, 0.5 * (q_gsv + q_sat), np.nan)
    strat_q_textonly = np.where(has_text, q_text, np.nan)
    strat_q = np.where(has_photos, strat_q, strat_q_textonly)

    qm = np.where(np.isfinite(strat_q) & np.isfinite(obj_q), strat_q - obj_q, np.nan)
    valid_qm = np.isfinite(qm)
    qm_median = float(np.nanmedian(qm))
    qm_fill = qm.copy()
    qm_fill[~valid_qm] = qm_median

    print(f"\nmisalignment: mean={qm[valid_qm].mean():.3f}  "
          f"std={qm[valid_qm].std():.3f}  n={valid_qm.sum()}")
    print(f"  with photos: {(valid_qm & has_photos).sum()}")
    print(f"  text-only: {(valid_qm & ~has_photos).sum()}")

    # per-attribute decomposition
    attr_scores = {}
    print("\nper-attribute gaps:")
    for aname, prompts in ATTR_PROMPTS.items():
        ts = qscore_batch(emb_text,   has_text,   *prompts['txt'], clip_model, device)
        ps = qscore_batch(emb_photos, has_photos, *prompts['img'], clip_model, device)
        gs = qscore_batch(emb_gsv,    has_gsv,    *prompts['img'], clip_model, device)
        sa = np.where(has_text & has_photos, 0.5 * (ts + ps), np.nan)
        oa = np.where(has_gsv, gs, np.nan)
        ga = np.where(np.isfinite(sa) & np.isfinite(oa), sa - oa, np.nan)
        attr_scores[aname] = {'S': sa, 'O': oa, 'gap': ga}
        v = np.isfinite(ga)
        print(f"  {aname:<22s} S={sa[v].mean():.3f}  O={oa[v].mean():.3f}  gap={ga[v].mean():+.3f}")

    pct_above = float((qm[valid_qm] > 0).mean())

    # for figures, store as plain arrays (replace NaN with median for visualization)
    np.savez('results_quality.npz',
             q_text=q_text, q_photo=q_photos, q_gsv=q_gsv, q_sat=q_sat,
             q_strat=np.nan_to_num(strat_q, nan=np.nanmedian(strat_q)),
             q_obj=np.nan_to_num(obj_q, nan=np.nanmedian(obj_q)),
             delta_i=qm_fill,
             pct_above=pct_above,
             attr_keys=list(attr_scores.keys()),
             attr_vals=[float(np.nanmean(attr_scores[k]['gap'])) for k in attr_scores])
    return {'qm': qm, 'valid_qm': valid_qm, 'qm_fill': qm_fill,
            'attr_scores': attr_scores, 'pct_above': pct_above}


def bidding_war_quartiles(data):
    print('\nbidding war by misalignment quartile...')
    try:
        delta = np.load('results_quality.npz')['delta_i']
    except FileNotFoundError:
        print('  prompted scores not available; skipping')
        return None

    df_all = data['df']
    valid_pg = data['valid_pg']
    bw = (df_all['price'] > df_all['price_listed']).astype(float).values
    mask = valid_pg & ~np.isnan(delta)
    sub_delta = delta[mask]
    sub_bw = bw[mask]

    qs = np.quantile(sub_delta, [0.25, 0.5, 0.75])
    bins = np.digitize(sub_delta, qs)
    rates = [float(sub_bw[bins == k].mean()) for k in range(4)]
    counts = [int((bins == k).sum()) for k in range(4)]
    print(f'  Q1 (aligned): {rates[0]*100:.1f}%   Q4 (misaligned): {rates[3]*100:.1f}%')

    df_lr = pd.DataFrame({'bw': sub_bw, 'd': sub_delta})
    lr = smf.logit('bw ~ d', data=df_lr).fit(disp=0)
    odds_ratio = float(np.exp(lr.params['d']))
    print(f'  logistic odds ratio: {odds_ratio:.3f}, p={lr.pvalues["d"]:.3g}')

    np.savez('results_bw_quartiles.npz',
             rates=rates, counts=counts, odds_ratio=odds_ratio,
             logit_p=float(lr.pvalues['d']))
    return {'rates': rates, 'counts': counts, 'odds_ratio': odds_ratio}


# -----------------------------------------------------------------------------
# architecture comparison
# -----------------------------------------------------------------------------

MASK_PROB = 0.15
EPOCHS = 200
PATIENCE = 20
BATCH = 64


def safe_heads(hidden, n_heads):
    while hidden % n_heads != 0 and n_heads > 1:
        n_heads //= 2
    return max(n_heads, 1)


class GatedFusion(nn.Module):
    def __init__(self, d_tab, d_mod, n_mods=4, hidden=64, dropout=0.2):
        super().__init__()
        self.mod_enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_mod, hidden), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_mods)])
        self.tab_enc = nn.Sequential(nn.Linear(d_tab, hidden), nn.GELU(), nn.Dropout(dropout))
        # vector gate: per-dimension weighting, not just scalar per modality
        self.gate = nn.Sequential(
            nn.Linear(d_tab + n_mods * d_mod, n_mods * hidden), nn.Sigmoid())
        self.n_mods = n_mods
        self.hidden = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x_tab, x_mods):
        tab_h = self.tab_enc(x_tab)
        mod_h = torch.stack([enc(x) for enc, x in zip(self.mod_enc, x_mods)], dim=1)
        gate_input = torch.cat([x_tab] + x_mods, dim=1)
        gates = self.gate(gate_input).view(-1, self.n_mods, self.hidden)
        fused = (mod_h * gates).mean(dim=1)
        return self.head(torch.cat([tab_h, fused], dim=1)).squeeze(-1)


class CrossModalAttention(nn.Module):
    def __init__(self, d_tab, d_mod, n_heads=4, hidden=64, dropout=0.2):
        super().__init__()
        n_heads = safe_heads(hidden, n_heads)
        self.n_heads, self.d_k = n_heads, hidden // n_heads
        self.proj = nn.ModuleDict({k: nn.Linear(d_mod * 2, hidden)
            for k in ['q_s', 'k_o', 'v_o', 'q_o', 'k_s', 'v_s']})
        self.tab_enc = nn.Sequential(nn.Linear(d_tab, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def attend(self, Q, K, V):
        b = Q.size(0)
        Q = Q.view(b, self.n_heads, self.d_k)
        K = K.view(b, self.n_heads, self.d_k)
        V = V.view(b, self.n_heads, self.d_k)
        scores = torch.einsum('bnd,bnd->bn', Q, K) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum('bn,bnd->bd', attn, V).reshape(b, -1)

    def forward(self, x_tab, x_strat, x_obj):
        tab_h = self.tab_enc(x_tab)
        s2o = self.attend(self.proj['q_s'](x_strat), self.proj['k_o'](x_obj), self.proj['v_o'](x_obj))
        o2s = self.attend(self.proj['q_o'](x_obj), self.proj['k_s'](x_strat), self.proj['v_s'](x_strat))
        return self.head(torch.cat([tab_h, s2o, o2s], dim=1)).squeeze(-1)


class BilinearFusion(nn.Module):
    def __init__(self, d_tab, d_strat, d_obj, rank=16, hidden=64, dropout=0.2):
        super().__init__()
        self.U = nn.Linear(d_strat, rank, bias=False)
        self.V = nn.Linear(d_obj, rank, bias=False)
        self.tab_enc = nn.Sequential(nn.Linear(d_tab, hidden), nn.GELU(), nn.Dropout(dropout))
        self.strat_enc = nn.Sequential(nn.Linear(d_strat, hidden), nn.GELU(), nn.Dropout(dropout))
        self.obj_enc = nn.Sequential(nn.Linear(d_obj, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden*3+rank, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_tab, x_strat, x_obj):
        bilinear = self.U(x_strat) * self.V(x_obj)
        return self.head(torch.cat([
            self.tab_enc(x_tab), self.strat_enc(x_strat),
            self.obj_enc(x_obj), bilinear], dim=1)).squeeze(-1)


class AugmentedTensorFusion(nn.Module):
    """Zadeh et al 2017: append 1 before outer product to capture main + interaction effects."""
    def __init__(self, d_tab, d_strat, d_obj, proj_dim=16, hidden=64, dropout=0.2):
        super().__init__()
        self.proj_s = nn.Linear(d_strat, proj_dim)
        self.proj_o = nn.Linear(d_obj, proj_dim)
        self.tab_enc = nn.Sequential(nn.Linear(d_tab, hidden), nn.GELU(), nn.Dropout(dropout))
        # augmented: (proj_dim+1)^2 from outer product of [proj; 1] vectors
        aug_dim = (proj_dim + 1) ** 2
        self.head = nn.Sequential(nn.Linear(hidden + aug_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.proj_dim = proj_dim

    def forward(self, x_tab, x_strat, x_obj):
        s = torch.relu(self.proj_s(x_strat))
        o = torch.relu(self.proj_o(x_obj))
        # append 1 for augmented outer product (captures unimodal + bimodal + bias)
        ones = torch.ones(s.size(0), 1, device=s.device)
        s_aug = torch.cat([s, ones], dim=1)
        o_aug = torch.cat([o, ones], dim=1)
        outer = torch.einsum('bi,bj->bij', s_aug, o_aug).reshape(s.size(0), -1)
        tab_h = self.tab_enc(x_tab)
        return self.head(torch.cat([tab_h, outer], dim=1)).squeeze(-1)


class MulTInspired(nn.Module):
    """Transformer over modality tokens (tab + 4 CLIP modalities).
    Uses self-attention so every modality can attend to every other.
    Inspired by Tsai et al 2019 but simplified: they use pairwise
    directional streams, we use shared self-attention with residuals.
    Stores attention weights when extract_attn=True."""
    def __init__(self, d_tab, d_mod, n_mods=4, hidden=64, n_layers=2, dropout=0.2, n_heads=4):
        super().__init__()
        n_heads = safe_heads(hidden, n_heads)
        self.projections = nn.ModuleList([nn.Linear(d_mod, hidden) for _ in range(n_mods)])
        self.tab_proj = nn.Linear(d_tab, hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden, num_heads=n_heads,
                    dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(hidden),
                'ff': nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden*2, hidden)),
                'norm2': nn.LayerNorm(hidden),
            }))
        self.head = nn.Sequential(nn.Linear(hidden*(n_mods+1), hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.extract_attn = False
        self.last_attn_weights = None

    def forward(self, x_tab, x_mods):
        tokens = [self.tab_proj(x_tab)]
        for proj, x in zip(self.projections, x_mods):
            tokens.append(proj(x))
        seq = torch.stack(tokens, dim=1)
        for layer in self.layers:
            if self.extract_attn:
                attended, attn_w = layer['attn'](seq, seq, seq, need_weights=True)
                self.last_attn_weights = attn_w.detach()
            else:
                attended, _ = layer['attn'](seq, seq, seq)
            seq = layer['norm1'](seq + attended)
            seq = layer['norm2'](seq + layer['ff'](seq))
        return self.head(seq.reshape(seq.size(0), -1)).squeeze(-1)


class MMoE(nn.Module):
    """Mixture of experts with per-sample gating. Stores gate weights for analysis."""
    def __init__(self, d_in, n_experts=4, hidden=32, dropout=0.2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
            for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(d_in, n_experts), nn.Softmax(dim=-1))
        self.last_gate_weights = None

    def forward(self, x):
        outs = torch.stack([e(x).squeeze(-1) for e in self.experts], dim=1)
        gw = self.gate(x)
        self.last_gate_weights = gw.detach()
        return (outs * gw).sum(dim=1)


class DeepSetsFusion(nn.Module):
    """Shared encoder across modalities, mean pool within groups, then cross-group interaction."""
    def __init__(self, d_tab, d_mod, hidden=64, dropout=0.2):
        super().__init__()
        self.shared_enc = nn.Sequential(nn.Linear(d_mod, hidden), nn.GELU(), nn.Dropout(dropout))
        self.tab_enc = nn.Sequential(nn.Linear(d_tab, hidden), nn.GELU(), nn.Dropout(dropout))
        # bilinear cross-group interaction
        self.U = nn.Linear(hidden, hidden // 2, bias=False)
        self.V = nn.Linear(hidden, hidden // 2, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden + hidden*2 + hidden//2, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_tab, x_mods):
        # encode all 4 modalities with shared weights
        encoded = [self.shared_enc(x) for x in x_mods]
        # mean pool within strategic (text, photos) and objective (gsv, sat) groups
        strat_pool = (encoded[0] + encoded[1]) / 2
        obj_pool = (encoded[2] + encoded[3]) / 2
        cross = self.U(strat_pool) * self.V(obj_pool)
        tab_h = self.tab_enc(x_tab)
        return self.head(torch.cat([tab_h, strat_pool, obj_pool, cross], dim=1)).squeeze(-1)


def train_eval(model, X_tr, y_tr, X_te, y_te, input_type,
               lr=1e-3, wd=1e-4, mask_prob=0.0):
    """Train and evaluate one fold. Returns R2 or nan on failure."""
    try:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.MSELoss()
        n_tr = len(X_tr[0]) if isinstance(X_tr, list) else len(X_tr)
        best_loss, best_state, wait = float('inf'), None, 0

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(n_tr)
            for start in range(0, n_tr, BATCH):
                idx = perm[start:start+BATCH]
                if input_type == '4mod':
                    inputs = [torch.tensor(X_tr[k][idx], dtype=torch.float32).to(DEVICE) for k in range(5)]
                    # modality masking: zero out each modality independently
                    if mask_prob > 0:
                        for k in range(1, 5):  # skip tabular
                            mask = (torch.rand(len(idx), 1, device=DEVICE) > mask_prob).float()
                            inputs[k] = inputs[k] * mask
                    pred = model(inputs[0], inputs[1:])
                elif input_type == '2grp':
                    t = torch.tensor(X_tr[0][idx], dtype=torch.float32).to(DEVICE)
                    s = torch.tensor(X_tr[1][idx], dtype=torch.float32).to(DEVICE)
                    o = torch.tensor(X_tr[2][idx], dtype=torch.float32).to(DEVICE)
                    if mask_prob > 0:
                        s = s * (torch.rand(len(idx), 1, device=DEVICE) > mask_prob).float()
                        o = o * (torch.rand(len(idx), 1, device=DEVICE) > mask_prob).float()
                    pred = model(t, s, o)
                else:
                    x = torch.tensor(X_tr[idx], dtype=torch.float32).to(DEVICE)
                    pred = model(x)

                y_b = torch.tensor(y_tr[idx], dtype=torch.float32).to(DEVICE)
                loss = loss_fn(pred, y_b)
                if torch.isnan(loss): return float('nan')
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            # val
            model.eval()
            with torch.no_grad():
                if input_type == '4mod':
                    inputs = [torch.tensor(X_te[k], dtype=torch.float32).to(DEVICE) for k in range(5)]
                    vp = model(inputs[0], inputs[1:])
                elif input_type == '2grp':
                    vp = model(torch.tensor(X_te[0], dtype=torch.float32).to(DEVICE),
                               torch.tensor(X_te[1], dtype=torch.float32).to(DEVICE),
                               torch.tensor(X_te[2], dtype=torch.float32).to(DEVICE))
                else:
                    vp = model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE))
                if torch.isnan(vp).any(): return float('nan')
                vl = loss_fn(vp, torch.tensor(y_te, dtype=torch.float32).to(DEVICE)).item()
            if np.isnan(vl): return float('nan')
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE: break

        if best_state is None: return float('nan')
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if input_type == '4mod':
                inputs = [torch.tensor(X_te[k], dtype=torch.float32).to(DEVICE) for k in range(5)]
                p = model(inputs[0], inputs[1:])
            elif input_type == '2grp':
                p = model(torch.tensor(X_te[0], dtype=torch.float32).to(DEVICE),
                           torch.tensor(X_te[1], dtype=torch.float32).to(DEVICE),
                           torch.tensor(X_te[2], dtype=torch.float32).to(DEVICE))
            else:
                p = model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE))
            return r2_score(y_te, p.cpu().numpy())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float('nan')
    except Exception:
        return float('nan')


def architecture_comparison(data, target='log_price'):
    print(f'\narchitecture comparison ({target})...')
    d = data
    mask = d['complete'] & d['valid_pg']
    idx = np.where(mask)[0]

    X_tab = d['TAB_s'][idx]
    X_text = d['pca_text'][idx]
    X_photo = d['pca_photos'][idx]
    X_gsv = d['pca_gsv'][idx]
    X_sat = d['pca_sat'][idx]
    X_strat = np.hstack([X_text, X_photo])
    X_obj = np.hstack([X_gsv, X_sat])
    X_all = np.hstack([X_tab, X_text, X_photo, X_gsv, X_sat])

    d_tab = X_tab.shape[1]
    d_mod = X_text.shape[1]
    d_strat = X_strat.shape[1]
    d_obj = X_obj.shape[1]
    d_all = X_all.shape[1]

    if target == 'log_price':
        y = d['log_price'][idx].astype(np.float64)
    else:
        y = d['df'].loc[idx, 'price_gap'].values.astype(np.float64)

    data_4mod = [X_tab, X_text, X_photo, X_gsv, X_sat]
    data_2grp = [X_tab, X_strat, X_obj]

    def cv_r2_ridge(X, y):
        scores = []
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        for tr, te in kf.split(X):
            m = Ridge(alpha=1.0).fit(X[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X[te])))
        return np.mean(scores), np.std(scores)

    ridge_r2, ridge_std = cv_r2_ridge(X_all, y)
    print(f"  {'Ridge baseline':<24s}  R2={ridge_r2:.4f} +/- {ridge_std:.4f}")

    mlp_scores = []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_all):
        ml = MLPRegressor(hidden_layer_sizes=(64,), alpha=0.1, max_iter=2000,
            early_stopping=True, validation_fraction=0.15, random_state=SEED,
            learning_rate='adaptive')
        ml.fit(X_all[tr], y[tr])
        mlp_scores.append(r2_score(y[te], ml.predict(X_all[te])))
    print(f"  {'sklearn MLP(64)':<24s}  R2={np.mean(mlp_scores):.4f} +/- {np.std(mlp_scores):.4f}")

    archs = [
        ('Gated fusion',     lambda: GatedFusion(d_tab, d_mod, 4, 64, 0.2),                   data_4mod, '4mod'),
        ('Cross-modal attn', lambda: CrossModalAttention(d_tab, d_mod, 4, 64, 0.2),           data_2grp, '2grp'),
        ('Bilinear fusion',  lambda: BilinearFusion(d_tab, d_strat, d_obj, 16, 64, 0.2),      data_2grp, '2grp'),
        ('Augmented tensor', lambda: AugmentedTensorFusion(d_tab, d_strat, d_obj, 16, 64, 0.2), data_2grp, '2grp'),
        ('MulT cross-modal', lambda: MulTInspired(d_tab, d_mod, 4, 64, 2, 0.2, 4),            data_4mod, '4mod'),
        ('MMoE (6 experts)', lambda: MMoE(d_all, 6, 48, 0.2),                                 X_all,     'flat'),
        ('DeepSets fusion',  lambda: DeepSetsFusion(d_tab, d_mod, 64, 0.2),                   data_4mod, '4mod'),
    ]

    def run_cv(name, make_fn, dat, input_type, mask_prob=0.0):
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
        scores = []
        for fold, (tr, te) in enumerate(kf.split(y)):
            model = make_fn().to(DEVICE)
            if input_type in ('4mod', '2grp'):
                X_tr = [arr[tr] for arr in dat]
                X_te = [arr[te] for arr in dat]
            else:
                X_tr, X_te = dat[tr], dat[te]
            s = train_eval(model, X_tr, y[tr], X_te, y[te], input_type, mask_prob=mask_prob)
            scores.append(s)
        valid = [s for s in scores if not np.isnan(s)]
        r2 = np.mean(valid) if valid else float('nan')
        std = np.std(valid) if valid else 0
        n_params = sum(p.numel() for p in make_fn().parameters())
        print(f"  {name:<24s} {n_params:>8,d}p  R2={r2:.4f} +/- {std:.4f}")
        return {'name': name, 'r2': r2, 'std': std, 'params': n_params}

    arch_results = []
    for name, make_fn, dat, itype in archs:
        arch_results.append(run_cv(name, make_fn, dat, itype))

    print(f'\nwith modality masking (p={MASK_PROB}):')
    arch_results_masked = []
    for name, make_fn, dat, itype in archs:
        if itype == 'flat': continue
        arch_results_masked.append(run_cv(f'{name} +mask', make_fn, dat, itype, mask_prob=MASK_PROB))

    all_arch = [{'name': 'Ridge', 'r2': ridge_r2, 'std': ridge_std, 'params': d_all + 1}]
    all_arch += arch_results + arch_results_masked
    all_arch.sort(key=lambda x: x['r2'] if not np.isnan(x['r2']) else -999, reverse=True)

    np.savez(f'results_archs_{target}.npz',
             names=np.array([a['name'] for a in all_arch], dtype=object),
             r2_means=np.array([a['r2'] for a in all_arch]),
             r2_ses=np.array([a['std'] for a in all_arch]),
             params=np.array([a['params'] for a in all_arch]),
             ridge_r2=ridge_r2, ridge_std=ridge_std,
             mlp_r2=float(np.mean(mlp_scores)), mlp_std=float(np.std(mlp_scores)))
    return all_arch


# -----------------------------------------------------------------------------
# cross-modal attention and gradient importance
# -----------------------------------------------------------------------------

def attention_and_gradients(data, headline):
    print('\ncross-modal attention and gradient importance...')
    d = data
    idx = headline['idx']
    phi = headline['phi_S']

    X_tab = d['TAB_s'][idx]
    X_text = d['pca_text'][idx]
    X_photo = d['pca_photos'][idx]
    X_gsv = d['pca_gsv'][idx]
    X_sat = d['pca_sat'][idx]
    d_tab = X_tab.shape[1]
    d_mod = X_text.shape[1]
    mod_names = ['Tab', 'Text', 'Photos', 'GSV', 'Sat']

    y = d['log_price'][idx].astype(np.float64)
    data_4mod = [X_tab, X_text, X_photo, X_gsv, X_sat]

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(perm))
    tr, te = perm[:cut], perm[cut:]

    model = MulTInspired(d_tab, d_mod, 4, 64, 2, 0.2, 4).to(DEVICE)
    X_tr = [arr[tr] for arr in data_4mod]
    X_te = [arr[te] for arr in data_4mod]
    _ = train_eval(model, X_tr, y[tr], X_te, y[te], '4mod')

    q = np.quantile(phi, [0.25, 0.5, 0.75])
    bins = np.digitize(phi, q)

    def avg_attn(rows_mask):
        rows = np.where(rows_mask)[0][:1000]
        if len(rows) == 0:
            return np.zeros((5, 5))
        model.extract_attn = True
        with torch.no_grad():
            x_tab_t = torch.tensor(X_tab[rows], dtype=torch.float32, device=DEVICE)
            x_mods = [torch.tensor(arr[rows], dtype=torch.float32, device=DEVICE)
                      for arr in [X_text, X_photo, X_gsv, X_sat]]
            _ = model(x_tab_t, x_mods)
        model.extract_attn = False
        # attention weights have shape (batch, n_tokens, n_tokens) = (B, 5, 5)
        return model.last_attn_weights.mean(0).cpu().numpy()

    aligned = bins == 0
    misaligned = bins == 3
    attn_aligned = avg_attn(aligned)
    attn_misaligned = avg_attn(misaligned)

    print('  gradient-based importance...')
    grad_norms = np.zeros((4, 5))
    for q_idx in range(4):
        rows = np.where(bins == q_idx)[0]
        if len(rows) == 0: continue
        sample = rows[:500] if len(rows) > 500 else rows
        for j in range(5):
            x_tab_t = torch.tensor(X_tab[sample], dtype=torch.float32, device=DEVICE)
            x_mods_t = [torch.tensor(arr[sample], dtype=torch.float32, device=DEVICE)
                        for arr in [X_text, X_photo, X_gsv, X_sat]]
            if j == 0:
                x_tab_t = x_tab_t.clone().requires_grad_(True)
            else:
                x_mods_t[j-1] = x_mods_t[j-1].clone().requires_grad_(True)
            model.eval()
            pred = model(x_tab_t, x_mods_t).sum()
            pred.backward()
            target = x_tab_t if j == 0 else x_mods_t[j-1]
            grad_norms[q_idx, j] = target.grad.norm(dim=1).mean().item()

    np.savez('results_attention.npz',
             mod_names=mod_names,
             attn_aligned=attn_aligned, attn_misaligned=attn_misaligned,
             grad_norms=grad_norms)
    return {'attn_aligned': attn_aligned, 'attn_misaligned': attn_misaligned,
            'grad_norms': grad_norms}


# -----------------------------------------------------------------------------
# modality value, CCA, interaction types
# -----------------------------------------------------------------------------

def modality_value_and_cca(data, headline):
    print('\nmodality value over tabular baseline + CCA spectrum...')
    d = data
    idx = headline['idx']
    phi = headline['phi_S']

    Xtab = d['TAB_s'][idx]
    Xt = d['pca_text'][idx]
    Xp = d['pca_photos'][idx]
    Xg = d['pca_gsv'][idx]
    Xs = d['pca_sat'][idx]
    y = d['log_price'][idx].astype(np.float64)

    q1 = np.quantile(phi, 0.25)
    q4 = np.quantile(phi, 0.75)
    aligned = phi <= q1
    misaligned = phi >= q4

    def cv_r2(X, y, splits):
        rs = []
        for tr, te in splits:
            m = Ridge(alpha=1.0).fit(X[tr], y[tr])
            rs.append(r2_score(y[te], m.predict(X[te])))
        return np.mean(rs)

    def class_increment(mask, name):
        Xt_c = Xtab[mask]; y_c = y[mask]
        splits = list(KFold(5, shuffle=True, random_state=SEED).split(Xt_c))
        tab_only = cv_r2(Xt_c, y_c, splits)
        out = {}
        for nm, X in [('Text', Xt[mask]), ('Photos', Xp[mask]), ('GSV', Xg[mask]), ('Sat', Xs[mask])]:
            X_aug = np.hstack([Xt_c, X])
            out[nm] = cv_r2(X_aug, y_c, splits) - tab_only
        print(f'  {name} increments over tabular: {", ".join(f"{k}={v:+.3f}" for k, v in out.items())}')
        return out

    inc_aligned = class_increment(aligned, 'aligned')
    inc_misaligned = class_increment(misaligned, 'misaligned')

    print('  CCA spectrum...')
    X_strat = np.hstack([Xt, Xp])
    X_obj   = np.hstack([Xg, Xs])
    n_comp = min(20, min(X_strat.shape[1], X_obj.shape[1]))
    cca = CCA(n_components=n_comp, max_iter=500)
    cca.fit(X_strat, X_obj)
    X_c, Y_c = cca.transform(X_strat, X_obj)
    corrs = np.array([pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(n_comp)])
    print(f'  top canonical correlations: {corrs[:5].round(3)}')

    private_strat_norm = np.linalg.norm(X_c[:, 5:], axis=1)
    r_private_phi = pearsonr(private_strat_norm, phi)[0]
    print(f'  r(private strat L2, phi) = {r_private_phi:+.3f}')

    np.savez('results_modality_value.npz',
             mod_names=['Text', 'Photos', 'GSV', 'Sat'],
             inc_aligned=np.array([inc_aligned[k] for k in ['Text', 'Photos', 'GSV', 'Sat']]),
             inc_misaligned=np.array([inc_misaligned[k] for k in ['Text', 'Photos', 'GSV', 'Sat']]),
             cca_corrs=corrs,
             r_private_phi=r_private_phi)
    return {'inc_aligned': inc_aligned, 'inc_misaligned': inc_misaligned,
            'cca_corrs': corrs, 'r_private_phi': r_private_phi}


def interaction_types(data, headline):
    print('\ninteraction type assignment...')
    d = data
    idx = headline['idx']
    phi = headline['phi_S']
    y_pg = headline['y_pg']

    Xt = d['pca_text'][idx]
    Xp = d['pca_photos'][idx]
    Xg = d['pca_gsv'][idx]
    Xs = d['pca_sat'][idx]
    X_S = np.hstack([Xt, Xp])
    X_O = np.hstack([Xg, Xs])

    kf = KFold(5, shuffle=True, random_state=SEED)
    pred_S = np.zeros(len(y_pg)); pred_O = np.zeros(len(y_pg)); pred_J = np.zeros(len(y_pg))
    for tr, te in kf.split(X_S):
        for X, p in [(X_S, pred_S), (X_O, pred_O), (np.hstack([X_S, X_O]), pred_J)]:
            m = Ridge(alpha=1.0).fit(X[tr], y_pg[tr])
            p[te] = m.predict(X[te])

    eS = (y_pg - pred_S) ** 2
    eO = (y_pg - pred_O) ** 2
    eJ = (y_pg - pred_J) ** 2
    eN = (y_pg - y_pg.mean()) ** 2
    types = np.full(len(y_pg), 'Neither', dtype=object)
    for i in range(len(y_pg)):
        eb = min(eS[i], eO[i], eJ[i], eN[i])
        if eb == eN[i]:
            types[i] = 'Neither'
        elif eJ[i] < min(eS[i], eO[i]) - 1e-6:
            types[i] = 'Synergy'
        elif abs(eS[i] - eO[i]) < 0.1 * eN[i]:
            types[i] = 'Redundancy'
        elif eS[i] < eO[i]:
            types[i] = 'Unique-S'
        else:
            types[i] = 'Unique-O'

    from collections import Counter
    dist = Counter(types)
    print(f'  type distribution: {dict(dist)}')

    phi_by_type = {t: float(phi[types == t].mean()) for t in dist}

    np.savez('results_interaction_types.npz',
             types=np.array(types, dtype=object),
             type_counts=np.array([dist[k] for k in dist]),
             type_names=np.array(list(dist.keys())),
             phi_by_type_keys=list(phi_by_type.keys()),
             phi_by_type_vals=list(phi_by_type.values()))
    return {'types': types, 'dist': dict(dist), 'phi_by_type': phi_by_type}


# -----------------------------------------------------------------------------
# no-photo negative control
# -----------------------------------------------------------------------------

def no_photo_negative_control(data):
    print('\nno-photo negative control...')
    d = data
    df_all = d['df']
    valid_pg = d['valid_pg']
    has_photos = d['has_photos']

    no_p = ~has_photos & valid_pg
    has_p = has_photos & valid_pg

    bw = (df_all['price'] > df_all['price_listed']).astype(float).values
    np_bw = bw[no_p].mean()
    p_bw  = bw[has_p].mean()

    np_dom = df_all.loc[no_p, 'days_on_market'].median()
    p_dom  = df_all.loc[has_p, 'days_on_market'].median()
    np_pg  = df_all.loc[no_p, 'price_gap'].mean()
    p_pg   = df_all.loc[has_p, 'price_gap'].mean()

    print(f'  no-photo n={no_p.sum()}: BW={np_bw*100:.1f}%, days={np_dom:.0f}, gap={np_pg:+.3f}')
    print(f'  has-photo n={has_p.sum()}: BW={p_bw*100:.1f}%, days={p_dom:.0f}, gap={p_pg:+.3f}')

    np.savez('results_nophoto.npz',
             np_bw=np_bw, p_bw=p_bw, np_dom=np_dom, p_dom=p_dom,
             np_pg=np_pg, p_pg=p_pg,
             np_n=int(no_p.sum()), p_n=int(has_p.sum()),
             np_gap_dist=df_all.loc[no_p, 'price_gap'].values,
             p_gap_dist=df_all.loc[has_p, 'price_gap'].values)
    return {'np_bw': np_bw, 'p_bw': p_bw}


# -----------------------------------------------------------------------------
# failed alternatives: value-supervised scoring methods
# -----------------------------------------------------------------------------

def failed_alternatives(data, headline):
    print('\nfailed alternatives (value-supervised methods)...')
    d = data
    idx = headline['idx']
    X_S = headline['X_S']
    X_O = headline['X_O']
    y_pg = headline['y_pg']
    y_lp = headline['y_lp']

    df_sub = d['df'].loc[idx].reset_index(drop=True)
    log_ppsf = np.log(df_sub['price'] / df_sub['area_sqft'].clip(lower=100))
    Z = d['TAB_s'][idx]
    resid = log_ppsf.values - LinearRegression().fit(Z, log_ppsf.values).predict(Z)
    score_ridge, _ = ridge_oof(X_S, resid)
    r_ridge = pearsonr(score_ridge, y_pg)[0]
    print(f'  Ridge on residualized log_ppsf: r={r_ridge:+.3f}')

    print('  MLP on log price...')
    kf = KFold(5, shuffle=True, random_state=SEED)
    score_mlp = np.zeros_like(y_lp)
    for tr, te in kf.split(X_S):
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED).fit(X_S[tr], y_lp[tr])
        score_mlp[te] = m.predict(X_S[te])
    r_mlp = pearsonr(score_mlp - y_lp, y_pg)[0]
    print(f'  MLP residual on log_price: r={r_mlp:+.3f}')

    print('  contrastive on log_price quartile...')
    bins = np.digitize(y_lp, np.quantile(y_lp, [0.25, 0.5, 0.75]))
    score_con = np.zeros_like(y_lp)
    for tr, te in kf.split(X_S):
        m = Ridge(alpha=1.0).fit(np.hstack([X_S[tr], X_O[tr]]), bins[tr].astype(float))
        score_con[te] = m.predict(np.hstack([X_S[te], X_O[te]]))
    r_con = pearsonr(score_con, y_pg)[0]
    print(f'  Contrastive surrogate: r={r_con:+.3f}')

    try:
        delta = np.load('results_quality.npz')['delta_i']
        df_all = d['df']
        valid_pg = d['valid_pg']
        complete = d['complete']
        m = complete & valid_pg
        delta_idx = delta[m]
        r_prompted = pearsonr(delta_idx, y_pg)[0]
        print(f'  Hand-crafted prompted: r={r_prompted:+.3f}')
    except (FileNotFoundError, KeyError):
        r_prompted = np.nan

    np.savez('results_failed_alts.npz',
             r_ridge=r_ridge, r_mlp=r_mlp, r_con=r_con, r_prompted=r_prompted)
    return {'r_ridge': r_ridge, 'r_mlp': r_mlp, 'r_con': r_con, 'r_prompted': r_prompted}


def raw_cosine_distributions(data):
    print('\nraw CLIP cosine distributions...')
    d = data
    valid = d['complete']
    et = d['emb_text'][valid]
    ep = d['emb_photos'][valid]
    eg = d['emb_gsv'][valid]
    es = d['emb_sat'][valid]
    def norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    et = norm(et); ep = norm(ep); eg = norm(eg); es = norm(es)
    cos_tg = (et * eg).sum(1)
    cos_ts = (et * es).sum(1)
    cos_pg = (ep * eg).sum(1)
    cos_ps = (ep * es).sum(1)
    cos_tp = (et * ep).sum(1)
    cos_gs = (eg * es).sum(1)

    M = 1 - (cos_tg + cos_ts + cos_pg + cos_ps) / 4

    valid_pg = d['valid_pg']
    df = d['df']
    M_full = np.full(len(df), np.nan)
    M_full[valid] = M
    pg = df['price_gap'].values
    msk = valid & valid_pg
    r_raw, p_raw = pearsonr(M_full[msk], pg[msk])
    print(f'  r(raw misalignment, gap) = {r_raw:+.3f} (p={p_raw:.3g})')

    np.savez('results_raw_cosine.npz',
             cos_tg=cos_tg, cos_ts=cos_ts, cos_pg=cos_pg, cos_ps=cos_ps,
             cos_tp=cos_tp, cos_gs=cos_gs,
             M_aggregate=M, r_raw=r_raw, p_raw=p_raw)
    return {'r_raw': r_raw}


def spatial_distribution(data, headline):
    print('\nspatial distribution arrays...')
    d = data
    idx = headline['idx']
    df_sub = d['df'].loc[idx].reset_index(drop=True)
    phi = headline['phi_S']
    np.savez('results_spatial.npz',
             lat=df_sub['lat'].values,
             lon=df_sub['lon'].values,
             phi=phi,
             gap=df_sub['price_gap'].values,
             city=df_sub['city_group'].values)
    print(f'  saved {len(idx)} listings with coords')


def data_overview(data):
    d = data
    df = d['df']
    print('\ndata overview...')
    np.savez('results_overview.npz',
             prices=df['price'].values,
             days=df['days_on_market'].values,
             types=df['home_type'].values,
             cities=df['city_group'].values)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    data = load_data()

    data_overview(data)
    fusion_comparison(data)
    headline = headline_atom(data)
    pca_upper_bound(headline)
    causal_battery(data, headline)
    cross_modal_synergy(data, headline)
    peer_effect_iv(data, headline)
    regime_moderation(data, headline)
    property_type_hetero(data, headline)
    concept_probe(headline)
    negative_results(data, headline)
    synthetic_calibration()
    ksg_validation(data, headline)
    fold_stability(headline)
    per_city_atoms(data, headline)
    brokerage_analysis(data, headline)
    prompted_quality(data)
    bidding_war_quartiles(data)
    architecture_comparison(data, target='log_price')
    architecture_comparison(data, target='price_gap')
    attention_and_gradients(data, headline)
    modality_value_and_cca(data, headline)
    interaction_types(data, headline)
    no_photo_negative_control(data)
    failed_alternatives(data, headline)
    raw_cosine_distributions(data)
    spatial_distribution(data, headline)
    write_summary()

    elapsed = (time.time() - t0) / 60
    print(f'\nall done in {elapsed:.1f} min')


def _load_npz_dict(path):
    if not os.path.exists(path): return {}
    try:
        f = np.load(path, allow_pickle=True)
        return {k: f[k].item() if f[k].shape == () else f[k] for k in f.files}
    except Exception:
        return {}


def write_summary():
    """Pull headline scalars into a single readable JSON."""
    s = {}

    h = _load_npz_dict('results_headline.npz')
    if h:
        s['headline'] = {
            'cmi_adv_gap':        float(h.get('cmi_adv_gap', np.nan)),
            'cmi_adv_log_price':  float(h.get('cmi_adv_lp', np.nan)),
            'us_standard_gap':    float(h.get('us_standard_gap', np.nan)),
            'z_vs_null':          float(h.get('z', np.nan)),
            'ci_95_low':          float(h.get('ci_lo', np.nan)),
            'ci_95_high':         float(h.get('ci_hi', np.nan)),
            'cmi_oos':            float(h.get('cmi_oos', np.nan)),
            'seed_std':           float(h.get('seed_std', np.nan)),
        }

    p = _load_npz_dict('results_pca_bound.npz')
    if p:
        s['pca_bound'] = {
            'supervised_1d_cmi':  float(p.get('sup_cmi', np.nan)),
            'pca_64d_bound':      float(p.get('bound_64', np.nan)),
            'variance_share':     float(p.get('var_share', np.nan)),
        }

    syn = _load_npz_dict('results_synergy.npz')
    if syn:
        s['synergy'] = {
            'redundancy':         float(syn.get('R', np.nan)),
            'unique_text':        float(syn.get('U_T', np.nan)),
            'unique_photo':       float(syn.get('U_P', np.nan)),
            'synergy_gap':        float(syn.get('S', np.nan)),
            'z_vs_null':          float(syn.get('z_S', np.nan)),
            'conditional_synergy':float(syn.get('S_cond', np.nan)),
        }

    peer = _load_npz_dict('results_peer.npz')
    if peer:
        s['peer_effects'] = {
            'r_5nn':    float(peer.get('r_peer', np.nan)),
            'ols_phi':  float(peer.get('ols_phi', np.nan)),
            'ols_peer': float(peer.get('ols_peer', np.nan)),
            'iv_phi':   float(peer.get('iv_phi', np.nan)),
            'iv_peer':  float(peer.get('iv_peer', np.nan)),
            'fs_t':     float(peer.get('fs_t', np.nan)),
        }

    syn_calib = _load_npz_dict('results_synth.npz')
    if syn_calib:
        s['synthetic_calibration_mean_rel_err'] = float(syn_calib.get('mean_rel_err', np.nan))

    ksg = _load_npz_dict('results_ksg.npz')
    if ksg:
        s['ksg_validation'] = {
            'gauss_gap': float(ksg.get('gauss_gap', np.nan)),
            'ksg_gap':   float(ksg.get('ksg_gap', np.nan)),
        }

    with open('summary.json', 'w') as f:
        json.dump(s, f, indent=2)
    print('  wrote summary.json')


if __name__ == '__main__':
    main()
