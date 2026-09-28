"""
meteo_cluster.py
================

Unsupervised meteorological regime classification for PEA station,
following Gorodetskaya et al. (2013).

Function `meteo_cluster()`:

    1. load the slowdata_cleaned (one pickle per year),
    2. resample to the requested time resolution (1h / 6h / 1d),
    3. compute the six clustering variables
       (specific humidity, pressure, wind speed, incoming longwave
        radiation, near-surface temperature inversion, wind direction),
    4. remove the mean seasonal (and, for sub-daily data, diurnal) cycle,
    5. z-score the anomalies and run Ward hierarchical clustering (k = 3),
    6. label the clusters by decreasing temperature inversion
       (Cold Katabatic / Transitional Synoptic / Warm Synoptic),
    7. save a single CSV with the meteo variables + the regime assignment.

Choices for input argument:
- period (start, end)
- climatology years
- sensor heights (for 2024, 2025 already defined)
- time resolution (1h, 6h, 1d)
- whether to include wind direction
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


# --------------------------------------------------------------------------
# Fixed labels / colours for the three regimes (in order of DECREASING
# near-surface temperature inversion)
# --------------------------------------------------------------------------
REGIME_ORDER = ['Cold Katabatic', 'Transitional Synoptic', 'Warm Synoptic']
REGIME_COLORS = {
    'Cold Katabatic':        "#1f77b4",
    'Transitional Synoptic': "#ff7f0e",
    'Warm Synoptic':         '#d62728',
}
REGIME_MARKERS = {
    'Cold Katabatic':        'o',
    'Transitional Synoptic': '^',
    'Warm Synoptic':         's',
}

# The six variables that define a regime. WD (wind direction) is optional.
CLUSTER_VARIABLES = ['q', 'P', 'WS', 'LW_in', 'T_inversion', 'WD']

# Physical constants for the surface-temperature retrieval.
SIGMA_SB = 5.67e-8   # Stefan-Boltzmann constant [W m-2 K-4]

# Sensor mounting heights [m] at the START of each year (i.e. above the snow
# surface at that moment)
#   h_T = HygroVUE10 air-temperature sensor
#   h_S = SR50A snow-distance sensor
# The change of the snow level during the season is considered with HS_Cor 
# (distance from snow surface) measured
# by SR50A
SENSOR_HEIGHTS = {
    2022: {'h_T': 1.425, 'h_S': 1.74},   # measured 10 Dec 2021 (start of 2021-2022 season)
    2023: {'h_T': 1.30,  'h_S': 1.63},   # setup 28 Nov 2022 (start of 2022-2023 season)
    2024: {'h_T': 1.55,  'h_S': 1.60},
    2025: {'h_T': 1.374, 'h_S': 1.374},
}


def _get_heights(year, h_T, h_S):
    """Return (h_T, h_S) for a given year.
    Uses h_T, h_S if given, otherwise uses the predefined ones in SENSOR_HEIGHTS
    """
    defaults = SENSOR_HEIGHTS.get(year, {})
    h_T = h_T if h_T is not None else defaults.get('h_T')
    h_S = h_S if h_S is not None else defaults.get('h_S')
    if h_T is None or h_S is None:
        raise ValueError(
            f'No default sensor heights for year {year}; '
            f'pass h_T and h_S explicitly.')
    return h_T, h_S


# ==========================================================================
# Helper functions
# ==========================================================================

def _load_year(year, data_template):
    """Load one yearly slow-data pickle and return it with a datetime index.

    A pressure correction of +80 kPa is applied only if the stored pressure
    is still in the (uncorrected) low range.
    """
    path = Path(str(data_template).format(year=year))
    df = pd.read_pickle(path)

    # Put the timestamp on the index.
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df = df.dropna(subset=['TIMESTAMP']).set_index('TIMESTAMP')
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.isna()]

    # Pressure correction (+80 kPa) if necessary.
    if 'LI_Pres_Avg' in df.columns:
        p = pd.to_numeric(df['LI_Pres_Avg'], errors='coerce')
        if p.median() < 10:
            df['LI_Pres_Avg'] = p + 80.0
        else:
            df['LI_Pres_Avg'] = p

    # Physical-range cleaning not done upstream:
    #  - wind speed: drop spikes above 50 m/s (and negatives). Some years have
    #    huge spurious values (e.g. 1334 or 7999 m/s) that would dominate the
    #    z-scores if left in.
    for c in ['WS1_Avg', 'WS2_Avg', 'WS1_Max', 'WS2_Max', 'WS_FC4']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            df[c] = s.where((s >= 0) & (s <= 50))
    #  - wind direction: must be within [0, 360]; a few years have negatives.
    for c in ['WD1', 'WD2']:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            df[c] = s.where((s >= 0) & (s <= 360))

    return df


def _resample(df, freq):
    """Average all numeric columns to the requested frequency (1h/6h/1d)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].resample(freq).mean()


def _temperature_inversion(df, h_T, h_S, eps):
    """Near-surface temperature inversion dT/dz  [K m-1].

    The surface temperature is retrieved from the outgoing longwave flux by
    inverting the Stefan-Boltzmann law for a grey body of emissivity `eps`
    that also reflects a fraction (1 - eps) of the incoming longwave.
    The sensor height above the snow is derived from the SR50A distance
    (HS_Cor), corrected for the offset between the two sensors.
    """
    sensor_offset = h_T - h_S                       # negative: TA sensor is lower

    hs_m = df['HS_Cor'] if 'HS_Cor' in df.columns else pd.Series(np.nan, index=df.index)
    H = (hs_m + sensor_offset).clip(lower=0.05)     # height of TA sensor above snow

    TA_K = df['TA'] + 273.15
    LWout = df['LWup1']   if 'LWup1'   in df.columns else pd.Series(np.nan, index=df.index)
    LWin  = df['LWdown1'] if 'LWdown1' in df.columns else pd.Series(np.nan, index=df.index)

    numerator = (LWout - (1 - eps) * LWin).clip(lower=0)
    Tsurf_K = np.power(numerator / (eps * SIGMA_SB), 0.25)

    return (TA_K - Tsurf_K) / H


def _build_variables(df, q_func, ws_col='WS2_Avg', wd_col='WD2'):
    """From a resampled dataframe build the named clustering variables.

    Returns a DataFrame with columns q, P, WS, LW_in, T_inversion, WD, TA.
    Falls back to alternative sensor columns when the preferred one is absent.
    `ws_col` / `wd_col` choose which anemometer is used (sensor 2 by default).
    """
    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series(np.nan, index=df.index)

    P  = pick('LI_Pres_Avg')
    WS = pick(ws_col)                # anemometer 2 is the reference
    if 'WS1_Avg' in df.columns:
        WS = WS.fillna(df['WS1_Avg'])  # use anemometer 1 only to fill gaps
    LW_in = pick('LWdown1')
    TA = pick('TA')
    RH = pick('RH')
    WD = pick(wd_col, 'WD2')
    WD = WD.fillna(df['WD1'])
    q  = q_func(RH, TA, P * 1000.0)                 # P in Pa for the humidity formula

    return pd.DataFrame({
        'q': q, 'P': P, 'WS': WS, 'LW_in': LW_in,
        'T_inversion': df['T_inversion'], 'WD': WD, 'TA': TA,
    })


def _year_variables(year, freq, data_template, h_T, h_S, eps, q_func,
                    ws_col, wd_col):
    """Load one year, resample, and build the clustering variables.

    The temperature inversion is computed with that year's sensor heights, so
    a multi-year period (or climatology) uses the right geometry for each year.
    """
    hT, hS = _get_heights(year, h_T, h_S)
    d = _resample(_load_year(year, data_template), freq)
    d['T_inversion'] = _temperature_inversion(d, hT, hS, eps)
    return _build_variables(d, q_func, ws_col, wd_col)


def _climatology_components(clim_years, freq, data_template, h_T, h_S, eps,
                            q_func, variables, ws_col, wd_col):
    """Build the climatological baseline from the clim_years pickles.

    Two components, returned as a (seasonal, diurnal) tuple:

    seasonal : DataFrame indexed by day-of-year (1..366). It is the SMOOTH
        seasonal trend, obtained from the multi-year day-of-year climatology of
        the daily means and a centred 30-day rolling mean applied circularly
        (the year is tripled so 31 Dec joins 1 Jan with no step). Being smooth,
        it follows the slow seasonal envelope but does NOT track individual
        stormy periods, so synoptic events survive as anomalies.

    diurnal : DataFrame indexed by (month, hour), or None for daily data. It is
        the mean day/night cycle, built from the deviation of each hour from the
        mean of its OWN day. Removing the day mean first strips the synoptic
        (day-level) signal, so the diurnal profile is pure; by construction each
        monthly profile averages to zero, so it does not touch the seasonal
        trend nor the cyclone intensity.
    """
    clim_src = pd.concat([
        _year_variables(year, freq, data_template, h_T, h_S, eps, q_func,
                        ws_col, wd_col)[variables]
        for year in clim_years
    ]).sort_index()
    clim_src = clim_src[~clim_src.index.duplicated(keep='first')]

    # --- STEP 1: smooth seasonal trend -----------------------------------
    daily = clim_src.resample('D').mean()
    doy = daily.groupby(daily.index.dayofyear).mean().reindex(range(1, 367))
    doy = doy.interpolate(limit_direction='both')
    tripled = pd.concat([doy, doy, doy], ignore_index=True)
    seasonal = tripled.rolling(30, center=True, min_periods=1).mean().iloc[366:732]
    seasonal.index = range(1, 367)

    # --- STEP 2: pure diurnal cycle, stratified by month (sub-daily only) -
    diurnal = None
    if freq.endswith('h'):
        day_mean = clim_src.groupby(clim_src.index.normalize()).transform('mean')
        dev = clim_src - day_mean
        month = clim_src.index.month.rename('month')
        hour = clim_src.index.hour.rename('hour')
        diurnal = dev.groupby([month, hour]).mean()

    return seasonal, diurnal


def _detrend(df, components, freq, variables):
    """Subtract the climatological baseline (seasonal + diurnal) from `df`.

    The baseline for every timestamp is the smooth seasonal value for its
    day-of-year plus, for sub-daily data, the diurnal value for its
    (month, hour). The result is the series of pure weather anomalies.
    """
    seasonal, diurnal = components

    baseline = seasonal.reindex(df.index.dayofyear)
    baseline.index = df.index                       # seasonal part
    if diurnal is not None:
        mh = pd.MultiIndex.from_arrays([df.index.month, df.index.hour])
        diu = diurnal.reindex(mh)
        diu.index = df.index
        baseline = baseline[variables] + diu[variables]

    return df[variables] - baseline[variables]


# ==========================================================================
# Main clustering function
# ==========================================================================

def meteo_cluster(
    start,
    end,
    freq='1h',
    clim_years=None,
    include_wd=True,
    h_T=None,
    h_S=None,
    ws_col='WS2_Avg',
    wd_col='WD2',
    data_template=r'D:\slowdata_24_06\slowdata_cleaned_{year}.pkl',
    src_path=r'C:\Users\lory2\Desktop\tesi\DataProcessingScripts\src',
    eps=0.99,
    method='ward',
    output_csv=None,
):
    """Run the meteorological regime clustering and return the labelled data.

    Parameters
    ----------
    start, end : str or pandas.Timestamp
        Period to classify (inclusive), e.g. '2025-01-01' .. '2025-12-31'.
    freq : {'1h', '6h', '1d'}
        Time resolution. For sub-daily ('1h', '6h') both the seasonal and the
        diurnal cycle are removed; for daily ('1d') only the seasonal cycle.
    clim_years : list of int, optional
        Years whose pickles are used to build the climatology (mean cycle).
        Recommended to include at least one year in addition to the analysed
        one (e.g. [2023, 2024, 2025]). Defaults to the years spanned by
        start..end.
    include_wd : bool
        If False, wind direction is dropped from the clustering variables.
    h_T, h_S : float, optional
        Mounting heights [m] (at the start of the year) of the air-temperature
        sensor (HygroVUE10) and of the snow-distance sensor (SR50A). If left
        as None they are looked up per-year in SENSOR_HEIGHTS (2024 / 2025).
        The seasonal change of the snow level is handled automatically through
        the time-varying HS_Cor reading, not through these constants.
    data_template : str
        Path template for the yearly pickles, with a `{year}` placeholder.
    src_path : str
        Path to DataProcessingScripts/src (for RH_to_specific_humidity).
    eps : float
        Snow emissivity used in the surface-temperature retrieval.
    output_csv : str or Path, optional
        If given, the result table is written there as CSV.

    Returns
    -------
    dict with keys:
        'data'         : DataFrame, one row per period, with the meteo
                         variables, 'cluster', 'regime' and 'dist_to_centroid'.
        'climatology'  : the mean-cycle table that was subtracted.
        'X_std'        : the standardised anomalies fed to the clustering.
        'linkage'      : the Ward linkage matrix (for a dendrogram).
        'variables'    : the list of clustering variables actually used.
        'regime_order' / 'regime_colors' / 'regime_markers' : plotting helpers.
    """
    # --- 0. set up -------------------------------------------------------
    if freq not in ('1h', '6h', '1d'):
        raise ValueError("freq must be one of '1h', '6h', '1d'")
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)
    from utils.utils import RH_to_specific_humidity as q_func

    variables = list(CLUSTER_VARIABLES)
    if not include_wd:
        variables.remove('WD')

    if clim_years is None:
        clim_years = list(range(start.year, end.year + 1))

    # --- 1. load + resample the period to classify -----------------------
    # Each year is processed with its own sensor heights, then concatenated.
    years = list(range(start.year, end.year + 1))
    df_vars = pd.concat([
        _year_variables(y, freq, data_template, h_T, h_S, eps, q_func,
                        ws_col, wd_col)
        for y in years
    ]).sort_index()
    # Drop duplicated year-boundary timestamps (e.g. 1 Jan 00:00 appears in both
    # the previous and the next yearly file).
    df_vars = df_vars[~df_vars.index.duplicated(keep='first')]
    df_vars = df_vars.loc[(df_vars.index >= start) & (df_vars.index <= end)]
    df_vars = df_vars.dropna(subset=variables)

    # --- 2. climatological baseline (smooth seasonal + diurnal) ----------
    # Built from the raw pickles of `clim_years`. The smooth 30-day seasonal
    # trend and the month-stratified diurnal cycle are removed, while synoptic
    # (day-to-day) variability is preserved.
    components = _climatology_components(
        clim_years, freq, data_template, h_T, h_S, eps, q_func, variables,
        ws_col, wd_col)

    # --- 3. detrend + z-score -------------------------------------------
    anomalies = _detrend(df_vars, components, freq, variables).dropna()
    df_vars = df_vars.loc[anomalies.index]          # keep the rows we can use

    X_std = pd.DataFrame(
        StandardScaler().fit_transform(anomalies),
        index=anomalies.index, columns=variables,
    )

    # --- 4. clustering, k = 3 -------------------------------------------
    # Ward hierarchical by default (gives a dendrogram); KMeans for large
    # datasets (e.g. multi-year hourly) where the O(n^2) Ward distance matrix
    # would not fit in memory.
    if method == 'ward':
        Z = linkage(X_std, method='ward')
        raw_labels = fcluster(Z, 3, criterion='maxclust')
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        Z = None
        raw_labels = KMeans(3, n_init=10, random_state=0).fit_predict(X_std) + 1
    else:
        raise ValueError("method must be 'ward' or 'kmeans'")

    out = df_vars.copy()
    out['cluster_raw'] = raw_labels

    # Order raw clusters by mean temperature inversion (high -> low) and map
    # them onto the three named regimes.
    ordered = (out.groupby('cluster_raw')['T_inversion']
                  .mean().sort_values(ascending=False).index.tolist())
    raw_to_regime = {raw: REGIME_ORDER[i] for i, raw in enumerate(ordered)}
    regime_to_rank = {name: i + 1 for i, name in enumerate(REGIME_ORDER)}

    out['regime'] = out['cluster_raw'].map(raw_to_regime)
    out['cluster'] = out['regime'].map(regime_to_rank)

    # --- 5. distance to the cluster centroid (in standardised space) ----
    dist = pd.Series(index=X_std.index, dtype=float)
    for cl in out['cluster'].unique():
        mask = out['cluster'] == cl
        centroid = X_std.loc[mask].mean()
        dist.loc[mask] = np.linalg.norm(X_std.loc[mask] - centroid, axis=1)
    out['dist_to_centroid'] = dist

    # --- 6. assemble + save ---------------------------------------------
    keep = variables + ['TA', 'cluster', 'regime', 'dist_to_centroid']
    # always keep wind direction in the output as an a-posteriori column, even
    # when it is not a clustering variable (include_wd=False), so the saved CSV
    # is always complete.
    if 'WD' not in keep and 'WD' in out.columns:
        keep = keep + ['WD']
    result = out[keep].copy()
    result.index.name = 'datetime'

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv)
        print(f'Saved {len(result)} periods -> {output_csv}')
        if Z is not None:
            linkage_path = Path(output_csv).with_suffix('.npy')
            np.save(linkage_path, Z)
            print(f'Saved linkage matrix  -> {linkage_path}')

    print('Regime counts:')
    print(result['regime'].value_counts().reindex(REGIME_ORDER))

    return {
        'data': result,
        'anomalies': anomalies,                 # detrended (pre z-score) series
        'baseline_seasonal': components[0],     # smooth seasonal, by day-of-year
        'baseline_diurnal': components[1],      # diurnal cycle, by (month, hour)
        'X_std': X_std,
        'linkage': Z,
        'variables': variables,
        'regime_order': REGIME_ORDER,
        'regime_colors': REGIME_COLORS,
        'regime_markers': REGIME_MARKERS,
    }
