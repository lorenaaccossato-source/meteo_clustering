"""
Run the transport-only meteorological clustering with the following transport thresholds
(SPC > 0.01 g/m2/s  OR  FlowCapt > 1 g/m2/s) and check that the three CRYOWRF
simulation events fall in their expected regime.

it reuses meteo_cluster() (loading, detrending and
standardising the meteo), then runs Ward k=3 only on the transport hours,
re-standardised over that subset, and labels
the clusters by decreasing near-surface inversion.

"""
import sys, gc
sys.path.insert(0, r'C:\Users\lory2\Desktop\tesi\final_scripts')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from meteo_cluster import meteo_cluster, REGIME_ORDER

# ---------------- config ----------------
SPC_THR = 0.01          # g/m2/s  (new SPC cut)
FC_THR  = 1.0           # g/m2/s  (new FlowCapt cut)
SPC_TO_GM2S = 1000.0    # SPC pickle is kg/m2/s
FC_TO_GM2S  = 1.0       # PF_FC4 already g/m2/s
CLIM_YEARS  = [2023, 2024, 2025]
YEARS       = [2024, 2025]

# ---------------- 1. meteo: reuse meteo_cluster() to get X_std ----------------
# method='kmeans' only to run fast over the full period; we ignore its labels
# and re-cluster the transport subset ourselves with Ward.
res = meteo_cluster(start='2024-01-01', end='2025-12-31', freq='1h',
                    clim_years=CLIM_YEARS, include_wd=False, method='kmeans')
Xstd = res['X_std']                 # standardised anomalies, all 2024-25 hours
data = res['data']                  # meteo variables incl. T_inversion
Tinv = data['T_inversion']

# ---------------- 2. transport hours from 1-min flux (new threshold) ----------------
def hourly_transport(y):
    d = pd.read_pickle(rf'D:\slowdata_24_06\slowdata_cleaned_{y}.pkl')
    d.index = pd.to_datetime(d.index, errors='coerce')
    fc = pd.to_numeric(d['PF_FC4'], errors='coerce') * FC_TO_GM2S
    del d; gc.collect()
    fc = fc[fc.index.notna()].sort_index(); fc = fc[~fc.index.duplicated()]

    s = pd.read_pickle(rf'D:\data\spc{y}.pkl')
    col = [c for c in s.columns if 'Corrected' in str(c) and 'Flux' in str(c)][-1]
    spc = pd.to_numeric(s[col], errors='coerce') * SPC_TO_GM2S
    del s; gc.collect()
    spc.index = pd.to_datetime(spc.index, errors='coerce')
    spc = spc[spc.index.notna()].sort_index(); spc = spc[~spc.index.duplicated()]

    fc5  = fc.resample('5min').mean()
    spc5 = spc.resample('5min').mean()
    tr5  = (spc5 > SPC_THR) | (fc5 > FC_THR)          # per-5-min transport flag
    return tr5.resample('1h').max().fillna(False).astype(bool)  # any step -> transport hour

tr  = pd.concat([hourly_transport(y) for y in YEARS]).sort_index()
hrs = Xstd.index[tr.reindex(Xstd.index).fillna(False).values]
print(f'transport hours: {len(hrs)} of {len(Xstd)} '
      f'({100*len(hrs)/len(Xstd):.0f}%)  [SPC>{SPC_THR}, FC>{FC_THR} g/m2/s]')

# ---------------- 3. Ward k=3 on the transport subset, re-standardised ----------------
# re-standardising X_std over the subset is equivalent to z-scoring the raw
# anomalies over the transport hours (standardisation removes the prior scaling).
Zsub = StandardScaler().fit_transform(Xstd.loc[hrs])
link = linkage(Zsub, method='ward')
lab  = fcluster(link, 3, criterion='maxclust')
order_c = Tinv.loc[hrs].groupby(lab).mean().sort_values(ascending=False).index.tolist()
cmap    = {c: REGIME_ORDER[i] for i, c in enumerate(order_c)}
regime  = pd.Series([cmap[l] for l in lab], index=hrs, name='dbs_regime')

print('\nregime counts (new-threshold DBS clustering):')
print(regime.value_counts().reindex(REGIME_ORDER).to_string())

# ---------------- 4. check the three simulation events ----------------
EV = {'COLD_KAT':  ('2024-05-13 01:00', '2024-05-14 14:00'),
      'TRANS_SYN': ('2025-04-23 05:00', '2025-04-24 00:00'),
      'WARM_SYN':  ('2025-04-24 12:00', '2025-04-25 12:00')}
print('\nregime composition of each simulated event (% of its transport hours):')
for name, (a, b) in EV.items():
    sub  = regime.loc[a:b]
    comp = (sub.value_counts(normalize=True).mul(100).round(0)
              .reindex(REGIME_ORDER).fillna(0).astype(int).to_dict())
    dom  = sub.mode().iloc[0] if len(sub) else 'NA'
    print(f'  {name:10s} -> dominant: {dom:22s} {comp}  (n={len(sub)})')

regime.to_frame().to_csv(r'D:\output_24_06\regimes_1h_dbs_newthr.csv')
print('\nsaved -> D:\\output_24_06\\regimes_1h_dbs_newthr.csv')
