import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tslearn.metrics import dtw_path, cdist_dtw
import seaborn as sns

def seq_derv(g):
    '''compute sequence derivative
       g: Pandas DataFrame.'''
    prev = g.cpi.shift()
    next = g.cpi.shift(-1)
    dif = g.cpi.diff()
    rl_neigh = next - prev
    derivative = (dif + (rl_neigh/2)) / 2
    derivative.dropna(inplace=True)
    return derivative.to_numpy()

df = pd.read_excel('usa_inflation_series_bls.xlsx', skiprows=11)
df = df.drop(['HALF1', 'HALF2'], axis=1)
df.columns = ['Year'] + [f'cpi{m}' for m in df.columns[1:]]
df = pd.wide_to_long(df, stubnames='cpi', i='Year', j='Month', suffix='\D+')

myid = df.index.map(lambda t: pd.to_datetime(f'{t[0]} {t[1]}'))
df = pd.DataFrame(data={'start_term': myid, 'cpi': df.cpi.to_numpy()})
df.sort_values(by='start_term', inplace=True)
df.dropna(inplace=True)

# labels (presidents names)
pres = pd.read_csv('us-presidents.csv', names=['name', 'start_term', 'end_term'])
pres['start_term'] = pd.to_datetime(pres.start_term)
pres.drop(['end_term'], axis=1, inplace=True)

## rounding starting term
mask_dates = pres.start_term.dt.day > 15
pres.loc[mask_dates, 'start_term'] = pres.loc[mask_dates, 'start_term'] + pd.offsets.MonthBegin()
pres.loc[~mask_dates, 'start_term'] = pres.loc[~mask_dates, 'start_term'] - pd.offsets.MonthBegin()

# merge two dataframes
cleandf = pd.merge(df, pres, how='left', on='start_term')
cleandf.name.fillna(method='ffill', inplace=True)
cleandf.dropna(inplace=True)

raw_series_group = cleandf.groupby('name', sort=False)
_, ax = plt.subplots(5, 4, layout='tight')
for a, (name, group) in zip(ax.flat, raw_series_group):
    a.plot(group['start_term'], group['cpi'])
    a.set_title(name, fontsize=10)
    a.set_xticks(ticks=[])
plt.savefig('raw_series.png', dpi=300)

# calculate derivatives
cleandf = cleandf.groupby('name', sort=False).apply(lambda g: seq_derv(g))
_, ax = plt.subplots(5, 4, layout='tight')
pres_count = 0
for i in range(5):
    for j in range(4):
        if pres_count < 19:
            name = cleandf.index[pres_count]
            ax[i,j].plot(cleandf.loc[name])
            ax[i,j].set_title(name, fontsize=10)
        pres_count += 1
plt.savefig('derivatives.png', dpi=300)

# raw series and its derivative
_, ax = plt.subplots(19, 2, layout='tight', figsize=(8, 16))
for a, (name, group) in zip(ax[:, 0], raw_series_group):
    a.plot(group['cpi'])
    a.set_title(name, fontsize=10)
    a.set_xticks(ticks=[])
for i in range(19):
    name = cleandf.index[i]
    ax[i, 1].plot(cleandf.loc[name])
    ax[i, 1].set_xticks(ticks=[])
plt.savefig('raw_derv.png', dpi=300)

# tslearn input requirement
cleanseries = [s for s in cleandf]

# series alignment
trump = cleanseries[-2]
biden = cleanseries[-1]
path_trump_biden, dist_trump_biden = dtw_path(trump, biden)
xA = [c[0] for c in path_trump_biden]
yA = [trump[x] for x in xA]
xyAs = list(zip(xA, yA))
xB = [c[1] for c in path_trump_biden]
yB = [biden[x] for x in xB]
xyBs = list(zip(xB, yB))
fig, ax = plt.subplots(nrows=2, sharex='all', sharey='all')
ax[0].plot(trump, label='Donald Trump')
ax[0].text(1, trump[0]+1.5, f'distance = {dist_trump_biden}')
ax[0].legend()
ax[0].set_axis_off()
ax[1].plot(biden, label='Joe Biden')
ax[1].legend()
ax[1].set_axis_off()
for coordA, coordB in zip(xyAs, xyBs):
    mypatch = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA=ax[0].transData, coordsB=ax[1].transData)
    mypatch.set_color('orange')
    mypatch.set_linewidth(0.5)
    ax[1].add_artist(mypatch)
plt.savefig('alignment.png', dpi=300)

# get the distance matrix
distmat = cdist_dtw(cleanseries)
plt.figure(figsize=(12, 10))
sns.heatmap(distmat, annot=True, linewidths=0.03, xticklabels=pres.name, yticklabels=pres.name)
plt.savefig('distmat.png', dpi=300)

# warping path
plt.figure(figsize=(8,8))
plt.plot(xA, xB, color='purple', linewidth=3.)
plt.xlabel('indices of CPI during Trump presidency')
plt.ylabel('indices of CPI during Biden presidency (so far)')
plt.savefig('warping_path.png', dpi=300)