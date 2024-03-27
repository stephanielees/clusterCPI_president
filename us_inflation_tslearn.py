import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

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

def plot_dendrogram(model, label_data, linkage, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, orientation='left', labels=label_data, **kwargs)
    plt.title(f'{linkage} linkage')

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
cleandf.groupby('name', sort=False).cpi.plot()
plt.show()
cleandf = cleandf.groupby('name', sort=False).apply(lambda g: seq_derv(g))
_, ax = plt.subplots(5, 4, layout='tight')
pres_count = 0
for i in range(5):
    for j in range(4):
        if pres_count < 19:
            name = cleandf.index[pres_count]
            ax[i,j].plot(cleandf.loc[name])
            ax[i,j].set_title(name)
        pres_count += 1
plt.show()

# get the distance matrix
cleanseries = [s for s in cleandf]
distmat = cdist_dtw(cleanseries)

# clustering
labels_numpy = pres.name.to_numpy()
ag = AgglomerativeClustering(metric='precomputed', linkage='complete', compute_distances=True)
fitclust = ag.fit(distmat)
plot_dendrogram(fitclust, labels_numpy, 'complete')
plt.show()

plot_dendrogram(fitclust, labels_numpy, 'complete')
plt.axvline(x=3, color='black')
plt.show()

ag = AgglomerativeClustering(metric='precomputed', linkage='average', compute_distances=True)
fitclust = ag.fit(distmat)
plot_dendrogram(fitclust, labels_numpy, 'average')
plt.show()

ag = AgglomerativeClustering(metric='precomputed', linkage='single', compute_distances=True)
fitclust = ag.fit(distmat)
plot_dendrogram(fitclust, labels_numpy, 'single')
plt.show()