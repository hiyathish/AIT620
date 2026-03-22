import fontTools.fontBuilder
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

## To remove warnings
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"   # or your logical core count
import warnings
warnings.filterwarnings("ignore")
from networkx.classes import non_neighbors

names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

customers = pd.read_csv('../datafiles/german.data', names=names, delimiter=' ')

print("CUSTOMERS HEAD:\n", customers.head())

from dython.nominal import associations
#fig = pt.figure(figsize=(16, 16))

pt.rcParams['font.family'] = 'serif'  # Example: use serif font family
pt.rcParams['font.size'] = 5         # Example: set default font size to 12

result = associations(customers, clustering=True, figsize=(16, 16),cmap='YlOrBr')
# Adjust tick label fonts

# Step 2 — get the axes AFTER associations() draws

pt.show()

catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
           'telephone', 'foreignworker']
numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
           'existingcredits', 'peopleliable', 'classification']

dummyvars = pd.get_dummies(customers[catvars])
transactions = pd.concat([customers[numvars], dummyvars], axis = 1)

print("TRANSACTIONS HEAD:\n", transactions.head())

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
sse = {}
for k in range(1, 15):
    kmeans = (KMeans(n_clusters=k))
    kmeans.fit(transactions)
    sse[k] = kmeans.inertia_

plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
kmeans =(KMeans(n_clusters=4)).fit(transactions)
y = kmeans.labels_

clusters = (transactions.join(pd.DataFrame(data=y, columns=['cluster'])).
            groupby('cluster')
            .agg(
                age_mean=pd.NamedAgg(column='age', aggfunc='mean'),
                age_std=pd.NamedAgg(column='age', aggfunc='std'),
                creditamount=pd.NamedAgg(column='creditamount',aggfunc='mean'),
                duration=pd.NamedAgg(column='duration', aggfunc='mean'),
                count=pd.NamedAgg(column='age', aggfunc='count'),
                class_mean=pd.NamedAgg(column='classification', aggfunc='mean'),
                class_std=pd.NamedAgg(column='classification',aggfunc='std'),
            ).sort_values(by='class_mean'))
print("CLUSTERS 1: \n", clusters)


from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Compute pairwise distances on scaled features
distances = squareform(
    pdist(
        StandardScaler().fit_transform(
            transactions[['classification', 'creditamount', 'duration']]
        )
    )
)

# Perform agglomerative clustering using the precomputed distance matrix
clustering = AgglomerativeClustering(
    n_clusters=5,
    metric='precomputed',
    linkage='average'
).fit(distances)

# Extract cluster labels
y = clustering.labels_

clusters = (
    transactions
        .join(pd.DataFrame(data=y, columns=['cluster']))
        .groupby(by='cluster')
        .agg(
            age_mean=pd.NamedAgg(column='age', aggfunc='mean'),
            age_std=pd.NamedAgg(column='age', aggfunc='std'),
            creditamount=pd.NamedAgg(column='creditamount', aggfunc='mean'),
            duration=pd.NamedAgg(column='duration', aggfunc='mean'),
            count=pd.NamedAgg(column='age', aggfunc='count'),
            class_mean=pd.NamedAgg(column='classification', aggfunc='mean'),
            class_std=pd.NamedAgg(column='classification', aggfunc='std'),
        )
        .sort_values(by='class_mean')
)

print("CLUSTERS 2: \n", clusters)

import jax.numpy as jnp
import numpy as np

from jax import jit, vmap
from sklearn.base import ClassifierMixin

import jax
import random

from scipy.stats import hmean

class KMeans(ClassifierMixin):

    def __init__(self, k, n_iter=100):
        self.k = k
        self.n_iter = n_iter

        # JIT‑compiled batched Euclidean distance
        self.euclidean = jit(
            vmap(
                lambda x, y: jnp.linalg.norm(
                    x - y, ord=2, axis=-1, keepdims=False
                ),
                in_axes=(0, None),
                out_axes=0
            )
        )

    # def adjust_centers(self, X):
    #     return jnp.row_stack([
    #         X[self.clusters == c].mean(axis=0)
    #         for c in self.clusters
    #     ])

    def adjust_centers(self, X):
        return jnp.vstack([
            X[self.clusters == c].mean(axis=0)
            for c in range(self.k)
        ])

    def initialize_centers(self):
        """
        Roughly the k-means++ initialization
        """
        key = jax.random.PRNGKey(0)

        # JAX doesn't have uniform_multivariate
        self.centers = jax.random.multivariate_normal(
            key,
            jnp.mean(X, axis=0),
            jnp.cov(X, rowvar=False),
            shape=(1,)
        )

        for c in range(1, self.k):
            weights = self.euclidean(X, self.centers)

            if c > 1:
                weights = hmean(weights, axis=-1)
                print(weights.shape)

            # new_center = jnp.array(
            #     random.choices(X, weights=weights, k=1)[0],
            #     ndmin=2
            # )
            weights = self.euclidean(X, self.centers)  # shape (n_samples, n_centers)

            # Reduce to 1-D: distance to nearest center
            weights = weights.min(axis=1)

            # Convert distances to probabilities (invert so far points get higher weight)
            weights = 1 / (weights + 1e-8)

            # Normalize
            weights = weights / weights.sum()

            # Sample index
            idx = np.random.choice(len(X), p=np.array(weights))
            new_center = X[idx:idx + 1]

            self.centers = jnp.vstack((self.centers, new_center))
            print(self.centers.shape)

    def fit(self, X, y=None):
        self.initialize_centers()

        for iter in range(self.n_iter):
            dists = self.euclidean(X, self.centers)
            self.clusters = jnp.argmin(dists, axis=-1)
            self.centers = self.adjust_centers(X)


        return self.clusters

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
# kmeans = KMeans(k=3)
# kmeans.fit(X)
# plt.plot(X, kmeans.clusters, "K-Means")
# plt.show()
#Plot().plot_in_2d(X, kmeans.clusters, "K-Means"))



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt

pca = PCA().fit(
    StandardScaler().fit_transform(transactions)
)
ax = plt.plot(
    range(len(pca.explained_variance_ratio_)),
    np.cumsum(pca.explained_variance_ratio_)
)
plt.ylabel('explained variance (cummulative)')
plt.xlabel('dimensions')
plt.show()

print("COLUMNS:\n", len(transactions.columns))

#let's find a good dimensionality reduction
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np

dimensions_range = [1, 11]   # n_components = 1..10
n_neighbors = 10             # must be >= max n_components

reconstruction_errors = np.zeros(dimensions_range[1])

for n_components in range(*dimensions_range):

    lle = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_jobs=-1,
        method='standard'
    ).fit(transactions)

    # LLE always provides the barycentric weight matrix

    error = lle.reconstruction_error_
    reconstruction_errors[n_components] = error
print("Reconstruction errors:", reconstruction_errors[1:11])
