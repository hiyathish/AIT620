import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load input data
X = np.loadtxt('data_clustering.txt', delimiter=',')
num_clusters = 5

# ---------------------------------------------------------
# PART 1 — VISUALIZE INPUT DATA
# ---------------------------------------------------------
plt.figure()
plt.scatter(X[:,0], X[:,1], s=10, edgecolors='black', facecolors='none')
plt.title("Input Data")
plt.show()

# ---------------------------------------------------------
# PART 2 — RUN K-MEANS 10 TIMES AND PLOT EACH RUN
# ---------------------------------------------------------
inertias = []
centers_list = []

print("\nRunning K-Means 10 times with different initializations...\n")

for i in range(10):
    km = KMeans(init='random', n_clusters=num_clusters, n_init=1, random_state=i)
    km.fit(X)

    inertias.append(km.inertia_)
    centers_list.append(km.cluster_centers_)

    print(f"Run {i+1}: inertia = {km.inertia_}")

    # Plot each run separately
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=10, edgecolors='black', facecolors='none')
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
                s=200, color='red', marker='X')
    plt.title(f"K-Means Initialization Run {i+1}")
    plt.show()

# ---------------------------------------------------------
# PART 3 — SELECT BEST RUN
# ---------------------------------------------------------
best_run_index = np.argmin(inertias)
print("\nBest run:", best_run_index + 1)
print("Best inertia:", inertias[best_run_index])

# Refit using best run
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=1,
                random_state=best_run_index)
kmeans.fit(X)

# ---------------------------------------------------------
# PART 4 — FINAL BOUNDARY VISUALIZATION
# ---------------------------------------------------------
step_size = 0.01

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
                   y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, origin='lower', aspect='auto')

plt.scatter(X[:,0], X[:,1], s=80, edgecolors='black', facecolors='none')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=210, color='black')

plt.title("Final Cluster Boundaries (Best Initialization)")
plt.xticks(())
plt.yticks(())
plt.show()
