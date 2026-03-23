import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

# Load the iris dataset
iris = datasets.load_iris()

# Create StratifiedKFold (modern API)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Take the first fold
train_index, test_index = next(skf.split(iris.data, iris.target))

# Extract training data and labels
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# Extract testing data and labels
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# Extract the number of classes
num_classes = len(np.unique(y_train))

# Build GMM (updated parameters)
classifier = GaussianMixture(
    n_components=num_classes,
    covariance_type='full',
    init_params='kmeans',   # 'wc' no longer exists
    max_iter=20,
    random_state=42
)

# Initialize the GMM means manually
classifier.means_init = np.array([
    X_train[y_train == i].mean(axis=0)
    for i in range(num_classes)
])

# Train the GMM classifier
classifier.fit(X_train)

# Draw boundaries
plt.figure()
colors = 'bgr'

for i, color in enumerate(colors):
    # Extract covariance matrix (modern API)
    cov_matrix = classifier.covariances_[i][:2, :2]

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Normalize eigenvector
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])

    # Angle of ellipse
    angle = np.degrees(np.arctan2(norm_vec[1], norm_vec[0]))

    # Scale eigenvalues for visualization
    scaling_factor = 8
    eigenvalues *= scaling_factor

    # Draw ellipse
    ellipse = patches.Ellipse(
        classifier.means_[i, :2],
        eigenvalues[0],
        eigenvalues[1],
        angle=angle + 180,
        color=color,
        alpha=0.6
    )

    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    axis_handle.add_artist(ellipse)

# Plot the data
for i, color in enumerate(colors):
    cur_data = iris.data[iris.target == i]
    plt.scatter(cur_data[:, 0], cur_data[:, 1],
                marker='o', facecolors='none',
                edgecolors='black', s=40,
                label=iris.target_names[i])

    test_data = X_test[y_test == i]
    plt.scatter(test_data[:, 0], test_data[:, 1],
                marker='s', facecolors='black',
                edgecolors='black', s=40)

# Compute predictions
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred == y_train) * 100
print("Accuracy on training data =", accuracy_training)

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred == y_test) * 100
print("Accuracy on testing data =", accuracy_testing)

plt.title("GMM classifier")
plt.xticks(())
plt.yticks(())
plt.show()
