import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyod.utils.data import evaluate_print
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np


cpu_data = pd.read_csv('../datafiles/cpu4.csv')


# Use seaborn style defaults and set the default figure size
cpu_data['datetime'] = cpu_data.timestamp.astype(int).apply(datetime.fromtimestamp)
sns.set(rc={'figure.figsize':(11, 4)})

time_data = cpu_data.set_index('datetime')
time_data.loc[time_data['label'] == 1.0, 'value'].plot(linewidth=0.5, marker='o', linestyle='')
time_data.loc[time_data['label'] == 0.0, 'value'].plot(linewidth=0.5)

plt.show()
cpu_data.info()

markers = ['r--', 'b-^']

def hist2d(X, by_col, n_bins=10, title=None):
    bins = np.linspace(X.min(), X.max(), n_bins)
    vals = np.unique(by_col)

    for marker, val in zip(markers, vals):
        n, edges = np.histogram(X[by_col == val], bins=bins)
        n = n / np.linalg.norm(n)

        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        plt.plot(bin_centers, n, marker, alpha=0.8, label=val)

    plt.legend(loc='upper right')

    if title is not None:
        plt.title(title)

    plt.show()

hist2d(cpu_data.value, cpu_data.label, n_bins=50, title='Values by label')

print("LENGTH CPU DATA: ", len(cpu_data))

print("----------------------")

cpu_data.timestamp.hist()
plt.show()

cpu_data.value.hist(bins=50)
plt.show()

import numpy as np
from matplotlib import pyplot as plt

markers = ['r--', 'b-^']


def hist2d(X, by_col, n_bins=10, title=None):
    '''plot two histograms against each other.

    I am using line plots here. Alternatively,
    we could be using hist() with opacity.
    '''
    bins = np.linspace(X.min(), X.max(), n_bins)

    vals = np.unique(by_col)
    for marker, val in zip(markers, vals):
        n, edges = np.histogram(X[by_col == val], bins=bins)
        n = n / np.linalg.norm(n)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        plt.plot(bin_centers, n, marker, alpha=0.8, label=val)

    plt.legend(loc='upper right')
    if title is not None:
        plt.title(title)
    plt.show()

hist2d(cpu_data.value, cpu_data.label, n_bins=50, title='Values by label')

#print("CPU DATA HIST: \n", cpu_data.label.hist())
cpu_data.label.hist()

print("-------------------------")

X_train, X_test, y_train, y_test = train_test_split(cpu_data[['value']].values, cpu_data.label.values)

print("X-TEST SHAPE ", X_test.shape)

# simplest example from the docs: train kNN detector
from pyod.models.knn import KNN
clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

hist2d(X_test, y_test_pred, title='Predicted values by label')

X_train[y_train==0.0].shape

X_train.shape

y_train.sum()

def test_outlier_detector(X_train, y_train,
                          X_test, y_test, only_neg=True,
                          basemethod=KNN()):
  '''Test an outlier detection method on a dataset.
  This function trains a model, get performance metrics of
  the model, and plots a visualization.

  Parameters:
  -----------
  X_train : training features
  y_train : training labels
  X_test : test features
  y_test : test labels
  only_neg : whether to use only normal (i.e. not outliers) for training
  basemethod : the model to test
  '''

  clf = basemethod
  if only_neg:
    clf.fit(X_train[y_train==0.0], np.zeros(shape=((y_train==0.0).sum(), 1)))
  else:
    clf.fit(X_train, y_train)  # most algorithms ignore y

  y_train_pred = clf.predict(X_train)  # labels_
  y_train_scores = clf.decision_scores_

  y_test_pred = clf.predict(X_test)
  y_test_scores = clf.decision_function(X_test)

  print("\nOn Test Data:")
  evaluate_print(type(clf).__name__, y_test, y_test_scores)
  hist2d(X_test, y_test_pred, title='Predicted values by label')

  test_outlier_detector(X_train, y_train, X_test, y_test, only_neg=False,
                        basemethod=KNN(n_neighbors=3, metric='hamming', method='mean', contamination=0.01))

  from pyod.models.auto_encoder import AutoEncoder

  ae = AutoEncoder(
      hidden_neuron_list=[1],
      batch_size=32,
      contamination=0.01,
      verbose=0)

  ae.model.summary()

  # from sklearn.metrics import roc_auc_score
  # roc_auc_score(y_test, preds.mean().numpy())

  from sklearn.metrics import roc_auc_score
  import tensorflow as tf

  roc_auc_score(y_test, mean_preds)

  auc = roc_auc_score(to_one_hot(y_test), class_probs)
  print("auc score: {:.3f}".format(auc))