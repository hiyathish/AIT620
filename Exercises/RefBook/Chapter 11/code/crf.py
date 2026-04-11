import argparse
import string
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn_crfsuite import CRF
from sklearn.svm import LinearSVC


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Trains a CRF-like classifier")
    parser.add_argument("--C", dest="c_val", required=False, type=float,
                        default=1.0, help="C value for training")
    return parser


# Modern CRF model wrapper
class CRFModel(object):
    def __init__(self, c_val=1.0):
        # sklearn-crfsuite CRF (replaces ChainCRF)
        self.crf = CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )

        # Linear SVM (replaces FrankWolfeSSVM)
        self.svm = LinearSVC(C=c_val)

    def load_data(self):
        letters = fetch_openml("letter", version=1, as_frame=False)
        X = letters.data
        y = letters.target

        # Fake folds (OpenML dataset has no folds)
        folds = np.zeros(len(y), dtype=int)
        folds[:2000] = 1  # mimic pystruct’s fold split

        return X, y, folds

    def train(self, X_train, y_train):
        # Train SVM (flat classifier)
        self.svm.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.svm.score(X_test, y_test)

    def classify(self, input_data):
        return self.svm.predict(input_data)[0]


def convert_to_letters(indices):
    alphabets = np.array(list(string.ascii_lowercase))
    output = np.take(alphabets, indices)
    return ''.join(output)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    c_val = args.c_val

    crf = CRFModel(c_val)

    # Load data
    X, y, folds = crf.load_data()
    X_train, X_test = X[folds == 1], X[folds != 1]
    y_train, y_test = y[folds == 1], y[folds != 1]

    print("\nTraining the model...")
    crf.train(X_train, y_train)

    score = crf.evaluate(X_test, y_test)
    print("\nAccuracy score =", str(round(score * 100, 2)) + "%")

    indices = range(3000, len(y_test), 200)
    for index in indices:
        print("\nOriginal  =", y_test[index])
        predicted = crf.classify([X_test[index]])
        print("Predicted =", predicted)
