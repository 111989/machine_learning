import numpy as np
from csv import reader
from random import randrange
from random import seed
import argparse
import matplotlib.pyplot as plt


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        sample_size = X.shape[0]  # get the sample size
        # get the feature column of our threshold
        f_column = X[:, self.feature_idx]
        predictions = np.ones(sample_size)  # default all to positive
        # update the negatives based on polarity. Polatity depends on error value
        if self.polarity == 1:
            predictions[f_column < self.threshold] = -1
        else:
            predictions[f_column > self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        sample_size, features_size = X.shape

        w = np.full(sample_size, (1/sample_size))  # w0 with each weight as 1/N

        self.clfs = []
        alpha = []

        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            for idx in range(features_size):
                f = np.expand_dims(X[:, idx], axis=1)
                thresholds = np.unique(f)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(np.shape(y))
                    predictions[X[:, idx] < threshold] = -1

                    err = sum(w[y != predictions])

                    if err > 0.5:
                        err = 1 - err
                        p = -1

                    if err < min_error:
                        min_error = err
                        clf.feature_idx = idx
                        clf.polarity = p
                        clf.threshold = threshold

            # calculate alpha
            clf.alpha = 0.5 * np.log((1-min_error) / (min_error + 1e-10))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)
            alpha.append(clf.alpha)

        return alpha

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        pred = np.sum(clf_preds, axis=0)
        pred = np.sign(pred)
        return pred


# Emperical risk minimization
def erm(y_pred, y_true):
    return 1 - sum(y_pred == y_true) / len(y_true)


# Split in to X,Y
def split_X_Y(dataset):
    x = []
    y = []
    for row in dataset:
        x.append([p for p in row[:-1]])
        if row[-1] == 0:
            y.append(-1)
        else:
            y.append(1)
    return np.array(x), np.array(y)


def load_csv(inp_file):
    dataset = []
    with open(inp_file, 'r') as file:
        csv_reader = reader(file)
        for data_row in csv_reader:
            dataset.append(data_row)
    return dataset


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    np.random.seed(seed)
    idx_val = np.arange(X.shape[0])
    np.random.shuffle(idx_val)
    X = X[idx_val]
    y = y[idx_val]

    split_index = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def cross_validation_split(dataset, n_folds):
    splits = []
    dataset_copy = list(dataset)
    split_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        split = []
        while len(split) < split_size:
            index = randrange(len(dataset_copy))
            split.append(dataset_copy.pop(index))
        splits.append(split)
    return splits


def predict(X_train, X_test, y_train, y_test, n_clf=6):
    clf = AdaBoost(n_clf=n_clf)
    # training
    hypothesis = clf.fit(X_train, y_train)
    # prediction
    y_pred = clf.predict(X_test)
    return hypothesis, erm(y_pred, y_test)


def cross_validation_train(dataset, n_folds=10, n_clf=6):
    # get n_fold data
    splits = cross_validation_split(dataset, n_folds)
    erms = []
    iteration = 1
    # Iterate over the n_fold data
    for split in splits:
        train_set = list(splits)
        train_set.remove(split)
        train_set = sum(train_set, [])
        test_set = []
        for row in split:
            test_set.append(list(row))
        # format the dataset
        X_train, y_train = split_X_Y(train_set)
        X_test, y_test = split_X_Y(test_set)

        # Train
        hypothesis, erm = predict(
            X_train, X_test, y_train, y_test, n_clf)
        # Error calculation
        erms.append(erm)
        print("Split %d details:" % iteration)
        print("Hypothesis:")
        print(hypothesis)
        print("Error:")
        print(erm)
        print("\n")
        iteration = iteration + 1
    return erms


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--mode', help='erm or nfold or tsize', required=True)
args = vars(parser.parse_args())

# get the dataset file
data_file = args['dataset']
mode = args["mode"]

seed(1)
# Load the input CSV file
dataset = load_csv(data_file)

dataset = dataset[1:]
i = 0
for row in dataset:
    dataset[i] = [float(x.strip()) for x in row]
    i += 1


n_folds = 10

if mode == "erm":
    X, y = split_X_Y(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    hypothesis, err = predict(X_train, X_test, y_train, y_test)
    print("Hypothesis:")
    print(hypothesis)
    print("Error:")
    print(err)
elif mode == "nfold":
    erm_values = cross_validation_train(dataset, n_folds)
    print("Cross validation mode: %d folds" % n_folds)
    print("Error values:")
    print(erm_values)
    print("Average error: %f" % (sum(erm_values) / n_folds))
elif mode == "tsize":
    print("T size analysis:")
    X, y = split_X_Y(dataset)
    # Test-Train splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    erm_normal = []
    erm_cross_val = []
    t_max = 10
    for i in range(1, t_max):
        t_val = i
        print("\n")
        print("T value: %d" % t_val)
        # Training and prediction
        hypothesis, err = predict(
            X_train, X_test, y_train, y_test, t_val)
        erm_normal.append(err)
        erm_values = cross_validation_train(dataset, n_folds, t_val)
        erm_cross_val.append((sum(erm_values) / n_folds))

    x_values = list(range(1, t_max))
    plt.plot(x_values, erm_normal, label="Emperical Risk")
    plt.plot(x_values, erm_cross_val, label="Cross Validation")
    plt.xlabel('T')
    plt.ylabel('ERM')
    plt.title('T vs ERM')
    x_axis = list(set(x_values[:-1]))
    y_axis = list(set(erm_normal))
    plt.xticks(x_axis, rotation=90)
    plt.yticks(y_axis)
    plt.legend(loc=2)
    plt.show()
else:
    print("Invalid mode. Select 'erm' or 'nfold' or 'tsize'")
