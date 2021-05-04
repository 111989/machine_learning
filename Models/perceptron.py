import numpy as np
from csv import reader
from random import randrange
from random import seed
import argparse


class Perceptron:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        len_w = X.shape[1] + 1
        w = np.zeros(len_w)
        w = w.reshape(len_w, 1)
        run_count = 0
        max_runs = 100

        properly_classified = False
        while properly_classified == False and run_count < max_runs:
            properly_classified = True
            for i in range(len(X)):
                curr_y = y[i]
                curr_row = X[i]
                curr_row = np.insert(curr_row, 0, 1)
                curr_row = curr_row.reshape(1, len_w)
                if((np.dot(curr_row, w)) * curr_y) <= 0:
                    w += np.dot(curr_row, curr_y).reshape(len_w, 1)
                    properly_classified = False
            run_count += 1

        self.weights = w.reshape(len_w)

        return self.weights

    def predict(self, X):
        len_w = X.shape[1] + 1
        w = self.weights.reshape(len_w, 1)
        y = []
        for i in range(len(X)):
            curr_row = X[i]
            curr_row = np.insert(curr_row, 0, 1)
            curr_row = curr_row.reshape(1, len_w)
            y.append(np.dot(curr_row, w))

        return np.array(y)

# Emperical risk minimization


def erm(y_pred, y_true):
    err = 0
    for idx in range(len(y_true)):
        if y_pred[idx] * y_true[idx] <= 0:
            err += 1
    return err / len(y_true)


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
    if shuffle:
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


def predict(X_train, X_test, y_train, y_test, n_iters=1):
    model = Perceptron()
    # training
    hypothesis = model.fit(X_train, y_train)
    # prediction
    y_pred = model.predict(X_test)
    return hypothesis, erm(y_pred, y_test)


def cross_validation_train(dataset, n_folds=10):
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
            X_train, X_test, y_train, y_test, 100)
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


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--mode', help='erm or nfold', required=True)
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    hypothesis, err = predict(X, X, y, y)
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
else:
    print("Invalid mode. Select 'erm' or 'nfold'")
