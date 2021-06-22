import numpy as np
from numpy import linalg
import argparse
from csv import reader
from random import seed


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def radial_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM:
    def __init__(self, kernel=linear_kernel, C=0.1):
        self.kernel = kernel
        self.C = C
        self.learning_rate = 0.001
        self.iterations = 1000

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y == 0, -1, 1)
        K = np.zeros((n_samples, n_samples))

        alpha = np.zeros(n_samples)
        b = 0
        for t in range(1, self.iterations + 1):
            i = np.random.randint(0, n_samples)
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
            condition = y[i] * np.sum(alpha * K[i, :]) < 1
            if condition:
                alpha += self.learning_rate * alpha + y[i]
            else:
                b += self.learning_rate * y[i]

        idx = np.logical_and(alpha > 0, alpha < self.C)
        self.X = X[idx.reshape(1, -1)[0], :]
        self.y = y[idx.reshape(1, -1)[0]]
        self.b = b
        self.alpha = alpha[idx.reshape(1, -1)[0]]
        self.w = np.transpose(np.matmul(np.transpose(alpha * y), X))

    def predict(self, X):
        n_samples, n_features = X.shape

        p = np.zeros(n_samples)
        for i in range(n_samples):
            prediction = 0
            for j in range(self.X.shape[0]):
                prediction = prediction + self.alpha[j] * self.y[j] * self.kernel(np.transpose(X[i, :]), np.transpose(self.X[j, :]))

            p[i] = prediction + self.b

        return np.where(p <= 0, 0, 1)

def erm(y_pred, y_true):
    c = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            c += 1
    return 1 - c / len(y_true)


def load_csv(inp_file):
    dataset = []
    with open(inp_file, 'r') as file:
        csv_reader = reader(file)
        for data_row in csv_reader:
            dataset.append(data_row)
    return dataset


# Split in to X,Y
def split_X_Y(dataset):
    x = []
    y = []
    for row in dataset:
        x.append([p for p in row[:-1]])
        y.append(int(row[-1]))
    return np.array(x), np.array(y)


def split_x_y_mnist(d):
    x = []
    y = []
    for r in d:
        x.append([p / 255.0 for p in r[1:]])
        y.append(int(r[0]))
    return np.array(x), np.array(y)


binary = {
    0: '0000',
    1: '0001',
    2: '0010',
    3: '0011',
    4: '0100',
    5: '0101',
    6: '0110',
    7: '0111',
    8: '1000',
    9: '1001',
}

sb = {
    '0000': 0,
    '0001': 1,
    '0010': 2,
    '0011': 3,
    '0100': 4,
    '0101': 5,
    '0110': 6,
    '0111': 7,
    '1000': 8,
    '1001': 9,
}


def split_yy_mnist(mainy):
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    for idx in range(len(mainy)):
        a = binary[mainy[idx]]
        y0.append(int(a[0]))
        y1.append(int(a[1]))
        y2.append(int(a[2]))
        y3.append(int(a[3]))
    return np.array([np.array(y0), np.array(y1), np.array(y2), np.array(y3)])


def merge_Y_MNIST(yy):
    res = np.where(yy[0] == 0, '0', '1')
    for i in range(1, len(yy)):
        res = np.char.add(res, np.where(yy[i] == 0, '0', '1'))
    er = []
    for i in range(len(res)):
        er.append(sb[res[i]])
    return np.array(er)

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


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--kernel', help='kernel function to use, linear|rbf', required=True)
parser.add_argument('--dataset', help='Dataset mnist|bcd', required=True)
parser.add_argument('--train', help='path to training data file', required=True)
parser.add_argument('--test', help='path to test data file', required=True)
parser.add_argument('--output', help='path to output weight vector', required=True)
args = vars(parser.parse_args())
seed(1)
output = open(args['output'], 'w')
# get the dataset file
data_file = args['train']
# Load the input CSV file
dataset = load_csv(data_file)

kf = linear_kernel
if args['kernel'] == 'linear':
    kf = linear_kernel
elif args['kernel'] == 'rbf':
    kf = radial_kernel
else:
    print('invalid kernel value passed defaulting to linear')

dataset = dataset[1:]
i = 0
for row in dataset:
    dataset[i] = [float(x.strip()) for x in row]
    i += 1
ds = args['dataset']
if ds == 'mnist':
    X, y = split_x_y_mnist(dataset)
    yy = split_yy_mnist(y)
    print('started training')
    svms = []
    for i in range(len(yy)):
        svm = SVM(kernel=kf)
        svm.fit(X, yy[i])
        if svm.w is not None:
            print(svm.w, file=output)
        svms.append(svm)

    test_data_set = load_csv(args['train'])
    test_data_set = test_data_set[1:]
    i = 0
    for row in test_data_set:
        test_data_set[i] = [float(x.strip()) for x in row]
        i += 1

    X_test, Y_test = split_x_y_mnist(test_data_set)
    yy_pred = []
    for i in range(len(svms)):
        yy_pred.append(svms[i].predict(X_test))
    y_pred = merge_Y_MNIST(yy_pred)
    print('error')
    print(erm(y_pred, Y_test))
elif ds == 'bcd':
    svm = SVM(kernel=kf)
    X, Y = split_X_Y(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    svm.fit(X_train, y_train)
    if svm.w is not None:
        print(svm.w, file=output)
    y_pred = svm.predict(X_test)
    print('error')
    print(erm(y_pred, y_test))
else:
    print('unknown dataset passed possible values are mnist and bcd')

output.close()
