import numpy as np
import argparse
from csv import reader

eps = 1e-8


# Probability density function of multivariate_normal for d dimensions where d > 1
def pdf(x, mean, cov):
    cl = (1 / ((np.linalg.det(cov) + eps) ** 0.5) * ((np.pi * 2) ** (cov.shape[0] / 2)))
    x_mu = (x - mean)
    e = np.sqrt(np.sum(np.square(np.matmul(np.matmul(x_mu, np.linalg.pinv(cov)), x_mu.T)), axis=-1))
    ans = cl * np.exp((-1 / 2) * e)
    return ans


class GMM:
    def __init__(self, k=1):
        self.k = k
        # Initialise weight vector
        self.w = np.ones(k) / k
        self.means = None
        self.cov = None

    def fit(self, x):
        x = np.array(x)
        self.means = np.random.choice(x.flatten(), (self.k, x.shape[1]))

        cov = []
        for i in range(self.k):
            cov.append(np.cov(x, rowvar=False))
        cov = np.array(cov)

        for step in range(55):
            # Expectation step: estimating the values of latent variables
            probabilities = []
            for j in range(self.k):
                probabilities.append(pdf(x=x, mean=self.means[j], cov=cov[j]) + eps)
            probabilities = np.array(probabilities)

            # Maximization step: update mean, covariance and weights
            for j in range(self.k):
                # Bayes' Theorem
                b = ((probabilities[j] * self.w[j]) / (
                        np.sum([probabilities[i] * self.w[i] for i in range(self.k)], axis=0) + eps))
                # update mean, covariance and weights to maximize b
                self.means[j] = np.sum(b.reshape(len(x), 1) * x, axis=0) / (np.sum(b + eps))
                cov[j] = np.dot((b.reshape(len(x), 1) * (x - self.means[j])).T, (x - self.means[j])) / (
                        np.sum(b) + eps)
                self.w[j] = np.mean(b)

        self.cov = cov

    def prob(self, x):
        x = np.array(x)
        p = 0
        for j in range(self.k):
            # calculate probability of each component and add all of them
            p += self.w[j] * pdf(x=x, mean=self.means[j], cov=self.cov[j])
        return p


def load_csv(inp_file):
    dataset = []
    with open(inp_file, 'r') as file:
        csv_reader = reader(file)
        for data_row in csv_reader:
            dataset.append(data_row)
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument('--components', help='components 1|3|4', required=True)
parser.add_argument('--train', help='path to training data file', required=True)
parser.add_argument('--test', help='path to test data file', required=True)
args = vars(parser.parse_args())

k = int(args['components'])

train_data = load_csv(args['train'])

train_datas = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

# Split data digit wise
for row in train_data:
    y = int(row[64])
    train_datas[y].append([float(x.strip()) for x in row[:-1]])

gmms = []
# Train GMM for each digit giving is 10 probability functions
for i in range(10):
    gmm = GMM(k)
    gmm.fit(train_datas[i])
    gmms.append(gmm)

print('trained')

test_data = load_csv(args['test'])

preds = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

# Test
for row in test_data:
    y_act = int(row[64])
    max_p = float('-inf')
    y_pred = -1
    # Calculate the probability that the given x may correspond to a digit for all digits
    for idx in range(len(gmms)):
        p = gmms[idx].prob([float(x.strip()) for x in row[:-1]])
        # select the digit with maximum probability
        if np.sum(p) > max_p:
            y_pred = idx
            max_p = np.sum(p)
    if y_pred == -1:
        print('never')
    accu = 0
    if y_act == y_pred:
        accu = 1
    # Save prediction according to digit
    preds[y_act].append(accu)

total = 0
for idx in range(len(preds)):
    sum = np.sum(np.array(preds[idx]))
    print(f'{idx}: {(sum * 100) / len(preds[idx])}')
    total += sum

print(f'total: {(total / len(test_data)) * 100}')
