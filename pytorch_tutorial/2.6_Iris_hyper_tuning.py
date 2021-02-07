import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import sklearn
import matplotlib.pyplot as plt
import numpy as np

from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

import tensorboardX


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def load_data():
    iris = datasets.load_iris()
    X = iris.data  # size: (150 * 4)
    y = iris.target  # size: (150)
    print('complete X: {}'.format(X.shape))
    print('complete y: {}'.format(y.shape))
    return X, y


class IrisDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X_sample = torch.FloatTensor(self.X[index])
        y_sample = torch.LongTensor([self.y[index]])
        return X_sample, y_sample


class MultiLayerPercptron(nn.Module):
    def __init__(self):
        super(MultiLayerPercptron, self).__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )
        return

    def forward(self, x):
        x = self.mlp_layer(x)
        return x


def train_and_eval(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed = 0
    set_seed(seed)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)  # we would call this as the validation dataset in Cross-Validation
    train_dataloader = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = MultiLayerPercptron().to(device)

    learning_rate = config['lr']
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = config['epochs']
    for e in range(epochs):
        accum_loss = 0
        for batch in train_dataloader:
            X_train_batch, y_train_batch = batch
            X_train_batch = X_train_batch.to(device)  # size: (batch_size, 4)
            y_train_batch = y_train_batch.to(device)  # size: (batch_size, 1)
            y_train_batch = y_train_batch.squeeze(1)  # size: (batch_size)

            y_train_batch_pred = model(X_train_batch)
            loss = criterion(y_train_batch_pred, y_train_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()

    model.eval()
    y_test, y_test_pred = [], []
    for batch in test_dataloader:
        X_test_batch, y_test_batch = batch
        X_test_batch = X_test_batch.to(device)  # size: (batch_size, 4)
        y_test_batch = y_test_batch.to(device)  # size: (batch_size, 1)
        y_test.append(y_test_batch)

        y_test_pred_batch = model(X_test_batch)  # size: (batch_size, 1)
        y_test_pred.append(y_test_pred_batch)

    y_test = torch.cat(y_test, dim=0)  # size: (30, 1)
    y_test = y_test.squeeze(1)  # size: (30)
    y_test_pred = torch.cat(y_test_pred, dim=0)  # (30, 3)

    _, y_test_pred = torch.max(y_test_pred, 1)
    acc = torch.true_divide(torch.sum(y_test_pred == y_test), y_test_pred.size()[0])
    tune.report(accuracy=acc)
    return


if __name__ == '__main__':
    config = {
        'lr': tune.grid_search([0.1, 0.01]),
        'epochs': tune.grid_search([1, 5, 10])
    }

    analysis = tune.run(train_and_eval, config=config)
    print('Best config: ', analysis.get_best_config(metric='accuracy', mode='max'))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()