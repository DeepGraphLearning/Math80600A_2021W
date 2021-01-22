import argparse
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
    X = iris.data
    y = iris.target
    print('complete X: {}'.format(X.shape))
    print('complete y: {}'.format(y.shape))
    return X, y


class IrisDataset(data.Dataset):
    def __init__(self):
        X, y = load_data()
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


def train(model, dataloader, device):
    accum_loss = 0
    for batch in dataloader:
        X_train_batch, y_train_batch = batch
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)
        y_train_batch = y_train_batch.squeeze(1)

        y_train_batch_pred = model(X_train_batch)
        loss = criterion(y_train_batch_pred, y_train_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.detach()
    return accum_loss


def evaluate(model, dataloader, device):
    model.eval()
    y_test, y_test_pred = [], []
    for batch in dataloader:
        X_test_batch, y_test_batch = batch
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)
        y_test.append(y_test_batch)

        y_test_pred_batch = model(X_test_batch)
        y_test_pred.append(y_test_pred_batch)

    y_test = torch.cat(y_test, dim=0)
    y_test = y_test.squeeze(1)
    y_test_pred = torch.cat(y_test_pred, dim=0)

    _, y_test_pred = torch.max(y_test_pred, 1)
    acc = torch.true_divide(torch.sum(y_test_pred == y_test), y_test_pred.size()[0])
    print('accuracy: {}'.format(acc))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp'])
    args = parser.parse_args()
    print('The arguments are {}.'.format(args))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    set_seed(args.seed)

    dataset = IrisDataset()

    ########## Split dataset into train and test dataloader ##########
    N = len(dataset)
    split_index = int(N * 0.8)
    print('first {} data samples for training, last {} data samples for test, {} data samples in all.'.format(
        split_index, N - split_index, N))
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    train_sampler = data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)
    train_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

    ########## Choose model ##########
    if args.model == 'linear':
        model = nn.Linear(4, 3).to(device)
    elif args.model == 'mlp':
        model = MultiLayerPercptron().to(device)
    else:
        raise ValueError('Model {} is not included.'.format(args.model))
    print('model\n', model)
    print()

    ########## Choose optimizer ##########
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        raise ValueError('Optimizer {} is not included.'.format(args.optimizer))
    criterion = nn.CrossEntropyLoss()

    for e in range(args.epochs):
        accum_loss = train(model, train_dataloader, device)
        print('Epoch: {}\tLoss: {:.5f}'.format(e, accum_loss / len(train_dataloader)))

    evaluate(model, test_dataloader, device)