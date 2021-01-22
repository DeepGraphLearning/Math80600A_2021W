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


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('device\t', device)

    ########## Set-up seed for reproducibility ##########
    seed = 0
    set_seed(seed)

    ########## Load dataset ##########
    X, y = load_data()

    ########## Split dataset into train and test ##########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ########## Wrap-up data with dataset ##########
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)
    ########## Set-up dataloader ##########
    train_dataloader = data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    ########## Set-up model ##########
    model = MultiLayerPercptron().to(device)
    print('model\n', model)
    print()

    ########## Set-up optimization method ##########
    learning_rate = 3e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    ########## Set-up loss function ##########
    criterion = nn.CrossEntropyLoss()

    ########## Training ##########
    model.train()
    epochs = 200
    for e in range(epochs):
        accum_loss = 0
        for batch in train_dataloader:
            X_train_batch, y_train_batch = batch
            X_train_batch = X_train_batch.to(device)  # size: (batch_size, 4)
            y_train_batch = y_train_batch.to(device)  # size: (batch_size, 1)
            y_train_batch = y_train_batch.squeeze(1)  # size: (batch_size)

            ########## Get prediction ##########
            y_train_batch_pred = model(X_train_batch)
            ########## Get loss ##########
            loss = criterion(y_train_batch_pred, y_train_batch)

            ########## Clean-up gradients from previous steps ##########
            optimizer.zero_grad()
            ########## Calculate gradients for current step ##########
            loss.backward()
            ########## Update weights using SGD for current step ##########
            optimizer.step()

            accum_loss += loss.detach()

        ########## Show the accumulated loss at each epoch ##########
        print('Epoch: {}\tLoss: {:.5f}'.format(e, accum_loss/len(train_dataloader)))

    ########## Evaluation ##########
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
    print('accuracy: {}'.format(acc))
    print()
