import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim


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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ########## Set-up seed for reproducibility ##########
    seed = 0
    set_seed(seed)

    ########## Load dataset ##########
    X, y = load_data()

    ########## Split dataset into train and test ##########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ########## Move to PyTorch Tensor ##########
    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    print('X_train', X_train.size())
    print('y_train', y_train.size())
    print('X_test', X_test.size())
    print('y_test', y_test.size())
    print()

    ########## Set-up model ##########
    model = MultiLayerPercptron().to(device)
    print('model\n', model)
    print(sum(np.prod(param.shape) for param in model.parameters()))
    print()

    ########## Set-up optimization method ##########
    learning_rate = 3e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    ########## Set-up loss function ##########
    criterion = nn.CrossEntropyLoss()

    ########## Training ##########
    epochs = 200
    model.train()
    for e in range(epochs):
        ########## Get prediction ##########
        y_train_pred = model(X_train)
        ########## Get loss ##########
        loss = criterion(y_train_pred, y_train)

        ########## Clean-up gradients ##########
        optimizer.zero_grad()
        ########## Calculate gradients ##########
        loss.backward()
        ########## Update gradients ##########
        optimizer.step()

        ########## Show the loss at each epoch ##########
        print('Epoch: {}\tLoss: {:.5f}'.format(e, loss.item()))

    ########## Evaluation ##########
    model.eval()
    y_test_pred_output = model(X_test)
    _, y_test_pred = torch.max(y_test_pred_output, 1)
    acc = torch.true_divide(torch.sum(y_test_pred == y_test), y_test_pred.size()[0])
    print('accuracy: {}'.format(acc))
    print()
