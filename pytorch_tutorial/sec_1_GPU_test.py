import torch
import torch.nn as nn


class MultiLayerPercptron(torch.nn.Module):
    def __init__(self):
        super(MultiLayerPercptron, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        return

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    print('if GPU is available:', torch.cuda.is_available())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    x = torch.zeros([3, 2]).to(device)
    model = torch.nn.Linear(2, 2).to(device)
    y = model.forward(x)
    