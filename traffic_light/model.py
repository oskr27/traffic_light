import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, -1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def save_model(model, path, overwrite=False):
    return 0


if __name__ == '__main__':
    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)
    print(loss.grad_fn.next_functions[0][0])
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
