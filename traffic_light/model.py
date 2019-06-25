import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def load_model(path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Model()

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])

        model.eval()
        model.to(device)

        return model


def save_model(model, optim, path):
    state_dict = model.module.state_dict()

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        # 'epoch': epoch,
        'state_dict': state_dict
        , 'optimizer': optim}
        , path)


def load_model(path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(device)

    return model


def main():
    print('This class is only callable, not executable')


if __name__ == '__main__':
    main()
