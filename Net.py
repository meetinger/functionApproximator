from torch import nn, Tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            # nn.Tanh(),

            nn.Linear(10, 10),
            nn.Tanh(),
            # nn.Dropout(p=0.2),

            nn.Linear(10, 1),
        )
        # self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)



