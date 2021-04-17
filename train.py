import math
import os

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from Net import Net
from PointDataset import PointDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

def fun(x):
    return math.sin(x)

def convert_arr(arr):
    res = [[i] for i in arr]
    return res

def train(model, dataset, epochs, lr, device=torch.device("cpu")):
    # track = convert_table_to_track('datasets/test.eep')
    # track = convert_table_to_track(path)

    # full_x, full_y = create_dataset(track, False)
    # full_x, full_y = create_big_dataset('datasets/tracks_mini')

    full_x, full_y = dataset

    full_x = torch.Tensor(full_x).to(device)
    full_y = torch.Tensor(full_y).to(device)

    full_dataset = PointDataset(full_x, full_y)

    train_size = int(0.6 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if os.path.isfile('model.pt'):
        model.load_state_dict(torch.load('model.pt'))

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # print(data, target)
            # print(data)
            output = model(data)
            # print(target)
            # calculate the loss

            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss
            valid_loss += loss.item() * data.size(0)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

    plt.clf()
    plt.ioff()
    plt.plot(list(range(0, epochs)), train_losses, label='train_loss')
    plt.plot(list(range(0, epochs)), valid_losses, label='valid_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(shadow=False, )
    # plt.gca().invert_xaxis()
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

net = Net().cuda(device)

# print(net)

if os.path.isfile('model.pt'):
    net.load_state_dict(torch.load('model.pt'))


x = np.arange(-math.pi+0.1, math.pi-0.05, math.pi/50).tolist()
y = [fun(i) for i in x]

dataset = (convert_arr(x), convert_arr(y))

# print(dataset)

train(model=net, dataset=dataset, epochs=500, lr=1e-3, device=device)


print(len(x),len(y))
plt.plot(x, y, label='Original')
plt.xlabel('X')
plt.ylabel('Y')

predicted = []

# net.to(torch.device("cpu"))
net.eval()
for i in x:
    tmp = Tensor([i]).to(device)
    pred = net(tmp).tolist()
    predicted.append(pred)

plt.plot(x, predicted, label = 'Predicted')
plt.legend(shadow=False, )
plt.show()