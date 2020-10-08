import os

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.nn import ReLU, Sequential, Conv2d, Module, BatchNorm2d, MaxPool2d, ConvTranspose2d, Sigmoid
from torch.optim import SGD
from torch.utils.data import DataLoader

from DidaDataset import DidaDataset
from constants import MOMENTUM_SGD, LR_SGD, WD_SGD, EPS, BATCH, EPOCHS


class FCN(Module):

    def __init__(self, name, in_size, device):
        super(FCN, self).__init__()

        assert (in_size % 16 == 0)

        self.name = name
        self.in_size = in_size
        self.device = device

        self.convBlock1 = Sequential(
            Conv2d(in_channels=3, kernel_size=5, out_channels=32, stride=2, padding=2),
            BatchNorm2d(num_features=32, momentum=0.1),
            ReLU(inplace=True),

            Conv2d(in_channels=32, kernel_size=3, out_channels=32, stride=1, padding=1),
            BatchNorm2d(num_features=32, momentum=0.1),
            ReLU(inplace=True)
        )

        self.upsampling1 = ConvTranspose2d(in_channels=32, kernel_size=int(self.in_size / 2) + 1, out_channels=1,
                                           stride=1,
                                           padding=0)

        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.convBlock2 = Sequential(
            Conv2d(in_channels=32, kernel_size=3, out_channels=64, stride=1, padding=1),
            BatchNorm2d(num_features=64, momentum=0.1),
            ReLU(inplace=True),

            Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding=1),
            BatchNorm2d(num_features=64, momentum=0.1),
            ReLU(inplace=True)
        )

        self.upsampling2 = ConvTranspose2d(in_channels=64, kernel_size=3 * int(self.in_size / 4) + 1, out_channels=1,
                                           stride=1, padding=0)

        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.convBlock3 = Sequential(
            Conv2d(in_channels=64, kernel_size=3, out_channels=96, stride=1, padding=1),
            BatchNorm2d(num_features=96, momentum=0.1),
            ReLU(inplace=True),

            Conv2d(in_channels=96, kernel_size=3, out_channels=96, stride=1, padding=1),
            BatchNorm2d(num_features=96, momentum=0.1),
            ReLU(inplace=True)
        )

        self.upsampling3 = ConvTranspose2d(in_channels=96, kernel_size=7 * int(self.in_size / 8) + 1, out_channels=1,
                                           stride=1, padding=0)

        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        self.convBlock4 = Sequential(
            Conv2d(in_channels=96, kernel_size=3, out_channels=128, stride=1, padding=1),
            BatchNorm2d(num_features=128, momentum=0.1),
            ReLU(inplace=True),

            Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding=1),
            BatchNorm2d(num_features=128, momentum=0.1),
            ReLU(inplace=True)
        )

        self.upsampling4 = ConvTranspose2d(in_channels=128, kernel_size=15 * int(self.in_size / 16) + 1, out_channels=1,
                                           stride=1, padding=0)

        self.convScore = Sequential(
            Conv2d(in_channels=4, kernel_size=1, out_channels=1, stride=1, padding=0),
            Sigmoid()
        )

        self = self.to(device)

        self.optimizer = SGD(self.parameters(), lr=LR_SGD, momentum=MOMENTUM_SGD,
                             nesterov=True, weight_decay=WD_SGD)

    def forward(self, x):
        x1_ = self.convBlock1(x)
        x1f = torch.squeeze(self.upsampling1(x1_))
        x1 = self.pool1(x1_)

        x2_ = self.convBlock2(x1)
        x2f = torch.squeeze(self.upsampling2(x2_))
        x2 = self.pool2(x2_)

        x3_ = self.convBlock3(x2)
        x3f = torch.squeeze(self.upsampling3(x3_))
        x3 = self.pool3(x3_)

        x4_ = self.convBlock4(x3)
        x4f = torch.squeeze(self.upsampling4(x4_))

        return self.convScore(torch.stack([x1f, x2f, x3f, x4f]).permute(1, 0, 2, 3))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x1 = torch.stack([x])
            # converting the data into GPU format (if available)
            x1 = x1.to(self.device)
            return self(x1)[0].numpy()

    def train_batch(self, x_train, y_train):

        self.train()
        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        # ========forward pass=====================================
        y_output = self(x_train)

        dims = list(y_output.size())
        res = torch.tensor(0.)

        for i in range(dims[0]):
            for j in range(dims[2]):
                for k in range(dims[3]):
                    res = res - y_train[i][0][j][k] * torch.log(y_output[i][0][j][k] + EPS)
                    res = res - (1 - y_train[i][0][j][k]) * torch.log(1 - y_output[i][0][j][k] + EPS)

        loss_train = res / (1. * dims[0] * dims[1] * dims[2])

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()

        return loss_train.item(), (torch.round(y_output) == y_train).float().sum().item()

    def validate(self, x, y):

        self.eval()
        with torch.no_grad():
            y_output = self(x)

            dims = list(y_output.size())
            res = torch.tensor(0.)

            for i in range(dims[0]):
                for j in range(dims[2]):
                    for k in range(dims[3]):
                        res = res - y[i][0][j][k] * torch.log(y_output[i][0][j][k] + EPS)
                        res = res - (1 - y[i][0][j][k]) * torch.log(1 - y_output[i][0][j][k] + EPS)

            loss_val = res / (1. * dims[0] * dims[1] * dims[2])

            return loss_val.item(), (torch.round(y_output) == y).float().sum().item()

    def train_model(self, X, Y, workpath):
        sim_data = []
        for i in range(len(X)):
            sim_data.append([X[i], Y[i]])
        trainset, testset = train_test_split(sim_data, test_size=0.2)

        trainloader = DataLoader(dataset=DidaDataset(trainset), batch_size=BATCH, shuffle=True)
        testloader = DataLoader(dataset=DidaDataset(testset), batch_size=BATCH, shuffle=True)

        train_losses, val_losses = [], []
        train_acc, val_acc = [], []

        # training the model
        for ep in range(EPOCHS):
            print("Epoch #", ep)
            loss_train, loss_val = 0, 0
            acc_train, acc_val = 0, 0
            div_train = 0
            total_train = 0
            for batch_idx, (x, y) in enumerate(trainloader):
                print("   Batch #", batch_idx)
                x, y = x.to(self.device).requires_grad_(), y.to(self.device).requires_grad_()
                div_train += 1
                total_train += x.size(0)
                loss_train_, acc_train_ = self.train_batch(x, y)
                loss_train += loss_train_
                acc_train += acc_train_

            div_val = 0
            total_val = 0
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to(self.device).requires_grad_(False), y.to(self.device).requires_grad_(False)
                div_val += 1
                total_val += x.size(0)
                loss_val_, acc_val_ = self.validate(x, y)
                loss_val += loss_val_
                acc_val += acc_val_

            train_losses.append(loss_train / div_train)
            val_losses.append(loss_val / div_val)

            train_acc.append(acc_train / total_train)
            val_acc.append(acc_val / total_val)

            print("Loss train:", loss_train / div_train)
            print("Loss valid:", loss_val / div_val)
            print("Accu train:", acc_train / total_train)
            print("Accu valid:", acc_val / total_val)

            torch.save(self.state_dict(), os.path.join(workpath, "models", "dida_model_" + ep + ".th"))

        plt.plot(train_losses, label='Loss (training data)')
        plt.plot(val_losses, label='Loss (validation data)')
        plt.title('Loss functions')
        plt.ylabel('Loss')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(train_acc, label='Accuracy (training data)')
        plt.plot(val_acc, label='Accuracy (validation data)')
        plt.title('Accuracy functions')
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
