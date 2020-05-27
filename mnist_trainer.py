import numpy as np
from tqdm import tqdm

import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    '''
    convolutional neural network that takes in an image of a number and outputs
    a 1x10 vector. each element of the vector is the probability that the input
    image is the same as the index of that element
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetTrainer:
    '''
    class used to train and test the above defined network by epochs
    '''

    def __init__(self, loss_criterion, optimizer, net, train_loader, test_loader):
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader

    def check_some_out(self):
        it = iter(self.train_loader)
        imgs, labels = next(it)
        plt.imshow(imgs[0][0])
        plt.show()

    def train_epoch(self, epoch):
        # self.check_some_out()

        losses = []
        correct = 0
        total = 0

        for data in tqdm(self.train_loader):
            self.net.zero_grad()
            X, y = data
            X = X.double()  # X is [batch_size, 1, 28, 28]
            y = y.long()  # y is [batch_size]

            output = self.net(X)  # output is [batch_size, 10]
            loss = self.loss_criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            output = output.argmax(dim=1)

            for out, lbl in zip(output, y):
                if out == lbl:
                    correct += 1
                total += 1

        avg_loss = sum(losses) / total
        avg_correct = correct / total

        return avg_loss, avg_correct

    def test_epoch(self, epoch):

        running_loss = 0.0
        running_correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                X, y = data
                X = X.double()
                y = y.long()

                output = self.net(X)
                test_loss = self.loss_criterion(output, y)
                running_loss += test_loss.item()

                output = output.argmax(dim=1)

                for out, lbl in zip(output, y):
                    if out == lbl:
                        running_correct += 1
                    total += 1

        avg_loss = running_loss / total
        avg_correct = running_correct / total
        print('correct: ', avg_correct)
        print('loss: ', avg_loss)

        return avg_loss, avg_correct


'''
save all the info about the network at the specified path
'''


def save_checkpoint(path, epoch, net, loss_criterion, optimizer, train_loader, test_loader, loss, correct, eval_loss, eval_correct):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'loss_criterion': loss_criterion,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'loss': loss,
        'correct': correct,
        'eval_loss': eval_loss,
        'eval_correct': eval_correct,
    }, path)


'''
load a previously saved model
'''


def load_checkpoint(path, net, optimizer):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['loss_criterion'], checkpoint['train_loader'], checkpoint['test_loader'], checkpoint['epoch'], checkpoint['loss'], checkpoint['correct'], checkpoint['eval_loss'], checkpoint['eval_correct']


'''
download the mnist dataset from torchvision if not already downloaded
returns dataloaders over the dataset
'''


def get_data_loaders(bs=64):
    train = datasets.MNIST('', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ]))

    test = datasets.MNIST('', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ]))
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=bs, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=bs, drop_last=True, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':

    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage: mnist_trainer.py <checkpoint_file> <num_epochs>")
        exit(-1)

    experiment_name = sys.argv[1]
    N_epochs = int(sys.argv[2])

    # net with CrossEntropyLoss and Adam
    net = Net()
    net = net.double()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    checkpoint_fname = f"{experiment_name}.pt"

    # checkpoint already present
    if os.path.isfile(checkpoint_fname):
        loss_criterion, train_loader, test_loader, epoch, loss, correct, eval_loss, eval_correct = load_checkpoint(
            checkpoint_fname, net, optimizer)
    # no checkpoint present
    else:
        train_loader, test_loader = get_data_loaders()

        epoch = 0
        loss = []
        correct = []
        eval_loss = []
        eval_correct = []

    trainer = NetTrainer(loss_criterion, optimizer,
                         net, train_loader, test_loader)

    n_eval = 10
    n_save = 10

    for epoch in range(epoch, N_epochs+epoch):
        # train loop
        e_loss, e_correct = trainer.train_epoch(epoch)
        print(f'Epoch {epoch}: loss={e_loss}, correct={e_correct}')
        loss.append(e_loss)
        correct.append(e_correct)

        if not (epoch % n_eval):
            ev_loss, ev_correct = trainer.test_epoch(epoch)
            eval_loss.append(ev_loss)
            eval_correct.append(ev_correct)
            print(eval_correct)

        if not (epoch % n_save):
            save_checkpoint(checkpoint_fname, epoch, net, loss_criterion, optimizer,
                            train_loader, test_loader, loss, correct, eval_loss, eval_correct)

        epoch += 1

    ev_loss, ev_correct = trainer.test_epoch(epoch)
    eval_loss.append(ev_loss)
    eval_correct.append(ev_correct)
    save_checkpoint(checkpoint_fname, epoch, net, loss_criterion, optimizer,
                    train_loader, test_loader, loss, correct, eval_loss, eval_correct)
    torch.save(net.state_dict(), 'model.pt')
