from PIL import Image, ImageFilter
import numpy as np
from mnist_trainer import Net, load_checkpoint
import torch
import torch.nn as nn


class Reader:

    def __init__(self, checkpoint, image):
        # open image and turn it black and white
        img = Image.open(image).convert('L')
        size = img.size[0]
        box_size = size // 28

        # decrease resolution to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS)
        self.img_vector = torch.zeros((28, 28), dtype=torch.double)
        for i in range(28):
            for j in range(28):
                self.img_vector[i][j] = img.getpixel((j, i))

        # plt.imshow(self.img_vector, cmap='gray')
        # plt.show()

        self.img_vector = self.img_vector.view(-1, 1, 28, 28)

        # net with CrossEntropyLoss and Adam
        net = Net()
        net = net.double()
        net.load_state_dict(torch.load(
            checkpoint, map_location=torch.device('cpu')))
        # net = net.double()
        # loss_criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        # loss_criterion, train_loader, test_loader, epoch, loss, correct, eval_loss, eval_correct = load_checkpoint(
        #     checkpoint, net, optimizer)

        self.net = net

    def read(self):
        output = self.net(self.img_vector)
        # print(output)
        guess = output.argmax(dim=1)
        return int(guess)


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage: number_reader.py <checkpoint_file> <image_file>")
        exit(-1)

    checkpoint = sys.argv[1]
    image = sys.argv[2]

    if not os.path.isfile(checkpoint):
        print("no checkpoint file, run mnist_trainer.py <checkpoint_file> <num_epochs>")
        exit(-1)
    elif not os.path.isfile(image):
        print("no image file")
        exit(-1)

    reader = Reader(checkpoint, image)
    guess = reader.read()
    print(guess)
