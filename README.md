# digits-ai
A digit recognition game. Play either the pygame desktop version or the online p5.js version.

# technical description
The desktop game was built using the pygame module and the online version was built using the p5.js library. Both allow the user to draw a number on the screen and then press guess. This feeds the image through a convolutional neural network(CNN) trained with the MNIST handwritten digit database and outputs a guess to the user. It isn't much of a game, but it is a live demo of a CNN in action.
# usage
pygame version: run `python game.py`

web version: `not currently available`

train network: `python mnist_trainer.py <checkpoint_file> <num_epochs>`

read number in image file: `number_reader.py <checkpoint_file> <image_file>`