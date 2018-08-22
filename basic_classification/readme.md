# Classification model
This is my implementation of neural network model to classify images of clothing accessories like sneakers and shirts. For now all of this has been done using keras which is a high level API for building and training models in tensorflow.

I've used the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) which contains 70,000 grayscale images in 10 categories.

60,000 images have been used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. 

Inspecting a random element, say the first element of the fashion-dataset, it's clear that the pixel values range from 0 to 255.

![alt text](https://i.imgur.com/NJaLPBV.png)

For verification of labelled data, I've displayed the first 16 images from the training set with associated class name below each image. 

![alt text](https://i.imgur.com/bH2qzcp.png)

# Building the model
Building requires configuring the layers of the model, then compiling the model.

## Setting up the layers
The first layer in this network transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. This layer has no parameters to learn; it only reformats the data.

After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes. The second (and last) layer is a 10-node **softmax** layer—this returns an array of 10 probability scores that sum to 1. Each node consists of a score that indicates the probability that the current image belongs to one of the 10 classes.

![softmax function](https://i.imgur.com/rPCqve9.png)

## Compiling the model
This includes addition of a loss function and an optimizer.


# Training, evaluation and testing
* At a *high-level*, training a network involves the following steps: 
  * Feeding the training images and labels as training data to the model
  * Model learns to associate the labels with images
  * Testing of model on a test set
 
 # Making predictions
 Following is a plot of several images with their predictions. Correct prediction labels are green ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) and incorrect prediction labels are red ![#f03c15](https://placehold.it/15/f03c15/000000?text=+).
 
![Plot_predictions](https://i.imgur.com/s5XkSPF.png)
 
