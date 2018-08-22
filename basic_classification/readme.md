# Classification model
This is my implementation of neural network model to classify images of clothing accessories like sneakers and shirts. For now all of this has been done using keras which is a high level API for building and training models in tensorflow.

I've used the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) which contains 70,000 grayscale images in 10 categories.

60,000 images have been used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. 

Inspecting a random element, say the first element of the fashion-dataset, it's clear that the pixel values range from 0 to 255.

![alt text](https://i.imgur.com/NJaLPBV.png)
