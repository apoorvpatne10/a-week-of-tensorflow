# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

# Preprocessing the data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
# Building the models

# Setting up the layers
model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

# Compiling the model
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10)
    
# Evaluating accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy :', test_acc)

# Make predictions
prediction = model.predict(test_images) 

prediction[0]

np.argmax(prediction[0])

test_labels[0]

# Plot several images with corressponding predictions
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
                                color=color)

img = test_images[0]
print(img.shape)    

img = (np.expand_dims(img, 0))
print(img.shape)

predictions = model.predict(img)
print(np.argmax(predictions))
