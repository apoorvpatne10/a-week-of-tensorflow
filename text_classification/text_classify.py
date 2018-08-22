# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Exploring the data
# Data is already in preprocessed form such that each review is in the form of sequence of integers
print("Training entries : {}, labels : {}".format(len(train_data), len(train_labels)))

print(train_data[0], train_data[1], sep='\n')
#import pandas as pd
#x = pd.DataFrame(train_data)

# Converting integers back to words
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"]  = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

# Preparing the data
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value = word_index["<PAD>"],
                                                        padding = 'post',
                                                        maxlen = 256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value = word_index["<PAD>"],
                                                        padding = 'post',
                                                        maxlen = 256)

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# Loss function and optimizer 
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Creating a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Training the model
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512,
                    validation_data=(x_val, y_val), verbose=1)

# Evaluating the model on test set
res = model.evaluate(test_data, test_labels)
print(res)

# Creating a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc_value = history_dict['acc']
val_acc_value = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()