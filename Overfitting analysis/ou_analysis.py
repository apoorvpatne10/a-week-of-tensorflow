# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

numWords = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=numWords)
#df = pd.DataFrame(train_data)
#df = enumerate(df)
def multi_hot_sequences(seq, dimension):
    res = np.zeros((len(seq), dimension))
    for i, indices in enumerate(seq):
        res[i, indices] = 1.0
    return res

train_data = multi_hot_sequences(train_data, dimension=numWords)
test_data = multi_hot_sequences(test_data, dimension=numWords)

plt.plot(train_data[0])

# Baseline model

baseline_model = keras.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(numWords,)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

baseline_model.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# Smaller model

smaller_model = keras.Sequential([
            keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(numWords,)),
            keras.layers.Dense(4, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
    
smaller_model.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# bigger model

bigger_model = keras.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(numWords,)),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
    
bigger_model.compile(optimizer = 'adam',
                     loss = 'binary_crossentropy',
                     metrics = ['accuracy', 'binary_crossentropy'])

bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs = 20,
                                  batch_size = 512,
                                  validation_data = (test_data, test_labels),
                                  verbose = 2)

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))
    
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--',
                       label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(),
                 label = name.title() + ' Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    
    plt.xlim([0, max(history.epoch)])

plot_history([('baseline', baseline_history), 
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

# Weight regularization
l2_model = keras.models.Sequential([
            keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                               activation=tf.nn.relu, input_shape=(numWords,)),
            keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])    

l2_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data,
                                train_labels,
                                epochs = 20,
                                batch_size = 512,
                                validation_data = (test_data, test_labels),
                                verbose = 2)

plot_history([('baseline', baseline_history),
              ('l2_model', l2_model_history)])

# Addition of drop-out
drop_model = keras.models.Sequential([
            keras.layers.Dense(16, activation=tf.nn.relu,
                               input_shape = (numWords,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(16, activation = tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

drop_model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy', 'binary_crossentropy'])

drop_model_history = drop_model.fit(train_data,
                                    train_labels,
                                    epochs = 20,
                                    batch_size = 512,
                                    validation_data = (test_data, test_labels),
                                    verbose = 2)
    
plot_history([('baseline', baseline_history),
              ('dropout', drop_model_history)])

    
