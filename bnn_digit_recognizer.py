import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import sklearn as skl

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     Flatten, Dropout, DepthwiseConv2D, SeparableConv2D,
                                     Activation, BatchNormalization, SpatialDropout2D,
                                     concatenate)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

tfd = tfp.distributions
tfpl = tfp.layers
# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode labels
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/60000.

def create_bnn():
    model = Sequential([
    Flatten(input_shape=(28,28)),
    tfpl.DenseFlipout(784,
                         activation='relu',
                         kernel_divergence_fn=divergence,
                         bias_divergence_fn=divergence, ),

    Dropout(0.25),
    tfpl.DenseFlipout(512,
                         activation='relu',
                         kernel_divergence_fn=divergence,
                         bias_divergence_fn=divergence, ),

    Dropout(0.25),
    tfpl.DenseFlipout(10,
                      activation='relu',
                      kernel_divergence_fn=divergence,
                      bias_divergence_fn=divergence),

    tfpl.OneHotCategorical(10,convert_to_tensor_fn=tfd.Distribution.mode)
    ])
    return model

model = create_bnn()
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=neg_loglike,
              metrics=['accuracy'],
              experimental_run_tf_function=False
              )

print(model.summary())


earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    start_from_epoch=50,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor = 0.95,
    patience=5,
    cooldown=5,
    mode='max',
    min_lr=1e-7,
    verbose=1,
    min_delta=0.01)

mdlhist = model.fit(train_images,
                    train_labels,
                    batch_size=256,
                    epochs=100,
                    validation_data=(test_images, test_labels),
                    callbacks=[earlystop, reduce_lr])

model.save('/home/lreclusa/repositories/BNN-Digits-Recognizer-App/mnist_bnn')