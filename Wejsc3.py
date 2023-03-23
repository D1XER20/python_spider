import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.datasets import fashion_mnist
train, test = tf.keras.datasets.fashion_mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
class_count = np.unique(y_train).shape[0]

model = Sequential(6)
filter_cnt = 32
units = 32
learning_rate = 0.0001
kernel_size = (3,3)
pooling_size = (2,2)
conv_rule = 'same'


model.add(MaxPooling2D(pooling_size))
repeat_num = 2
model.add(Conv2D(input_shape = X_train.shape[1:],
                 filters=filter_cnt,
                 kernel_size = kernel_size,
                 padding = conv_rule, activation = act_func))
repeat_num = 2
model.add(Dense(class_count, activation='softmax'))
repeat_num = 2
model.compile(optimizer=Adam(learning_rate),
              loss='SparseCategoricalCrossentropy',
              metrics='accuracy')
history = model.fit(X_train, y_train, batch_size = 8, epochs=5, validation_data = ())
acc=max(model.history.history['val_accuracy'])
