# Alec & Austin
# CS 445/545
# Card Recognition Project CNN

import tensorflow as tf
import numpy as np  # for vector & matrix math operations, also has its own random number generator
import pandas as pd  # pandas loading of csv is much quicker than np.loadtext! (5 sec vs 38 sec)
import matplotlib.pyplot as plt  # for displaying graphs of accuracy per epoch
import seaborn # for displaying confusion matrices

# import pathlib
# import os
# from tensorflow import keras
# from tensorflow.keras import layers
# import PIL
# import PIL.Image
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# np.set_printoptions(precision=4)

path = "D:/Homework/CS 445 Machine Learning/_Project - Card Recognition/Images"

trainDS = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(40, 40),
    batch_size=3
)

valDS = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(40, 40),
    # batch_size=3
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1./255))
model.add(tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), padding="valid",
            activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(30, 3, strides=(1,1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(70, activation='relu'))
model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dense(53))

# model.build(input_shape=(864,1152,3))
# print(model.summary())
# import sys; sys.exit()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]
model.compile(optimizer=optim, loss=loss, metrics=metrics)
history = model.fit(trainDS, epochs=50, verbose=2, validation_data=valDS)
model.evaluate(valDS, verbose=2)
accuracy = history.history['val_accuracy']


# Stuff to generate confusion matrix & accuracy per epoch plot:

preds = []
true = []
for images, labels in valDS:
    preds.append(tf.argmax(model.predict(images), axis=1))
    true.append(labels)
preds = np.concatenate(preds)
true = np.concatenate(true)

classes = trainDS.class_names
con_mat = tf.math.confusion_matrix(labels=true, predictions=preds).numpy()

fig1 = plt.figure()
plt.title("Test Data Confusion Matrix")
seaborn.set(font_scale=0.7)
seaborn.heatmap(con_mat, cmap="Blues", linewidth=0.5, annot=True, xticklabels=classes, yticklabels=classes)
plt.tight_layout()
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

fig2 = plt.figure()
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.plot(accuracy,'b',marker='.')

plt.show(block=True)
