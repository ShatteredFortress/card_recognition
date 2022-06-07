# Alec Diefenbaugh & Austin Britton
# CS 445
# Card Recognition Project CNN

import tensorflow as tf

# import pathlib
# import os
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np  # for vector & matrix math operations, also has its own random number generator
# import pandas as pd  # pandas loading of csv is much quicker than np.loadtext! (5 sec vs 38 sec)
# import matplotlib.pyplot as plt  # for displaying graphs of accuracy per epoch
# import seaborn # for displaying confusion matrices
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
    batch_size=3
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
model.fit(trainDS, epochs=10, verbose=2)
model.evaluate(valDS, verbose=2)

y_pred=model.predict_classes(valDS)
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

print(con_mat_df)
#I cannot download matplotlib on my linux machine So i am hoping someone else can test the actual heat map
'''
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

'''