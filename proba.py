import os
import random
from io import BytesIO

import numpy as np
import tensorflow as tf

BASE_DIR = "/tf/cats_and_dogs_filtered"

train_dir = os.path.join(BASE_DIR, "train")
validation_dir = os.path.join(BASE_DIR, "validation")

# Directory with training cat/dog pictures
train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

# Directory with validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

model = tf.keras.models.Sequential(
    [
        # Rescale the image. Note the input shape is the desired size of the image: 150x150 with 3 bytes for color
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        # Convolution and Pooling layers
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for one class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

# Instantiate the Dataset object for the training set
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(150, 150), batch_size=20, label_mode="binary"
)

# Instantiate the Dataset object for the validation set
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir, image_size=(150, 150), batch_size=20, label_mode="binary"
)

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE)

history = model.fit(train_dataset_final, epochs=15, validation_data=validation_dataset_final, verbose=2)
