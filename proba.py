import os
import random
from io import BytesIO

import numpy as np
import tensorflow as tf

BASE_DIR = "./rps"

rock_dir = os.path.join(BASE_DIR, "rock")
paper_dir = os.path.join(BASE_DIR, "paper")
scissors_dir = os.path.join(BASE_DIR, "scissors")

print(f"total training rock images: {len(os.listdir(rock_dir))}")
print(f"total training paper images: {len(os.listdir(paper_dir))}")
print(f"total training scissors images: {len(os.listdir(scissors_dir))}")

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

print()
print(f"5 files in the rock subdir: {rock_files[:5]}")
print(f"5 files in the paper subdir: {paper_files[:5]}")
print(f"5 files in the scissors subdir: {scissors_files[:5]}")

model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        # Rescale the image. Note the input shape is the desired size of the image: 150x150 with 3 bytes for color
        tf.keras.layers.Rescaling(1.0 / 255),
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Print the model summary
model.summary()

TRAINING_DIR = "./rps"
VALIDATION_DIR = "./rps-test-set"

# Instantiate the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DIR, image_size=(150, 150), batch_size=32, label_mode="categorical"
)

# Instantiate the validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR, image_size=(150, 150), batch_size=32, label_mode="categorical"
)

# Optimize the datasets for training
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE)

# Create a model with data augmentation layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

# Attach the data augmentation model to the base model
model_with_aug = tf.keras.models.Sequential([data_augmentation, model])

# Set the training parameters
model_with_aug.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Train the model
history = model_with_aug.fit(train_dataset_final, epochs=25, validation_data=validation_dataset_final, verbose=1)
