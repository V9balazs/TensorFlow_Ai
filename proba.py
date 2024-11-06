import os
import random
from io import BytesIO

import numpy as np
import tensorflow as tf

# Only an example, not a real path
TRAIN_DIR = "c:/Users/micha/Documents/GitHub/tensorflow-training/horse-or-human"
VAL_DIR = "c:/Users/micha/Documents/GitHub/tensorflow-validation/horse-or-human"

# Directory with training horse pictures
train_horse_dir = os.path.join(TRAIN_DIR, "horses")
# Directory with training human pictures
train_human_dir = os.path.join(TRAIN_DIR, "humans")

# Directory with validation horse pictures
validation_horse_dir = os.path.join(VAL_DIR, "horses")
# Directory with validation human pictures
validation_human_dir = os.path.join(VAL_DIR, "humans")

# Check the filenames
train_horse_names = os.listdir(train_horse_dir)
print(f"5 files in horses subdir: {train_horse_names[:5]}")
train_human_names = os.listdir(train_human_dir)
print(f"5 files in humans subdir:{train_human_names[:5]}")

print(f"total training horse images: {len(os.listdir(train_horse_dir))}")
print(f"total training human images: {len(os.listdir(train_human_dir))}")

model = tf.keras.models.Sequential(
    [
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.Input(shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        # Only 1 output neuron. It will contain a value from 0 to 1 where 0 is for 'horses' and 1 for 'humans'
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    loss="binary_crossentropy", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=["accuracy"]
)

# Instantiate the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(300, 300), batch_size=32, label_mode="binary"
)

# Instantiate the validation set
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=(300, 300), batch_size=32, label_mode="binary"
)

# Check the type
dataset_type = type(train_dataset)
print(f"train_dataset inherits from tf.data.Dataset: {issubclass(dataset_type, tf.data.Dataset)}")

# Get one batch from the dataset
sample_batch = list(train_dataset.take(1))[0]

# Check that the output is a pair
print(f"sample batch data type: {type(sample_batch)}")
print(f"number of elements: {len(sample_batch)}")

# Extract image and label
image_batch = sample_batch[0]
label_batch = sample_batch[1]

# Check the shapes
print(f"image batch shape: {image_batch.shape}")
print(f"label batch shape: {label_batch.shape}")

print(image_batch[0].numpy())

rescale_layer = tf.keras.layers.Rescaling(scale=1.0 / 255)

image_scaled = rescale_layer(image_batch[0]).numpy()

# Rescale both datasets
train_dataset_scaled = train_dataset.map(lambda image, label: (rescale_layer(image), label))
validation_dataset_scaled = validation_dataset.map(lambda image, label: (rescale_layer(image), label))

# Get one batch of data
sample_batch = list(train_dataset_scaled.take(1))[0]

# Get the image
image_scaled = sample_batch[0][1].numpy()

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_dataset_scaled.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

# Configure the training set
train_dataset_final = train_dataset_scaled.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
# Configure the validation dataset
validation_dataset_final = validation_dataset_scaled.cache().prefetch(PREFETCH_BUFFER_SIZE)
history = model.fit(train_dataset_final, epochs=15, validation_data=validation_dataset_final, verbose=2)

# Create the widget and take care of the display
# uploader = widgets.FileUpload(accept="image/*", multiple=True)
# display(uploader)
# out = widgets.Output()
# display(out)


def file_predict(filename, file, out):
    """A function for creating the prediction and printing the output."""
    image = tf.keras.utils.load_img(file, target_size=(300, 300))
    image = tf.keras.utils.img_to_array(image)
    image = rescale_layer(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    with out:
        if prediction > 0.5:
            print(filename + " is a human")
        else:
            print(filename + " is a horse")


def on_upload_change(change):
    """A function for geting files from the widget and running the prediction."""
    # Get the newly uploaded file(s)

    items = change.new
    for item in items:  # Loop if there is more than one file uploaded
        file_jpgdata = BytesIO(item.content)
        file_predict(item.name, file_jpgdata, out)
