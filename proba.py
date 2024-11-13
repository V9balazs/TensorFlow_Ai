import os

import tensorflow as tf

# Set the weights file you downloaded into a variable
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Initialize the base model.
# Set the input shape and remove the dense layers.
pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None
)

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# Choose `mixed7` as the last layer of your base model
last_layer = pre_trained_model.get_layer("mixed7")
print("last layer output shape: ", last_layer.output.shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation="relu")(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# Append the dense network to the base model
model = tf.keras.Model(pre_trained_model.input, x)

# Print the model summary. See your dense network connected at the end.
model.summary()

BASE_DIR = "/tf/cats_and_dogs_filtered"

train_dir = os.path.join(BASE_DIR, "train")
validation_dir = os.path.join(BASE_DIR, "validation")

# Directory with training cat/dog pictures
train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

# Directory with validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

# Prepare the training set
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(150, 150), batch_size=20, label_mode="binary"
)

# Prepare the validation set
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir, image_size=(150, 150), batch_size=20, label_mode="binary"
)


# Define the preprocess function
def preprocess(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label


# Apply the preprocessing to the datasets
train_dataset_scaled = train_dataset.map(preprocess)
validation_dataset_scaled = validation_dataset.map(preprocess)

# Optimize the datasets for training
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_dataset_scaled.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)

validation_dataset_final = validation_dataset_scaled.cache().prefetch(PREFETCH_BUFFER_SIZE)

# Create a model with data augmentation layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

# Attach the data augmentation model to the base model
inputs = tf.keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)
x = model(x)

model_with_aug = tf.keras.Model(inputs, x)

# Set the training parameters
model_with_aug.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
)

EPOCHS = 20

# Train the model.
history = model_with_aug.fit(train_dataset_final, validation_data=validation_dataset_final, epochs=EPOCHS, verbose=2)
