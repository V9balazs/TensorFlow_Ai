import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# The dataset is already downloaded for you. For downloading you can use the code below.
imdb = tfds.load("imdb_reviews", as_supervised=True, data_dir="../data/", download=False)

# Get the train and test sets
train_dataset, test_dataset = imdb["train"], imdb["test"]

# Vectorization and Padding Parameters

VOCAB_SIZE = 10000
MAX_LENGTH = 120
PADDING_TYPE = "pre"
TRUNC_TYPE = "post"

# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)

# Get the string inputs and integer outputs of the training set
train_reviews = train_dataset.map(lambda review, label: review)

# Generate the vocabulary based only on the training set
vectorize_layer.adapt(train_reviews)

# Delete because it's no longer needed
del train_reviews


def preprocessing_fn(dataset):
    """Generates padded sequences from a tf.data.Dataset"""

    # Apply the vectorization layer to the string features
    dataset_sequences = dataset.map(lambda text, label: (vectorize_layer(text), label))

    # Put all elements in a single ragged batch
    dataset_sequences = dataset_sequences.ragged_batch(batch_size=dataset_sequences.cardinality())

    # Output a tensor from the single batch. Extract the sequences and labels.
    sequences, labels = dataset_sequences.get_single_element()

    # Pad the sequences
    padded_sequences = tf.keras.utils.pad_sequences(
        sequences.numpy(), maxlen=MAX_LENGTH, truncating=TRUNC_TYPE, padding=PADDING_TYPE
    )

    # Convert back to a tf.data.Dataset
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    # Combine the padded sequences and labels
    dataset_vectorized = tf.data.Dataset.zip(padded_sequences, labels)

    return dataset_vectorized


# Preprocess the train and test data
train_dataset_vectorized = train_dataset.apply(preprocessing_fn)
test_dataset_vectorized = test_dataset.apply(preprocessing_fn)

# View 2 training sequences and its labels
for example in train_dataset_vectorized.take(2):
    print(example)
    print()

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Optimize and batch the datasets for training
train_dataset_final = (
    train_dataset_vectorized.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)
)

test_dataset_final = test_dataset_vectorized.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)

# Flatten model
# Parameters
EMBEDDING_DIM = 16
DENSE_DIM = 6

# Model Definition with a Flatten layer
model_flatten = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(DENSE_DIM, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Set the training parameters
model_flatten.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model_flatten.summary()

NUM_EPOCHS = 10

# Train the model
history_flatten = model_flatten.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=(test_dataset_final))

# LSTM model
# Parameters
EMBEDDING_DIM = 16
LSTM_DIM = 32
DENSE_DIM = 6

# Model Definition with LSTM
model_lstm = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
        tf.keras.layers.Dense(DENSE_DIM, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Set the training parameters
model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model_lstm.summary()

NUM_EPOCHS = 10

# Train the model
history_lstm = model_lstm.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)

# GRU model
# Parameters
EMBEDDING_DIM = 16
GRU_DIM = 32
DENSE_DIM = 6

# Model Definition with GRU
model_gru = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(GRU_DIM)),
        tf.keras.layers.Dense(DENSE_DIM, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Set the training parameters
model_gru.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model_gru.summary()

NUM_EPOCHS = 10

# Train the model
history_gru = model_gru.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=(test_dataset_final))

# Convolutional model
# Parameters
EMBEDDING_DIM = 16
FILTERS = 128
KERNEL_SIZE = 5
DENSE_DIM = 6

# Model Definition with Conv1D
model_conv = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(DENSE_DIM, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Set the training parameters
model_conv.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model_conv.summary()

NUM_EPOCHS = 10

# Train the model
history_conv = model_conv.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=(test_dataset_final))
