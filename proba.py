import io

import tensorflow as tf
import tensorflow_datasets as tfds

# Load the IMDB Reviews dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True, data_dir="./data/", download=False)

# Print information about the dataset
print(info)

# Print the contents of the dataset
print(imdb)

# View 4 training examples
for example in imdb["train"].take(4):
    print(example)

# Get the train and test sets
train_dataset, test_dataset = imdb["train"], imdb["test"]

# Parameters

VOCAB_SIZE = 10000
MAX_LENGTH = 120
EMBEDDING_DIM = 16
PADDING_TYPE = "pre"
TRUNC_TYPE = "post"

# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
)

# Get the string inputs and integer outputs of the training set
train_reviews = train_dataset.map(lambda review, label: review)
train_labels = train_dataset.map(lambda review, label: label)

# Get the string inputs and integer outputs of the test set
test_reviews = test_dataset.map(lambda review, label: review)
test_labels = test_dataset.map(lambda review, label: label)

# Generate the vocabulary based only on the training set
vectorize_layer.adapt(train_reviews)


def padding_func(sequences):
    """Generates padded sequences from a tf.data.Dataset"""

    # Put all elements in a single ragged batch
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality())

    # Output a tensor from the single batch
    sequences = sequences.get_single_element()

    # Pad the sequences
    padded_sequences = tf.keras.utils.pad_sequences(
        sequences.numpy(), maxlen=MAX_LENGTH, truncating=TRUNC_TYPE, padding=PADDING_TYPE
    )

    # Convert back to a tf.data.Dataset
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

    return padded_sequences


def padding_func(sequences):
    """Generates padded sequences from a tf.data.Dataset"""

    # Put all elements in a single ragged batch
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality())

    # Output a tensor from the single batch
    sequences = sequences.get_single_element()

    # Pad the sequences
    padded_sequences = tf.keras.utils.pad_sequences(
        sequences.numpy(), maxlen=MAX_LENGTH, truncating=TRUNC_TYPE, padding=PADDING_TYPE
    )

    # Convert back to a tf.data.Dataset
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

    return padded_sequences


# Apply the layer to the train and test data
train_sequences = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
test_sequences = test_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)

# View 2 training sequences
for example in train_sequences.take(2):
    print(example)
    print()

train_dataset_vectorized = tf.data.Dataset.zip(train_sequences, train_labels)
test_dataset_vectorized = tf.data.Dataset.zip(test_sequences, test_labels)

# View 2 training sequences and its labels
for example in train_dataset_vectorized.take(2):
    print(example)
    print()

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Optimize the datasets for training
train_dataset_final = (
    train_dataset_vectorized.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)
)

test_dataset_final = test_dataset_vectorized.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Setup the training parameters
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

NUM_EPOCHS = 5

# Train the model
model.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)
