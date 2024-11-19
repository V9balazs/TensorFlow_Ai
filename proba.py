import keras_nlp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# The dataset is already downloaded for you. For downloading you can use the code below.
imdb = tfds.load("imdb_reviews", as_supervised=True, data_dir="../data/", download=False)

# Extract the train reviews and labels
train_reviews = imdb["train"].map(lambda review, label: review)
train_labels = imdb["train"].map(lambda review, label: label)

# Extract the test reviews and labels
test_reviews = imdb["test"].map(lambda review, label: review)
test_labels = imdb["test"].map(lambda review, label: label)

# Initialize the subword tokenizer
subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary="./imdb_vocab_subwords.txt")

# Data pipeline and padding parameters
SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 256
PADDING_TYPE = "pre"
TRUNC_TYPE = "post"


def padding_func(sequences):
    """Generates padded sequences from a tf.data.Dataset"""

    # Put all elements in a single ragged batch
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality())

    # Output a tensor from the single batch
    sequences = sequences.get_single_element()

    # Pad the sequences
    padded_sequences = tf.keras.utils.pad_sequences(sequences.numpy(), truncating=TRUNC_TYPE, padding=PADDING_TYPE)

    # Convert back to a tf.data.Dataset
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

    return padded_sequences


# Generate integer sequences using the subword tokenizer
train_sequences_subword = train_reviews.map(lambda review: subword_tokenizer.tokenize(review)).apply(padding_func)
test_sequences_subword = test_reviews.map(lambda review: subword_tokenizer.tokenize(review)).apply(padding_func)

# Combine the integer sequence and labels
train_dataset_vectorized = tf.data.Dataset.zip(train_sequences_subword, train_labels)
test_dataset_vectorized = tf.data.Dataset.zip(test_sequences_subword, test_labels)

# Optimize the datasets for training
train_dataset_final = (
    train_dataset_vectorized.shuffle(SHUFFLE_BUFFER_SIZE)
    .cache()
    .prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

test_dataset_final = test_dataset_vectorized.cache().prefetch(buffer_size=PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)

# Parameters
BATCH_SIZE = 1
TIMESTEPS = 20
FEATURES = 20
FILTERS = 128
KERNEL_SIZE = 5

print(f"batch_size: {BATCH_SIZE}")
print(f"timesteps (sequence length): {TIMESTEPS}")
print(f"features (embedding size): {FEATURES}")
print(f"filters: {FILTERS}")
print(f"kernel_size: {KERNEL_SIZE}")

# Define array input with random values
random_input = np.random.rand(BATCH_SIZE, TIMESTEPS, FEATURES)
print(f"shape of input array: {random_input.shape}")

# Pass array to convolution layer and inspect output shape
conv1d = tf.keras.layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu")
result = conv1d(random_input)
print(f"shape of conv1d output: {result.shape}")

# Pass array to max pooling layer and inspect output shape
gmp = tf.keras.layers.GlobalMaxPooling1D()
result = gmp(result)
print(f"shape of global max pooling output: {result.shape}")

# Hyperparameters
EMBEDDING_DIM = 64
FILTERS = 128
KERNEL_SIZE = 5
DENSE_DIM = 64

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(None,)),
        tf.keras.layers.Embedding(subword_tokenizer.vocabulary_size(), EMBEDDING_DIM),
        tf.keras.layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(DENSE_DIM, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Print the model summary
model.summary()

# Set the training parameters
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

NUM_EPOCHS = 10

# Train the model
history = model.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)
