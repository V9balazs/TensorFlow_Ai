import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
imdb = tfds.load("imdb_reviews", as_supervised=True, data_dir="./data", download=False)

train_reviews = imdb["train"].map(lambda review, label: review)
train_labels = imdb["train"].map(lambda review, label: label)

test_reviews = imdb["test"].map(lambda review, label: review)
test_labels = imdb["test"].map(lambda review, label: label)

# Show two reviews
list(train_reviews.take(2))

# Parameters for tokenization and padding
VOCAB_SIZE = 10000
MAX_LENGTH = 120
PADDING_TYPE = "pre"
TRUNC_TYPE = "post"

# Instantiate the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)

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


# Apply the vectorization layer and padding on the training inputs
train_sequences = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)

# Get the vocabulary
imdb_vocab_fullword = vectorize_layer.get_vocabulary()

# Get a sample integer sequence
sample_sequence = train_sequences.take(1).get_single_element()

# Lookup each token in the vocabulary
decoded_text = [imdb_vocab_fullword[token] for token in sample_sequence]

# Combine the words
decoded_text = " ".join(decoded_text)

# Print the output
print(decoded_text)

# Compute the subword vocabulary and save to a file
keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_reviews,
    vocabulary_size=8000,
    reserved_tokens=["[PAD]", "[UNK]"],
    vocabulary_output_file="imdb_vocab_subwords.txt",
)

# Initialize the subword tokenizer
subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary="./imdb_vocab_subwords.txt")

# Print the subwords
subword_tokenizer.get_vocabulary()

# Show the size of the subword vocabulary
subword_tokenizer.vocabulary_size()

# Get a sample review
sample_review = train_reviews.take(1).get_single_element()

# Encode the first plaintext sentence using the subword text encoder
tokenized_string = subword_tokenizer.tokenize(sample_review)
print("Tokenized string is {}".format(tokenized_string))

# Decode the sequence
original_string = subword_tokenizer.detokenize(tokenized_string)

# Print the result
print("The original string: {}".format(original_string))

# Define sample sentence
sample_string = "TensorFlow, from basics to mastery"

# Encode using the plain text tokenizer
tokenized_string = vectorize_layer(sample_string)
print("Tokenized string is {}".format(tokenized_string))

# Decode and print the result
decoded_text = [imdb_vocab_fullword[token] for token in tokenized_string]
original_string = " ".join(decoded_text)
print("The original string: {}".format(original_string))

# Encode using the subword text encoder
tokenized_string = subword_tokenizer.tokenize(sample_string)
print("Tokenized string is {}".format(tokenized_string))

# Decode and print the results
original_string = subword_tokenizer.detokenize(tokenized_string).numpy().decode("utf-8")
print("The original string: {}".format(original_string))

# Show token to subword mapping:
for ts in tokenized_string:
    print("{} ----> {}".format(ts, subword_tokenizer.detokenize([ts]).numpy().decode("utf-8")))

SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

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

# Define dimensionality of the embedding
EMBEDDING_DIM = 64

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(subword_tokenizer.vocabulary_size(), EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Print the model summary
model.summary()

num_epochs = 10

# Set the training parameters
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Start training
history = model.fit(train_dataset_final, epochs=num_epochs, validation_data=test_dataset_final)
