import numpy as np
import tensorflow as tf

# Load the dataset
data = open("Laurences_generated_poetry.txt", encoding="utf-8").read()

# Lowercase and split the text
corpus = data.lower().split("\n")

# Preview the result
# print(corpus)

# Initialize the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Build the vocabulary
vectorize_layer.adapt(corpus)

# Get the vocabulary and its size
vocabulary = vectorize_layer.get_vocabulary()
vocab_size = len(vocabulary)

# print(f"{vocabulary}")
# print(f"{vocab_size}")

# Initialize the sequences list
input_sequences = []

# Loop over every line
for line in corpus:

    # Generate the integer sequence of the current line
    sequence = vectorize_layer(line).numpy()

    # Loop over the line several times to generate the subphrases
    for i in range(1, len(sequence)):

        # Generate the subphrase
        n_gram_sequence = sequence[: i + 1]

        # Append the subphrase to the sequences list
        input_sequences.append(n_gram_sequence)

# Get the length of the longest line
max_sequence_len = max([len(x) for x in input_sequences])

# Pad all sequences
input_sequences = np.array(tf.keras.utils.pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

# Create inputs and label by splitting the last token in the subphrases
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

# Convert the label into one-hot arrays
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

# Get sample sentence
sentence = corpus[0].split()
# print(f"sample sentence: {sentence}")

# Initialize token list
token_list = []

# Look up the indices of each word and append to the list
for word in sentence:
    token_list.append(vocabulary.index(word))

# Print the token list
# print(token_list)


def sequence_to_text(sequence, vocabulary):
    """utility to convert integer sequence back to text"""

    # Loop through the integer sequence and look up the word from the vocabulary
    words = [vocabulary[index] for index in sequence]

    # Combine the words into one sentence
    text = tf.strings.reduce_join(words, separator=" ").numpy().decode()

    return text


# Pick element
# elem_number = 5

# Print token list and phrase
# print(f"token list: {xs[elem_number]}")
# print(f"decoded to text: {sequence_to_text(xs[elem_number], vocabulary)}")

# Print label
# print(f"one-hot label: {ys[elem_number]}")
# print(f"index of label: {np.argmax(ys[elem_number])}")

PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Put the inputs and labels to a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((xs, ys))

# Optimize the dataset for training
dataset = dataset.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)

# Parameters
embedding_dim = 100
lstm_units = 150
learning_rate = 0.01

# Build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(max_sequence_len - 1,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

# Use categorical crossentropy because this is a multi-class problem
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()

# Parameters
embedding_dim = 100
lstm_units = 150
learning_rate = 0.01

# Build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(max_sequence_len - 1,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

# Use categorical crossentropy because this is a multi-class problem
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()

# Parameters
embedding_dim = 100
lstm_units = 150
learning_rate = 0.01

# Build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(max_sequence_len - 1,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

# Use categorical crossentropy because this is a multi-class problem
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()

# --------------------------------------------------------------------------------------------------------

# Define seed text
seed_text = "help me obi-wan kenobi youre my only hope"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

    # Generate the integer sequence of the current line
    sequence = vectorize_layer(seed_text)

    # Pad the sequence
    sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_sequence_len - 1, padding="pre")

    # Feed to the model and get the probabilities for each index
    probabilities = model.predict(sequence, verbose=0)

    # Get the index with the highest probability
    predicted = np.argmax(probabilities, axis=-1)[0]

    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:

        # Look up the word associated with the index.
        output_word = vocabulary[predicted]

        # Combine with the seed text
        seed_text += " " + output_word

# Print the result
print(seed_text, "\n")

# --------------------------------------------------------------------------------------------------------

# Define seed text
seed_text = "help me obi-wan kenobi youre my only hope"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

    # Convert the seed text to an integer sequence
    sequence = vectorize_layer(seed_text)

    # Pad the sequence
    sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_sequence_len - 1, padding="pre")

    # Feed to the model and get the probabilities for each index
    probabilities = model.predict(sequence, verbose=0)

    # Pick a random number from [1,2,3]
    choice = np.random.choice([1, 2, 3])

    # Sort the probabilities in ascending order
    # and get the random choice from the end of the array
    predicted = np.argsort(probabilities)[0][-choice]

    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:

        # Look up the word associated with the index.
        output_word = vocabulary[predicted]

        # Combine with the seed text
        seed_text += " " + output_word

# Print the result
print(seed_text)
