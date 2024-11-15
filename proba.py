import tensorflow as tf

# Sample inputs
sentences = ["I love my dog", "I love my cat", "You love my dog!", "Do you think my dog is amazing?"]

# Initialize the layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Compute the vocabulary
vectorize_layer.adapt(sentences)

# Get the vocabulary
vocabulary = vectorize_layer.get_vocabulary()

# Print the token index
for index, word in enumerate(vocabulary):
    print(index, word)

# String input
sample_input = "I love my dog"

# Convert the string input to an integer sequence
sequence = vectorize_layer(sample_input)

# Print the result
print(sequence)

# Convert the list to tf.data.Dataset
sentences_dataset = tf.data.Dataset.from_tensor_slices(sentences)

# Define a mapping function to convert each sample input
sequences = sentences_dataset.map(vectorize_layer)

# Print the integer sequences
for sentence, sequence in zip(sentences, sequences):
    print(f"{sentence} ---> {sequence}")

# Apply the layer to the string input list
sequences_post = vectorize_layer(sentences)

# Print the results
print("INPUT:")
print(sentences)
print()

print("OUTPUT:")
print(sequences_post)

# Pre-pad the sequences to a uniform length.
# You can remove the `padding` argument and get the same result.
sequences_pre = tf.keras.utils.pad_sequences(sequences, padding="pre")

# Print the results
print("INPUT:")
[print(sequence.numpy()) for sequence in sequences]
print()

print("OUTPUT:")
print(sequences_pre)

# Post-pad the sequences and limit the size to 5.
sequences_post_trunc = tf.keras.utils.pad_sequences(sequences, maxlen=5, padding="pre")

# Print the results
print("INPUT:")
[print(sequence.numpy()) for sequence in sequences]
print()

print("OUTPUT:")
print(sequences_post_trunc)

# Set the layer to output a ragged tensor
vectorize_layer = tf.keras.layers.TextVectorization(ragged=True)

# Compute the vocabulary
vectorize_layer.adapt(sentences)

# Apply the layer to the sentences
ragged_sequences = vectorize_layer(sentences)

# Print the results
print(ragged_sequences)

# Pre-pad the sequences in the ragged tensor
sequences_pre = tf.keras.utils.pad_sequences(ragged_sequences.numpy())

# Print the results
print(sequences_pre)

# Try with words that are not in the vocabulary
sentences_with_oov = ["i really love my dog", "my dog loves my manatee"]

# Generate the sequences
sequences_with_oov = vectorize_layer(sentences_with_oov)

# Print the integer sequences
for sentence, sequence in zip(sentences_with_oov, sequences_with_oov):
    print(f"{sentence} ---> {sequence}")
