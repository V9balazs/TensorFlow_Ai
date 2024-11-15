import tensorflow as tf
import json
import tensorflow_datasets as tfds
from tensorflow.keras.utils import pad_sequences

# Download the dataset
!wget -nc https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json

# Load the JSON file
with open("./sarcasm.json", 'r') as f:
    datastore = json.load(f)

# Non-sarcastic headline
print(datastore[0])

# Sarcastic headline
print(datastore[20000])

# Append the headline elements into the list
sentences = [item['headline'] for item in datastore]

# Instantiate the layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Build the vocabulary
vectorize_layer.adapt(sentences)

# Apply the layer for post padding
post_padded_sequences = vectorize_layer(sentences)

# Print a sample headline and sequence
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {post_padded_sequences[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of padded sequences: {post_padded_sequences.shape}')

# Instantiate the layer and set the `ragged` flag to `True`
vectorize_layer = tf.keras.layers.TextVectorization(ragged=True)

# Build the vocabulary
vectorize_layer.adapt(sentences)

# Apply the layer to generate a ragged tensor
ragged_sequences = vectorize_layer(sentences)

# Print a sample headline and sequence
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {ragged_sequences[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of padded sequences: {ragged_sequences.shape}')

# Apply pre-padding to the ragged tensor
pre_padded_sequences = pad_sequences(ragged_sequences.numpy())

# Preview the result for the 2nd sequence
pre_padded_sequences[2]

# Print a sample headline and sequence
index = 2
print(f'sample headline: {sentences[index]}')
print()
print(f'post-padded sequence: {post_padded_sequences[index]}')
print()
print(f'pre-padded sequence: {pre_padded_sequences[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of post-padded sequences: {post_padded_sequences.shape}')
print(f'shape of pre-padded sequences: {pre_padded_sequences.shape}')

