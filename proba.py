import numpy as np
import tensorflow as tf

# Define the lyrics of the song
data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

# Split the long string per line and put in a list
corpus = data.lower().split("\n")

# Preview the result
print(corpus)

# Initialize the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Build the vocabulary
vectorize_layer.adapt(corpus)

# Get the vocabulary and its size
vocabulary = vectorize_layer.get_vocabulary()
vocab_size = len(vocabulary)

print(f"{vocabulary}")
print(f"{vocab_size}")

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
print(f"sample sentence: {sentence}")

# Initialize token list
token_list = []

# Look up the indices of each word and append to the list
for word in sentence:
    token_list.append(vocabulary.index(word))

# Print the token list
print(token_list)


def sequence_to_text(sequence, vocabulary):
    """utility to convert integer sequence back to text"""

    # Loop through the integer sequence and look up the word from the vocabulary
    words = [vocabulary[index] for index in sequence]

    # Combine the words into one sentence
    text = tf.strings.reduce_join(words, separator=" ").numpy().decode()

    return text


# Pick element
elem_number = 6

# Print token list and phrase
print(f"token list: {xs[elem_number]}")
print(f"decoded to text: {sequence_to_text(xs[elem_number], vocabulary)}")

# Print label
print(f"one-hot label: {ys[elem_number]}")
print(f"index of label: {np.argmax(ys[elem_number])}")

# Pick element
elem_number = 5

# Print token list and phrase
print(f"token list: {xs[elem_number]}")
print(f"decoded to text: {sequence_to_text(xs[elem_number], vocabulary)}")

# Print label
print(f"one-hot label: {ys[elem_number]}")
print(f"index of label: {np.argmax(ys[elem_number])}")

# Build the model
model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(max_sequence_len - 1,)),
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)

# Use categorical crossentropy because this is a multi-class problem
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

# Train the model
history = model.fit(xs, ys, epochs=300)

# Define seed text
seed_text = "Laurence went to Dublin"

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

    # Get the index with the highest probability
    predicted = np.argmax(probabilities, axis=-1)[0]

    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:

        # Look up the word associated with the index.
        output_word = vocabulary[predicted]

        # Combine with the seed text
        seed_text += " " + output_word

# Print the result
print(seed_text)

# Define seed text
seed_text = "Laurence went to Dublin"

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
