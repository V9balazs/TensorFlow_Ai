import numpy as np
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") >= 0.6:  # Experiment with changing this value
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

index = 1001

np.set_printoptions(linewidth=320)

print(f"LABEL: {training_labels[index]}")
print(f"\nIMAGE PIXEL ARRAY:\n\n{training_images[index]}\n\n")

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f"input to softmax function: {inputs.numpy()}")

outputs = tf.keras.activations.softmax(inputs)
print(f"output of softmax function: {outputs.numpy()}")

sum = tf.reduce_sum(outputs)
print(f"sum of outputs: {sum}")

prediction = np.argmax(outputs)
print(f"class with highest probability: {prediction}")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
