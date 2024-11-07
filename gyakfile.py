import os

import numpy as np
import tensorflow as tf

MODEL_PATH = "saved_model.keras"
WEIGHTS_PATH = "model.weights.h5"
CHECKPOINT_PATH = "checkpoints/checkpoint.weights.h5"

# Adat előállítása
first_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)
second_array = np.array(
    [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
    dtype=float,
)

# Normalizálás
first_array_normalized = (first_array - np.mean(first_array)) / np.std(first_array)
second_array_normalized = (second_array - np.mean(second_array)) / np.std(second_array)

# Checkpoint callback létrehozása
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH, save_weights_only=True, save_best_only=True, monitor="loss", verbose=1
)

# Modell betöltése ha létezik, különben új létrehozása
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modell betöltve!")
else:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    print("Új modell létrehozva!")

# Modell fordítása
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mean_squared_error", metrics=["accuracy"])

# Tanítás
model.fit(first_array_normalized, second_array_normalized, epochs=500, callbacks=[checkpoint_callback])

# Modell mentése
model.save(MODEL_PATH)
model.save_weights(WEIGHTS_PATH)
print("Modell elmentve!")


# Predikció
prediction_normalized = (20.0 - np.mean(first_array)) / np.std(first_array)

prediction = model.predict(np.array([prediction_normalized]), verbose=0).item()

final_prediction = prediction * np.std(second_array) + np.mean(second_array)
print(f"Prediction: {final_prediction:.5f}")
