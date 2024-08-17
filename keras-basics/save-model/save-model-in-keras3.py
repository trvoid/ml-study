import numpy as np
import tensorflow as tf
import keras

def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model

model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("my_model.keras")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

print(">>>>>>>>>> my_model.keras Test OK!!!")

# Older Keras H5 format
model.save("my_model.h5", save_format="h5")

reconstructed_model = keras.models.load_model("my_model.h5")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

print(">>>>>>>>>> my_model.h5 Test OK!!!")

# TensorFlow SavedModel format (Seems not supported in this API any more)
#model.save("my_model", save_format="tf")
