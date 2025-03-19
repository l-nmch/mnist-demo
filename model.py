# Partialy based on https://www.tensorflow.org/datasets/keras_example

import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a model on the MNIST dataset.')
parser.add_argument('--save', type=str, help='Name of the model file to save as .keras')
parser.add_argument('--export', type=str, help='Name of the folder to export the model')
parser.add_argument('--epochs', type=int, default=6, help='Number of epochs to train the model')
args = parser.parse_args()

# Load the MNIST dataset and split it into training and test sets
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# Define the normalization function
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# Normalize the training set
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Normalize the test set
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Define the model
model = tf.keras.Sequential(
    [   
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(
    ds_train,
    epochs=args.epochs,
    validation_data=ds_test,
)

# Show model summary
model.summary()

# Show model accuracy
model.evaluate(ds_test)

# Save the model based on the chosen option
if args.save:
    model.save(args.save)
elif args.export:
    model.export(args.export)
else:
    print("Model not saved as per user request.")