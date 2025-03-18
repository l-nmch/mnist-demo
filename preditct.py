import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import argparse

parser = argparse.ArgumentParser(description='Train a model on the MNIST dataset.')
parser.add_argument('--model_name', type=str, default='mnist_model.keras', help='Name of the model file to save')
parser.add_argument('--input-file', type=str, help='Input file to predict', required=True)

args = parser.parse_args()

model = tf.keras.models.load_model(args.model_name)

image = Image.open(args.input_file)


def predict_digit():
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized.convert("L"))

    img_array = np.array(img_inverted)
    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=-1)  # (1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction


predicted_class, predicion = predict_digit()
print(f"Predicted class: {predicted_class} with confidence: {predicion[0][predicted_class]}")