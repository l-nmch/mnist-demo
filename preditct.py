import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

model = tf.keras.models.load_model('mnist_model.h5')

image = Image.open('0.png')

def show_image(image, title):
    image.show(title=title)

def predict_digit():
    show_image(image, "Image Brute")

    img_resized = image.resize((28, 28))
    show_image(img_resized, "Image Redimensionnée")

    img_inverted = ImageOps.invert(img_resized.convert("L"))
    show_image(img_inverted, "Image Inversée")

    img_array = np.array(img_inverted)
    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=-1)  # (1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

predicted_class = predict_digit()
print(f"Predicted class: {predicted_class}")