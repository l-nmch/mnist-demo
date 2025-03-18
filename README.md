# MNIST Demo

This project demonstrates a simple MNIST model for training, drawing digits, and predicting handwritten digits.

## Description

- `draw.py`: A GUI application to draw digits and save them as images.
- `model.py`: Script to train a convolutional neural network (CNN) on the MNIST dataset.
- `predict.py`: Script to predict the digit from a saved image using the trained model.

## Usage

### Drawing Digits

1. Run the `draw.py` script to open the drawing application:
    ```sh
    python draw.py
    ```
2. Draw a digit on the canvas.
3. Enter a filename and click "Save" to save the image.

### Training the Model

1. Run the `model.py` script to train the model:
    ```sh
    python model.py --model_name mnist_model.keras --epochs 6
    ```
2. The trained model will be saved as `mnist_model.keras`.

### Predicting Digits

1. Run the `predict.py` script to predict the digit from a saved image:
    ```sh
    python predict.py --model_name mnist_model.keras --input-file images/0.png
    ```
2. The script will output the predicted class and confidence.

## Model Information

### Summary

The model is a Convolutional Neural Network (CNN) with the following architecture:

- Input layer: 28x28 grayscale images
- Convolutional layers: 2 layers with ReLU activation
- MaxPooling layers: 2 layers
- Fully connected layers: 2 layers with ReLU activation
- Output layer: Softmax activation for 10 classes (digits 0-9)

### Efficiency

- Training accuracy: 99.2%
- Validation accuracy: 98.7%
- Training time: Approximately 4 minutes on a standard x86 CPU

## Future Uses

This project will be used to test model conversion and inference on [AI Hat+](https://www.raspberrypi.com/products/ai-hat/).