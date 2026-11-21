
import numpy as np


def predict_digit(model, processed_image):
    """
    Predict handwritten digit and return
    predicted class, confidence and probabilities.
    """

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return predicted_digit, confidence, probabilities
