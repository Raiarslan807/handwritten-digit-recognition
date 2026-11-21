
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Preprocess handwritten digit images.
    Specifically handles blue ink on light notebook paper.
    Returns CNN input with shape (1, 28, 28, 1).
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert RGB to HSV
    hsv = np.array(image.convert("HSV"))

    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    # Detect blue ink
    blue_mask = (
        (hue >= 140) &
        (hue <= 190) &
        (saturation >= 35) &
        (value >= 40)
    )

    # If blue ink is detected, use color segmentation
    if np.count_nonzero(blue_mask) > 100:

        # Use saturation as ink intensity
        ink = np.zeros(saturation.shape, dtype=np.float32)

        ink[blue_mask] = np.clip(
            (saturation[blue_mask] - 25) / 100,
            0,
            1
        ) * 255

    else:

        # Fallback for black ink or grayscale images
        gray = np.array(
            image.convert("L"),
            dtype=np.float32
        )

        threshold = min(180, np.percentile(gray, 35))

        ink = np.clip(
            threshold - gray,
            0,
            255
        )

    # Remove weak noise
    ink[ink < 25] = 0

    # Find digit boundaries
    rows, cols = np.where(ink > 0)

    if len(rows) == 0:
        raise ValueError(
            "No handwritten digit detected. "
            "Please upload a clearer image."
        )

    # Crop only the digit
    cropped = ink[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Convert to PIL image
    digit = Image.fromarray(
        cropped.astype(np.uint8)
    )

    # Resize while preserving aspect ratio
    digit.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create black 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)

    # Paste digit in center
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    canvas.paste(digit, (x, y))

    # Normalize pixel values
    processed = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # Return CNN input
    return processed.reshape(1, 28, 28, 1)
