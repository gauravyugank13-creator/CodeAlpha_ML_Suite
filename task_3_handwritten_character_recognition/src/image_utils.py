"""
image_utils.py
Task 3 — Handwritten Character Recognition

Purpose:
    Provides utility functions for image loading, validation, and preprocessing
    specifically designed to match the MNIST training pipeline inputs.
"""

import os
import cv2
import numpy as np


def preprocess_grayscale(img: np.ndarray, method: str = "new") -> np.ndarray:
    """
    Helper function to preprocess a 2D grayscale NumPy array.
    Supports baseline direct resizing ("old") and advanced MNIST-style centering ("new").
    """
    # Auto-inversion:
    # MNIST digits are white strokes on a black background.
    # If the user uploads a black digit on a white page, we must invert it.
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    if method == "old":
        # Direct resize to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=(0, -1))

    # "new" method: Bounding box crop, resize preserving aspect ratio, center-of-mass shift.
    # Filter noise and threshold to isolate the digit
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        # If the image is blank, return a blank 28x28 image
        canvas = np.zeros((28, 28), dtype=np.float32)
        return np.expand_dims(canvas, axis=(0, -1))

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = img[y:y+h, x:x+w]

    # Resize to fit within a 20x20 box preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = int(round(h * 20.0 / w))
        new_h = max(1, new_h)
    else:
        new_h = 20
        new_w = int(round(w * 20.0 / h))
        new_w = max(1, new_w)

    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center bounding-box inside a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_digit

    # Shift using center of mass to match MNIST dataset distribution
    M = cv2.moments(canvas)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Shift center of mass to target position (14.0, 14.0)
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        
        # Translation warp matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, translation_matrix, (28, 28), flags=cv2.INTER_CUBIC)

    # Normalize to [0.0, 1.0] and clip to prevent bicubic overshoots
    processed = canvas.astype("float32") / 255.0
    processed = np.clip(processed, 0.0, 1.0)

    return np.expand_dims(processed, axis=(0, -1))


def preprocess_image(image_path: str, method: str = "new") -> np.ndarray:
    """
    Load a custom image from disk and prepare it for model inference.

    Steps:
        1. Validates the file format (.png, .jpg, .jpeg)
        2. Loads the image as grayscale
        3. Preprocesses the image according to selected method (old/new)
        4. Reshapes to (1, 28, 28, 1) for the CNN input layer

    Args:
        image_path: Path to the input image file.
        method: Preprocessing algorithm - "new" (advanced) or "old" (baseline).

    Returns:
        numpy.ndarray: Preprocessed image array of shape (1, 28, 28, 1), float32.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError:        If the file format is unsupported or the image is corrupt.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Validate file extension
    allowed_exts = {".png", ".jpg", ".jpeg"}
    _, ext = os.path.splitext(image_path.lower())
    if ext not in allowed_exts:
        raise ValueError(
            f"Unsupported image format '{ext}'. Allowed formats: {', '.join(allowed_exts)}"
        )

    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image. File may be corrupt: {image_path}")

    return preprocess_grayscale(img, method)


def preprocess_image_bytes(image_bytes: bytes, method: str = "new") -> np.ndarray:
    """
    Decodes image bytes from memory and prepares it for model inference.

    Args:
        image_bytes: Raw binary bytes of the image file.
        method: Preprocessing algorithm - "new" (advanced) or "old" (baseline).

    Returns:
        numpy.ndarray: Preprocessed image array of shape (1, 28, 28, 1), float32.

    Raises:
        ValueError: If image bytes cannot be decoded or are corrupt.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image bytes. File may be corrupt.")

    return preprocess_grayscale(img, method)

