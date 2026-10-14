import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


IMAGE_SIZE = (48, 48)
NUM_CLASSES = 7
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def _pixels_to_array(pixel_string: str) -> np.ndarray:
    """Convert a FER2013 pixel string to a normalized 48x48x1 image array."""
    pixels = np.fromstring(pixel_string, dtype=np.float32, sep=" ")
    image = pixels.reshape(IMAGE_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image


def preprocess_face_image(face_gray: np.ndarray) -> np.ndarray:
    """
    Prepare a grayscale face image for model inference.

    Args:
        face_gray: 2D NumPy array (grayscale face crop).
    Returns:
        4D array with shape (1, 48, 48, 1), normalized to [0, 1].
    """
    if face_gray.shape != IMAGE_SIZE:
        raise ValueError(f"Expected image shape {IMAGE_SIZE}, got {face_gray.shape}")

    face = face_gray.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def load_fer2013(csv_path: str):
    """
    Load FER2013 from CSV, preprocess images, and split by Usage column.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """
    df = pd.read_csv(csv_path)

    images = np.stack(df["pixels"].apply(_pixels_to_array).to_numpy())
    labels = to_categorical(df["emotion"].values, NUM_CLASSES)
    usage = df["Usage"].astype(str).values

    train_mask = usage == "Training"
    val_mask = usage == "PublicTest"
    test_mask = usage == "PrivateTest"

    x_train, y_train = images[train_mask], labels[train_mask]
    x_val, y_val = images[val_mask], labels[val_mask]
    x_test, y_test = images[test_mask], labels[test_mask]

    return x_train, y_train, x_val, y_val, x_test, y_test
