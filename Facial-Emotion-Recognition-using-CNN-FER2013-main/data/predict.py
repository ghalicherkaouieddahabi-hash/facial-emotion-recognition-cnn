import argparse

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import CLASS_NAMES, IMAGE_SIZE, preprocess_face_image


def predict_from_face_array(model, face_gray: np.ndarray):
    """Predict emotion from a grayscale face crop (48x48)."""
    x = preprocess_face_image(face_gray)
    probabilities = model.predict(x, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    return CLASS_NAMES[class_index], confidence, probabilities


def predict_image_file(model_path: str, image_path: str):
    """Predict emotion from a single image file."""
    model = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, IMAGE_SIZE)
    label, confidence, _ = predict_from_face_array(model, gray_resized)
    print(f"Prediction: {label} ({confidence * 100:.2f}%)")


def run_webcam(model_path: str):
    """
    Real-time webcam emotion recognition with Haar Cascade face detection.
    Press Q to quit.
    """
    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    print("Webcam started. Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, IMAGE_SIZE)
            label, confidence, _ = predict_from_face_array(model, face)

            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Emotion Recognition - Webcam", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Predict facial emotion from image or webcam."
    )
    parser.add_argument("--model", default="emotion_model.h5", help="Path to model file.")
    parser.add_argument("--image", help="Path to input image.")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for real-time prediction.",
    )
    args = parser.parse_args()

    if args.webcam:
        run_webcam(args.model)
    elif args.image:
        predict_image_file(args.model, args.image)
    else:
        parser.error("Provide --image <path> or --webcam.")


if __name__ == "__main__":
    main()
