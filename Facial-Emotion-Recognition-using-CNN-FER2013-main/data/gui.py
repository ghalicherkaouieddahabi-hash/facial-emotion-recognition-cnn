import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from preprocess import CLASS_NAMES, IMAGE_SIZE, preprocess_face_image


MODEL_PATH = "emotion_model.h5"


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Emotion Recognition - FER2013")
        self.root.geometry("700x650")

        self.model = None
        self.selected_gray = None
        self.photo_preview = None

        self.status_var = tk.StringVar(value="Loading model...")
        self.result_var = tk.StringVar(value="Prediction: -")

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Facial Emotion Recognition (CNN)",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(pady=10)

        self.preview_label = tk.Label(
            self.root,
            text="No image selected",
            width=60,
            height=20,
            bg="#f0f0f0",
            relief="groove",
        )
        self.preview_label.pack(padx=10, pady=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        load_btn = tk.Button(
            button_frame,
            text="Load Image",
            command=self.load_image,
            width=16,
            font=("Segoe UI", 11),
        )
        load_btn.grid(row=0, column=0, padx=8)

        predict_btn = tk.Button(
            button_frame,
            text="Predict Emotion",
            command=self.predict_emotion,
            width=16,
            font=("Segoe UI", 11),
        )
        predict_btn.grid(row=0, column=1, padx=8)

        webcam_btn = tk.Button(
            button_frame,
            text="Webcam",
            command=self.run_webcam,
            width=16,
            font=("Segoe UI", 11),
        )
        webcam_btn.grid(row=0, column=2, padx=8)

        result_label = tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Segoe UI", 14, "bold"),
            fg="#1a237e",
        )
        result_label.pack(pady=8)

        status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Segoe UI", 10), fg="#444"
        )
        status_label.pack(pady=5)

    def _load_model(self):
        try:
            self.model = load_model(MODEL_PATH)
            self.status_var.set(f"Model loaded: {MODEL_PATH}")
        except Exception as exc:
            self.status_var.set("Model loading failed.")
            messagebox.showerror(
                "Model Error",
                f"Unable to load model file '{MODEL_PATH}'.\n\nDetails: {exc}",
            )

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            messagebox.showerror("File Error", "Unable to read selected image.")
            return

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self.selected_gray = cv2.resize(gray, IMAGE_SIZE)

        # Show original image in GUI
        display_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        display_pil = Image.fromarray(display_rgb)
        display_pil.thumbnail((500, 420))
        self.photo_preview = ImageTk.PhotoImage(display_pil)
        self.preview_label.configure(image=self.photo_preview, text="")

        self.result_var.set("Prediction: -")
        self.status_var.set(f"Image loaded: {path}")

    def predict_emotion(self):
        if self.model is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return
        if self.selected_gray is None:
            messagebox.showwarning("Image Required", "Please load an image first.")
            return

        x = preprocess_face_image(self.selected_gray)
        probabilities = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx]) * 100.0
        label = CLASS_NAMES[idx]

        self.result_var.set(f"Prediction: {label} ({confidence:.2f}%)")
        self.status_var.set("Prediction completed.")

    def run_webcam(self):
        if self.model is None:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Webcam Error", "Unable to access webcam.")
            return

        self.status_var.set("Webcam running. Press Q to quit.")
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
                x_input = preprocess_face_image(face)
                probs = self.model.predict(x_input, verbose=0)[0]
                idx = int(np.argmax(probs))
                label = CLASS_NAMES[idx]
                conf = float(probs[idx]) * 100.0

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Recognition Webcam", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Webcam stopped.")


def main():
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
