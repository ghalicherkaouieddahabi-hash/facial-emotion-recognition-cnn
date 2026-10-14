# Facial Emotion Recognition using CNN

A deep learning project for facial emotion recognition using a Convolutional Neural Network (CNN), the FER2013 dataset, TensorFlow/Keras, and OpenCV. The project includes a desktop graphical interface built with Tkinter for image-based emotion prediction.

## Overview

This project explores facial emotion recognition by processing facial images and predicting the corresponding emotion using a trained CNN model.

The desktop application allows users to select an image and view the model's prediction through a graphical interface.

## Features

- Facial image selection through a desktop interface
- Face image preprocessing using OpenCV
- CNN-based emotion classification
- Trained model loading with TensorFlow/Keras
- Prediction results displayed in a Tkinter interface

## Technologies

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Numerical Computing:** NumPy
- **Desktop GUI:** Tkinter
- **Image Processing:** Pillow
- **Dataset:** FER2013

## Project Structure

```text
Facial-Emotion-Recognition-using-CNN-FER2013-main/
├── data/
│   ├── best_emotion_model.h5
│   ├── emotion_model.h5
│   ├── gui.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ghalicherkaouieddahabi-hash/facial-emotion-recognition-cnn.git
cd facial-emotion-recognition-cnn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

Install the required libraries:

```bash
pip install tensorflow opencv-python numpy pillow
```

Tkinter is usually included with standard Python installations on Windows.

## Usage

From the project root, run the desktop application:

```bash
python data/gui.py
```

If your local project structure differs, open a terminal in the directory containing `gui.py` and run:

```bash
python gui.py
```

The application loads a trained model, lets you select an image, and displays the predicted emotion.

## Model Files

The repository contains two trained model files:

- `emotion_model.h5`
- `best_emotion_model.h5`

The desktop application currently references `emotion_model.h5`. The `best_emotion_model.h5` file is also included for experimentation.

## Project Goals

This project demonstrates practical applications of:

- Convolutional Neural Networks
- Deep learning for image classification
- Computer vision and image preprocessing
- Integration of trained machine learning models into desktop applications

## Limitations

Predictions depend on image quality, facial visibility, preprocessing, and the model's training data. The predicted emotion should be treated as a model estimate rather than a definitive statement about a person's emotional state.

## Author

**Ghali Cherkaoui Eddahabi**

GitHub: [@ghalicherkaouieddahabi-hash](https://github.com/ghalicherkaouieddahabi-hash)

---

*This project was developed for learning and experimentation in deep learning and computer vision.*