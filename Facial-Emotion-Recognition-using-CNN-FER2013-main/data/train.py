import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_emotion_cnn
from preprocess import CLASS_NAMES, load_fer2013


CSV_PATH = "fer2013.csv"
MODEL_PATH = "emotion_model.h5"
BEST_MODEL_PATH = "best_emotion_model.h5"
EPOCHS = 20
BATCH_SIZE = 64


def plot_history(history):
    """Display training and validation curves for loss and accuracy."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_fer2013(CSV_PATH)

    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Validation: {x_val.shape}, {y_val.shape}")
    print(f"Test: {x_test.shape}, {y_test.shape}")
    print(f"Classes: {CLASS_NAMES}")

    model = build_emotion_cnn()
    model.summary()

    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best validation model saved to {BEST_MODEL_PATH}")

    # Save final epoch model as requested.
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()
