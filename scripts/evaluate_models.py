import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from tensorflow import keras


def load_test_dataset(dataset_dir, image_size, batch_size):
    test_ds = keras.utils.image_dataset_from_directory(
        dataset_dir / "test",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    return test_ds, test_ds.class_names


def collect_predictions(model, test_ds, num_classes):
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    y_true_one_hot = np.eye(num_classes)[y_true]
    return y_true, y_pred, y_prob, y_true_one_hot


def plot_history(history_path, output_path, title):
    data = json.loads(history_path.read_text())
    history = data["history"]
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["loss"], label="Training loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.65)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cat breed classifiers.")
    parser.add_argument("--dataset-dir", default="dataset_split")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    graphs_dir = results_dir / "graphs"
    cm_dir = results_dir / "confusion_matrices"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    test_ds, class_names = load_test_dataset(dataset_dir, args.image_size, args.batch_size)
    num_classes = len(class_names)

    rows = []
    for model_path in sorted(models_dir.glob("*.keras")):
        model_name = model_path.stem
        print(f"Evaluating {model_name}")
        model = keras.models.load_model(model_path, compile=False)
        y_true, y_pred, y_prob, y_true_one_hot = collect_predictions(model, test_ds, num_classes)

        accuracy = accuracy_score(y_true, y_pred)
        map_score = average_precision_score(y_true_one_hot, y_prob, average="macro")

        history_path = results_dir / f"{model_name}_history.json"
        parameter_count = model.count_params()
        training_time_seconds = None
        if history_path.exists():
            history_data = json.loads(history_path.read_text())
            parameter_count = history_data.get("parameter_count", parameter_count)
            training_time_seconds = history_data.get("training_time_seconds")
            plot_history(
                history_path,
                graphs_dir / f"{model_name}_accuracy_loss.png",
                f"{model_name} Training Performance",
            )

        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            cm_dir / f"{model_name}_confusion_matrix.png",
            f"{model_name} Confusion Matrix",
        )

        rows.append({
            "model": model_name,
            "test_accuracy": accuracy,
            "test_mAP": map_score,
            "parameter_count": parameter_count,
            "training_time_seconds": training_time_seconds,
        })

    comparison = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    print(comparison)
    print(f"Saved comparison table to {results_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
