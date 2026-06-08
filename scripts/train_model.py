import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_BUILDERS = {
    "resnet50": keras.applications.ResNet50,
    "densenet121": keras.applications.DenseNet121,
    "mobilenetv3": keras.applications.MobileNetV3Large,
}

PREPROCESSORS = {
    "resnet50": keras.applications.resnet50.preprocess_input,
    "densenet121": keras.applications.densenet.preprocess_input,
    "mobilenetv3": keras.applications.mobilenet_v3.preprocess_input,
}


class MeanAveragePrecision(keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_average_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ap = keras.metrics.AUC(curve="PR", multi_label=True, num_labels=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        self.ap.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.ap.result()

    def reset_state(self):
        self.ap.reset_state()


def load_datasets(dataset_dir, image_size, batch_size, seed):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    return train_ds, val_ds, train_ds.class_names


def build_model(model_name, num_classes, image_size, dropout):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = PREPROCESSORS[model_name](inputs)
    base = MODEL_BUILDERS[model_name](
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{model_name}_cat_breed_classifier")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train one CNN model for cat breed classification.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--dataset-dir", default="dataset_split")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets(
        dataset_dir, args.image_size, args.batch_size, args.seed
    )
    num_classes = len(class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(args.model, num_classes, args.image_size, args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            MeanAveragePrecision(num_classes=num_classes),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / f"{args.model}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        keras.callbacks.CSVLogger(str(results_dir / f"{args.model}_training_log.csv")),
    ]

    print(model.summary())
    start_time = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    training_time_seconds = time.perf_counter() - start_time

    metadata = {
        "model": args.model,
        "class_names": class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "parameter_count": int(model.count_params()),
        "training_time_seconds": training_time_seconds,
        "history": history.history,
    }

    metadata_path = results_dir / f"{args.model}_history.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved best model to {models_dir / f'{args.model}.keras'}")
    print(f"Saved training metadata to {metadata_path}")


if __name__ == "__main__":
    main()
