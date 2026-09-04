"""Trainer module for a TFX GenericExecutor Trainer component."""

import os
from typing import Dict, List, Text

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "target"

FEATURE_KEYS = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]


def transformed_name(key: str) -> str:
    return f"{key}_xf"


def _gzip_reader_fn(filenames: List[Text]) -> tf.data.TFRecordDataset:
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(
    file_pattern: List[Text],
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 32,
) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    data_files = []
    for pattern in file_pattern:
        data_files.extend(
            candidate
            for candidate in tf.io.gfile.glob(pattern)
            if not tf.io.gfile.isdir(candidate) and candidate.endswith(".gz")
        )
    if not data_files:
        raise ValueError(f"No transformed TFRecord files matched: {file_pattern}")
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=data_files,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
        shuffle=True,
    )


def _build_keras_model(hp: kt.HyperParameters) -> tf.keras.Model:
    inputs: Dict[str, tf.keras.layers.Input] = {
        transformed_name(key): tf.keras.layers.Input(
            shape=(1,), name=transformed_name(key), dtype=tf.float32
        )
        for key in FEATURE_KEYS
    }
    concatenated = tf.keras.layers.concatenate(list(inputs.values()))
    x = tf.keras.layers.Dense(hp.get("units_1"), activation="relu")(concatenated)
    x = tf.keras.layers.Dropout(hp.get("dropout_rate"))(x)
    x = tf.keras.layers.Dense(hp.get("units_2"), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.get("learning_rate")),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return {"outputs": model(transformed_features)}

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Train and export the model to the location supplied by TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
    model = _build_keras_model(hp)
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3, restore_best_weights=True
            )
        ],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    os.makedirs(fn_args.serving_model_dir, exist_ok=True)
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
