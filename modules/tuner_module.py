"""Tuner module for automatic hyperparameter search."""

from typing import Dict, List, Text

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

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
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
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
        features=feature_spec,
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
    x = tf.keras.layers.Dense(
        units=hp.Choice("units_1", values=[32, 64, 128]), activation="relu"
    )(concatenated)
    x = tf.keras.layers.Dropout(
        rate=hp.Choice("dropout_rate", values=[0.1, 0.2, 0.3])
    )(x)
    x = tf.keras.layers.Dense(
        units=hp.Choice("units_2", values=[16, 32, 64]), activation="relu"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Return a lightweight RandomSearch tuner and its fit arguments."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    tuner = kt.RandomSearch(
        hypermodel=_build_keras_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name="breast_cancer_tuning",
        overwrite=True,
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 5,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=2
                )
            ],
        },
    )
