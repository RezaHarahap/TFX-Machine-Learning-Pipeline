"""Transform module for the breast-cancer classification pipeline."""

import tensorflow as tf
import tensorflow_transform as tft

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
    """Return the transformed feature name."""
    return f"{key}_xf"


def preprocessing_fn(inputs):
    """Normalize numeric features and cast the binary label."""
    outputs = {
        transformed_name(key): tft.scale_to_z_score(inputs[key])
        for key in FEATURE_KEYS
    }
    if LABEL_KEY in inputs:
        outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
