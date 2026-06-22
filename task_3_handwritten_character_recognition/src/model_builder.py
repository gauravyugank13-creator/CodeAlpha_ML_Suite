"""
Task 3 — Handwritten Character Recognition
src/model_builder.py

Purpose:
    Defines and compiles the CNN (Convolutional Neural Network) architecture
    for handwritten digit recognition on MNIST.

    This module is responsible for architecture design only.
    Actual training is handled by trainer.py.

Usage:
    from src.model_builder import ModelBuilder
    model = ModelBuilder.build_cnn()
    model.summary()
"""


class ModelBuilder:
    """
    Constructs and compiles TensorFlow/Keras models.

    Available architectures:
        build_cnn()  — A simple, effective CNN for MNIST digit classification.

    Design philosophy:
        - Start with the simplest architecture that achieves >98% accuracy.
        - Avoid overengineering for a beginner-friendly baseline.
    """

    @staticmethod
    def build_cnn_model(
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Build and compile a lightweight Convolutional Neural Network for MNIST.

        Architecture:
            Input (28x28x1)
            -> Conv2D(32, (3,3), activation="relu")
            -> MaxPooling2D((2,2))
            -> Conv2D(64, (3,3), activation="relu")
            -> MaxPooling2D((2,2))
            -> Flatten
            -> Dense(128, activation="relu")
            -> Dropout(0.3)
            -> Dense(10, activation="softmax")

        Args:
            input_shape:    Shape of a single input image. Default: (28, 28, 1).
            num_classes:    Number of output classes. Default: 10 (digits 0-9).
            learning_rate:  Adam optimizer learning rate. Default: 0.001.

        Returns:
            tf.keras.Model: A compiled Keras model ready for training.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.optimizers import Adam
        except ImportError as e:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from e

        model = models.Sequential(name="MNIST_Lightweight_CNN")

        # Block 1
        model.add(layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            name="conv_block1_conv",
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name="conv_block1_pool"))

        # Block 2
        model.add(layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            name="conv_block2_conv",
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name="conv_block2_pool"))

        # Classifier
        model.add(layers.Flatten(name="flatten"))
        model.add(layers.Dense(128, activation="relu", name="dense_dense"))
        model.add(layers.Dropout(0.3, name="dense_dropout"))
        model.add(layers.Dense(num_classes, activation="softmax", name="output_softmax"))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("[ModelBuilder] Lightweight CNN model built and compiled successfully.")
        return model

    @staticmethod
    def build_cnn(
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 10,
        learning_rate: float = 0.001,
    ):
        """Wrapper for build_cnn_model to preserve backward compatibility."""
        return ModelBuilder.build_cnn_model(input_shape, num_classes, learning_rate)

    @staticmethod
    def get_parameter_count(model) -> dict:
        """
        Get the parameter counts of the model.

        Args:
            model: tf.keras.Model instance.

        Returns:
            dict: {
                'total_params': int,
                'trainable_params': int,
                'non_trainable_params': int
            }
        """
        import numpy as np
        trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights])) if model.trainable_weights else 0
        non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights])) if model.non_trainable_weights else 0
        return {
            "total_params": trainable + non_trainable,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable
        }

    @staticmethod
    def load_model(model_path: str):
        """
        Load a previously saved Keras model from disk.

        Args:
            model_path: Absolute or relative path to the .keras model file.

        Returns:
            tf.keras.Model: The loaded model.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ImportError:       If TensorFlow is not installed.

        TODO (Phase 4 — Evaluation):
            - Add version checking to ensure the model was saved with
              a compatible TensorFlow version.
        """
        import os
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from e

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ModelBuilder] Model file not found: {model_path}"
            )

        print(f"[ModelBuilder] Loading model from: {model_path}")
        return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# TODO: Phase 3 — Model Training
#   - Add build_mlp() for a simpler baseline (no convolutions) comparison.
#
# TODO: Phase 5 — Deployment
#   - Add convert_to_tflite(model) to produce a lightweight model for
#     deployment in the Flask app.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check — build and print model summary without training.
    model = ModelBuilder.build_cnn()
