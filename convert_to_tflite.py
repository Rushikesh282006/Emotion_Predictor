import tensorflow as tf
from pathlib import Path
import sys

# Define paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

models_to_convert = [
    ("Bidirec_gru_model_kaggle.keras", "Bidirec_gru_model_kaggle.tflite"),
    ("Bidirec_gru_model.keras", "Bidirec_gru_model.tflite"),
]

converted_any = False
for keras_name, tflite_name in models_to_convert:
    keras_path = ARTIFACTS_DIR / keras_name
    tflite_path = ARTIFACTS_DIR / tflite_name
    if keras_path.exists():
        print(f"Loading Keras model from: {keras_path}")
        model = tf.keras.models.load_model(str(keras_path))

        print(f"Converting {keras_name} to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Successfully converted and saved TFLite model to: {tflite_path}")
        converted_any = True

if not converted_any:
    print(f"Error: No Keras models found in {ARTIFACTS_DIR}")
    sys.exit(1)
