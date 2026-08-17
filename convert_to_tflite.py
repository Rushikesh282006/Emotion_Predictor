import tensorflow as tf
from pathlib import Path
import sys

# Define paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"
KERAS_MODEL_PATH = ARTIFACTS_DIR / "Bidirec_gru_model.keras"
TFLITE_MODEL_PATH = ARTIFACTS_DIR / "Bidirec_gru_model.tflite"

if not KERAS_MODEL_PATH.exists():
    print(f"Error: Keras model not found at {KERAS_MODEL_PATH}")
    sys.exit(1)

print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))

print("Converting model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully converted and saved TFLite model to: {TFLITE_MODEL_PATH}")
