import tensorflow as tf
from joblib import load

def load_keras_model(path: str):
    return tf.keras.models.load_model(path)

def load_pipeline(path: str):
    return load(path)
