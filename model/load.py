import tensorflow as tf

def load_model(model_path="Ishihara_recognition/model/Red_CNN_FinalModel83.keras"):
    """Load and return the trained model"""
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model
