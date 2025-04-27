from tensorflow.keras.models import Model
from tensorflow.keras import models, layers, regularizers
from config import NUM_CLASSES

def initialize_model(num_classes=NUM_CLASSES): # rename the method to Normal vision
    """
    Bulid and compile the CNN model.
    """
    model = models.Sequential([
        layers.InputLayer(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.6),  # <- Dropout
        layers.Dense(10, activation='softmax')
    ])
    print("✅ Model initialized")
    return model

# Bulid a Green blind model
# def initialize_model_greenblind(num_classes=NUM_CLASSES):

#     model = models.Sequential([
#         layers.InputLayer(input_shape=(224, 224, 3)),
#         layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.5),  # <- Dropout
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     print("✅ GreenBlind model initialized")
#     return model


def compile_model(model: Model):
    """
    Compile the Neural Network
    """

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    print("✅ Model compiled")
    return model
