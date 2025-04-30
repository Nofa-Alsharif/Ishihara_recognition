import numpy as np
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_ds, val_ds, epochs,patience):
    """
    Train the model on given train_ds and val_ds data.
    """
    model: Model
    epochs= epochs
    batch_size = 32
    es = EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size =batch_size,
        callbacks=[es]
    )

    best_val_acc = round(np.max(history.history['val_accuracy']), 4)
    print(f"✅ Model trained with best validation accuracy: {best_val_acc}")

    return history
