from pathlib import Path
import tensorflow as tf
from config import IMAGE_SIZE


def load_datasets(data_path, image_size=IMAGE_SIZE, seed=123):
    """
    Load images and labelsfor normal vision data from the raw_data path.
    """
    data_dir = Path(data_path) # تعديل

    class_names = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred"
    ).class_names
    print(f'✅ Data loaded{class_names}')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        subset="training",
        seed=seed,
        image_size=image_size,
        label_mode='int',
        validation_split=0.2
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        subset="validation",
        seed=seed,
        image_size=image_size,
        label_mode='int',
        validation_split=0.2
    )


    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    print("✅ Data split, normalized")
    return train_ds, val_ds, class_names
