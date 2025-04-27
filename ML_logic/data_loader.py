from pathlib import Path
import tensorflow as tf
from config import IMAGE_SIZE


def load_datasets(data_path):
    """
    Load images and labels for normal vision data from the raw_data path.
    """
    data_dir = Path(data_path)

    class_names = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred"
    ).class_names
    print(f'✅ Data loaded {class_names}')

    return data_dir


def data_split(data_dir, image_size=IMAGE_SIZE, seed=123):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        subset="training",
        seed=seed,
        image_size=image_size,
        label_mode='int',
        validation_split=0.2
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
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
    return train_ds, val_ds


def test_split(data_dir, image_size=IMAGE_SIZE, seed=123):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        seed=seed,
        image_size=image_size,
        label_mode='int',
        shuffle=False
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    print("✅ Data split into test set, normalized")
    return test_ds
