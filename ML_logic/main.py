from ML_logic.data_loader import load_datasets, data_split, test_split
from ML_logic.model import initialize_model, compile_model
from ML_logic.trainer import train_model, evaluate_model
from config import IMAGE_SIZE, NUM_CLASSES


Normal_vision = "raw_data/Normal_vision_split1/train"
Normal_vision_test = "raw_data/Normal_vision_split1/test"

Green_blind = "raw_data/Simulated_GreenBlind_split1/train"
Green_blind_test = "raw_data/Simulated_GreenBlind_split1/test"

Red_blind = "raw_data/Simulated_RedBlind50_split1/train"
Red_blind_test = "raw_data/Simulated_RedBlind50_split1/test"

blue_blind = "raw_data/Simulated_BlueBlind50_split1/train"
blue_blind_test = "raw_data/Simulated_BlueBlind50_split1/test"


def run_red_blind(data_path, test_path):
    print("Running training for Red-Blind simulated dataset...")

    path = load_datasets(data_path=data_path)
    test_dir = load_datasets(data_path=test_path)

    train_ds, val_ds = data_split(path, image_size=IMAGE_SIZE)
    test_ds = test_split(test_dir, image_size=IMAGE_SIZE)

    model = initialize_model(num_classes=NUM_CLASSES)
    model = compile_model(model)
    history = train_model(model, train_ds, val_ds, epochs=100, patience=20)

    evaluate_model(model, test_ds)
    return history


if __name__ == "__main__":
    run_red_blind(Red_blind, Normal_vision_test)
