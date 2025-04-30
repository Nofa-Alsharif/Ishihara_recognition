
from Normal_vison.data_loader import load_datasets
from Normal_vison.model import initialize_model, compile_model, initialize_model_greenblind
from Normal_vison.trainer import train_model
from config import IMAGE_SIZE,NUM_CLASSES


Normal_vision = "raw_data/Normal_vision"
Green_blind = "raw_data/Simulated_GreenBlind50"

def run_normal_vision(data_path):
    print("Running training for Normal Vision dataset...")
    train_ds, val_ds, class_names = load_datasets(data_path=data_path, image_size=IMAGE_SIZE)
    model = initialize_model(num_classes=NUM_CLASSES)
    model = compile_model(model)
    history = train_model(model, train_ds, val_ds,epochs =10,patience=10)
    return history

def run_green_blind(data_path):
    print("Running training for Green-Blind simulated dataset...")
    train_ds, val_ds, class_names = load_datasets(data_path=data_path, image_size=IMAGE_SIZE)
    model = initialize_model_greenblind(num_classes=NUM_CLASSES)
    model = compile_model(model)
    history = train_model(model, train_ds, val_ds ,epochs =50,patience=20)

    return history


if __name__ == "__main__":
    choice = input("Which dataset do you want to train on? (normal / green): ").strip().lower()

    if choice == "normal":
        run_normal_vision(Normal_vision)
    elif choice == "green":
        run_green_blind(Green_blind)
    else:
        print("Invalid choice. Please enter 'normal' or 'green'.")
