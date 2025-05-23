from fastapi import FastAPI
import tensorflow as tf
from PIL import Image
import random
from fastapi.responses import JSONResponse
from ML_logic.data_loader import load_image_paths
from PIL import Image
import base64

app = FastAPI()

model = tf.keras.models.load_model("raw_data/Red_CNN_FinalModel83.keras")
print("✅ Loaded model successfully")
print(":rocket: Server running! Access http://localhost:8000 :rocket:")

Image_path = {
    "Normal": "raw_data/Normal_vision_split1/test",
    "Green":"raw_data/Simulated_BlueBlind50_split1/test",
    "Blue": "raw_data/Simulated_GreenBlind_split1/test",
    "Red": "raw_data/Simulated_RedBlind50_split1/test"
}

@app.get("/")
def home():
    return {"message": "Welcome to Whisper in Hue API!"}


@app.get("/predict_image/")
async def predict_image(vision_type: str):
    """
    Predict the number inside the image
    """
    if vision_type not in Image_path:
        return JSONResponse(content={"error": "Invalid vision type"}, status_code=400)
    image_list = load_image_paths(Image_path[vision_type])
    if not image_list:
        return JSONResponse(content={"error": "No images found in the folder"}, status_code=404)
    random_image_path = random.choice(image_list)
    img = Image.open(random_image_path).convert("RGB")
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = int(tf.argmax(prediction, axis=1).numpy()[0])
    try:
        with open(random_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "message": str(predicted_class),
            "image_base64": encoded_image,
            "format": "jpeg"
        }
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Image not found"})
