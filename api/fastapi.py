from fastapi import FastAPI
import tensorflow as tf
from PIL import Image
import random
import os
from fastapi.responses import JSONResponse

data_folder = "raw_data/Normal_vision_split1/train"
model = tf.keras.models.load_model("Ishihara_recognition/model/Red_CNN_FinalModel83.keras")

app = FastAPI()

@app.get("/predict-random/")
async def predict_random_image():

    images = [file for file in os.listdir(data_folder) if file.endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        return JSONResponse(content={"error": "No images found in the folder"}, status_code=404)

    random_image = random.choice(images)
    image_path = os.path.join(data_folder, random_image)

    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)


    prediction = model.predict(img)
    predicted_class = prediction.argmax()

    return JSONResponse(content={
        "random_image": random_image,
        "predicted_class": int(predicted_class)
    })
