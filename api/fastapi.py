from fastapi import FastAPI
import tensorflow as tf
from PIL import Image
import random
import os
from fastapi.responses import JSONResponse ,FileResponse
from ML_logic.data_loader import load_image_paths

from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()



model = tf.keras.models.load_model("model/Red_CNN_FinalModel83.keras")
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

#Test how to work
@app.get("/show-image")
def show_image():
     # Open the image using Pillow
    img_path = "raw_data/Normal_vision_split1/train/0/0_Asap-MediumItalictheme_1 type_1.png"
    img = Image.open(img_path)

    # Resize the image (example size)
    img = img.resize((224, 224))  # Resize to 800x800, modify as needed

    # You can also apply transformations here, like rotating or adjusting brightness, etc.
    # img = img.rotate(90)  # Example: rotate the image by 90 degrees

    # Save the image to a byte stream
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(img_byte_arr, media_type="image/png")
#Test End


@app.get("/get_random_image/")
def get_random_image(vision_type: str):
    """
    Get a random image from vision type.
    """
    if vision_type not in Image_path:
        return JSONResponse(content={"error": "Invalid vision type"}, status_code=400)

    image_list = load_image_paths(Image_path[vision_type])
    #print(image_list)

    if not image_list:
        return JSONResponse(content={"error": "No images found"}, status_code=404)

    random_image = random.choice(image_list)
    #Hear
    #return JSONResponse(str(random_image))
    #end
    img = Image.open(random_image)
    img = img.resize((224, 224))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/predict_number/")
async def predict_number(vision_type: str):
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

    return JSONResponse(content={
        "random_image": random_image_path, #--> String path
        "predicted_class": predicted_class
    })
