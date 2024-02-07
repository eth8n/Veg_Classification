from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("../createdModels/Potato3")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']
app = FastAPI()


def read_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    img = read_image(await file.read())
    img_batch = np.expand_dims(img,0)
    prediction = MODEL.predict(img_batch)
    pred_class = CLASS_NAMES[np.argmax(prediction[0])]
    pred_conf = np.max(prediction[0])
    return{ 'class': pred_class,  'confidence': float(pred_conf)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=7777)
