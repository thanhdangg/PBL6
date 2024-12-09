import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import SQLModel

from models.segment_model import predict
from models.classify_model import predict_image
from infra.Database.database import engine, SessionLocal, get_db
from infra.Database import crud, schema, login_services
from models.segment_model import getting_segmet_model
from models.classify_model import getting_classify_model
app = FastAPI()
# SQLModel.metadata.create_all(bind=engine)
classify_model = getting_classify_model()
segment_model = getting_segmet_model()

class UserBase(BaseModel):
    username: str
    password: str


class PredictionBase(BaseModel):
    raw_image: str
    segment_image: str
    prediction_result: str



def load_image_and_predict(url):
    """
    :param url: Image URL
    :return: Json response with label, prediction probabilities and segment result
    """
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to retrieve image from the URL"}, 400
    np_arr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    label, pred = predict_image(image,classify_model)
    segment_result = predict(url, "test.jpg",segment_model)
    return JSONResponse(
        content={
            "label": label,
            "prediction_probabilities": pred.flatten().tolist(),
            "segment_result": segment_result,
        }
    )


@app.post("/predict/")
async def predict_by_link(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse(content={"error": "URL is missing"}, status_code=400)

    return load_image_and_predict(url)


@app.post("/register/")
async def create_user(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    print(username, password)
    crud.create_user(get_db(), schema.UserCreate(username=username, password=password))
    if not username or not password:
        return JSONResponse(content={"error": "Username or password is missing"}, status_code=400)
    return JSONResponse(content={"message": "User created successfully"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3100)