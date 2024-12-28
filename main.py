import numpy as np
import cv2
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel

from models.segment_model import predict
from models.classify_model import predict_image
from infra.Database.database import engine, get_db
from infra.Database import crud, schema, login_services
from models.segment_model import getting_segmet_model
from models.classify_model import getting_classify_model
from models.attribute_detection_model import getting_attribute_detection_model,detect_attributes
app = FastAPI()
SQLModel.metadata.create_all(bind=engine)
classify_model = getting_classify_model()
segment_model = getting_segmet_model()
attribute_dectection_model = getting_attribute_detection_model()


def load_image_and_predict(url, userid):
    """
    :param url: Image URL
    :param userid: User ID
    :return: Json response with label, prediction probabilities and segment result
    """
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to retrieve image from the URL"}, 400
    np_arr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    label, pred = predict_image(image, classify_model)
    segment_result = predict(url, "test.jpg", segment_model)
    attribute_result = detect_attributes(url,attribute_dectection_model)
    crud.create_prediction(
        next(get_db()),
        schema.PredictionCreate(
            raw_image=url,
            segment_image=segment_result,
            prediction_result=label,
            user_id=userid,
        ),
    )
    return JSONResponse(
        content={
            "label": label,
            "prediction_probabilities": pred.flatten().tolist(),
            "segment_result": segment_result,
            "attribute_result": attribute_result
        }
    )


@app.post("/predict")
async def predict_by_link(request: Request):
    data = await request.json()
    url = data.get("url")
    userid = data.get("userid")
    if not url:
        return JSONResponse(content={"error": "URL is missing"}, status_code=400)

    return load_image_and_predict(url, userid)


@app.post("/register")
async def create_user(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return JSONResponse(
            content={"error": "Username or password is missing"}, status_code=400
        )
    if crud.find_user_by_username(next(get_db()), username):
        return JSONResponse(content={"error": "User already exists"}, status_code=400)
    user = crud.create_user(
        next(get_db()), schema.UserCreate(username=username, password=password)
    )
    return JSONResponse(content={"message": "User created successfully",
                                 "userid": user.id}, status_code=201)


@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return JSONResponse(
            content={"error": "Username or password is missing"}, status_code=400
        )
    user = login_services.authenticate_user(next(get_db()), username, password)
    userid = user.id
    if not user:
        return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)
    return JSONResponse(content={"userid": userid}, status_code=200)


@app.get("/history")
async def get_history(request: Request):
    userid = request.query_params.get("userid")
    if not userid:
        return JSONResponse(content={"error": "User id is missing"}, status_code=400)
    predictions = crud.get_predictions(next(get_db()), user_id=int(userid))
    return JSONResponse(
        content={"predictions": [prediction.dict() for prediction in predictions]}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
