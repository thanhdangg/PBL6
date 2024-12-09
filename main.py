import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from models.segmentmodel import predict
from models.classifymodel import predict_image

app = FastAPI()


def load_image_and_predict(url):
    """

    :param url: Image URL
    :return: Json response with label, prediction probabilities and segment result
    """
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to retrieve image from the URL"}, 400
    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    label, pred = predict_image(image)
    segment_result = predict(url, "test.jpg")
    return JSONResponse(
        content={
            "label": label,
            "prediction_probabilities": pred.flatten().tolist(),
            "segment_result": segment_result,
        }
    )


@app.post("/predict/")
async def predictbylink(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse(content={"error": "URL is missing"}, status_code=400)

    return load_image_and_predict(url)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3100)
