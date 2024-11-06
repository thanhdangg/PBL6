import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from models.segmentmodel import predict 
from tensorflow.keras.models import load_model

model = load_model('/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/models/skin-cancer-mnist-ham10000.keras')
classes = {
    4: ('nv', 'melanocytic nevi'), 
    6: ('mel', 'melanoma'), 
    2: ('bkl', 'benign keratosis-like lesions'), 
    1: ('bcc', 'basal cell carcinoma'), 
    5: ('vasc', 'pyogenic granulomas and hemorrhage'), 
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  
    3: ('df', 'dermatofibroma')
}

app = FastAPI()

def predict_image(image):
    img_resized = cv2.resize(image, (28, 28))
    result = model.predict(img_resized.reshape(1, 28, 28, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind][1] 
    return  class_name,result

def load_image_and_predict(url):
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to retrieve image from the URL"}, 400

    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dự đoán lớp
    label, pred = predict_image(image) 

    
    segment_result = predict(url, "test.jpg") 

    return JSONResponse(content={
        "label": label,
        "prediction_probabilities": pred.flatten().tolist(),
        "segment_result": segment_result
    })


@app.post("/predict/")
async def predictbylink(request: Request):
    data = await request.json()
    url = data.get('url')
    if not url:
        return JSONResponse(content={"error": "URL is missing"}, status_code=400)

    return load_image_and_predict(url)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3100)
