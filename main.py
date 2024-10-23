import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from models.model import create_model, load_model_weights
from models.segmentmodel import predict  # Giả sử hàm này là hàm dự đoán từ mô hình phân đoạn

# Tải mô hình phân loại
classification_model = create_model(input_shape=(224, 224, 3))
classification_model = load_model_weights(classification_model, '/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/models/best_model.h5')

# Định nghĩa các lớp cho mô hình phân loại
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

# Hàm dự đoán cho mô hình phân loại
def predict_image(image):
    img = cv2.resize(image, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = classification_model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    return classes[class_idx], pred

# Hàm tải hình ảnh và thực hiện dự đoán
def load_image_and_predict(url):
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to retrieve image from the URL"}, 400

    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dự đoán lớp
    label, pred = predict_image(image) 

    # Resize ảnh về kích thước 256x256 cho mô hình phân đoạn
    image_segment = cv2.resize(image, (256, 256))

    # Dự đoán phân đoạn
    segment_result = predict(url, "test.jpg")  # Giả sử hàm predict là hàm dự đoán phân đoạn

    return JSONResponse(content={
        "predicted_class": label[1],
        "label": label[0],
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
