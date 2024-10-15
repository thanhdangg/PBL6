# main.py

import numpy as np
import cv2
import requests
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from models.model import create_model, load_model_weights

# Khởi tạo mô hình
model = create_model()
model = load_model_weights(model, '/mnt/01D9E8A400C52160/Ki7/pbl6/Skin-cancer-Analyzer/models/best_model.h5')

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

# Hàm dự đoán
def predict_image(image):
    img = cv2.resize(image, (28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    return classes[class_idx], pred

@app.post("/predict/")
async def predict(url: str = Form(...)):
    # Tải ảnh từ URL
    response = requests.get(url)
    if response.status_code != 200:
        return JSONResponse(content={"error": "Failed to retrieve image from the URL"}, status_code=400)

    # Chuyển đổi nội dung thành mảng NumPy
    nparr = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dự đoán hình ảnh
    label, pred = predict_image(image)
    
    # Trả về kết quả
    return JSONResponse(content={
        "predicted_class": label[1],
        "label": label[0],
        "prediction_probabilities": pred.flatten().tolist()
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3100)
