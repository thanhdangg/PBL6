from flask import Flask, render_template, request # type: ignore
import os 
from deeplearning import predict
import cv2 # type: ignore

app = Flask(__name__)

BASE_PATH = os.getcwd()
# print("bbbbbbbbbbbbbbbbb", BASE_PATH)
UPLOAD_PATH = os.path.join(BASE_PATH, "static/uploads")
# print("aaaaaaaaaaaaaaaaa: ", UPLOAD_PATH)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        upload_file = request.files["image_name"]
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)

        # Lưu tạm file vào một đường dẫn tạm thời
        temp_path = os.path.join(UPLOAD_PATH, 'temp_' + filename)
        upload_file.save(temp_path)
        
        # Resize ảnh sau khi tải lên
        img = cv2.imread(temp_path)  # Đọc ảnh bằng OpenCV
        resized_img = cv2.resize(img, (256, 256))  # Resize ảnh về 512x512
        
        # Lưu ảnh đã resize vào đường dẫn cuối cùng
        cv2.imwrite(path_save, resized_img)

        check = predict(path_save, filename)

        os.remove(temp_path)
        if check:
            return render_template("index.html", upload=True, upload_image=filename)
    return render_template("index.html", upload=False)

if __name__ == "__main__":
    app.run(debug=True)