from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import os
import cv2
from vectorInfer import infer_parameters, apply_correction_with_strength
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # CORS 적용

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/process", methods=["POST"])
def process():
    if 'input' not in request.files or 'target' not in request.files:
        return {"error": "필수 이미지가 누락되었습니다."}, 400

    input_img = request.files["input"]
    target_img = request.files["target"]
    strength = float(request.form.get("strength", 0.5))

    # 파일 이름 보안 처리 및 저장
    input_filename = secure_filename(input_img.filename)
    target_filename = secure_filename(target_img.filename)
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    target_path = os.path.join(UPLOAD_FOLDER, target_filename)
    input_img.save(input_path)
    target_img.save(target_path)

    try:
        # 이미지 보정
        predicted_params = infer_parameters(input_path, target_path)
        corrected_img = apply_correction_with_strength(input_path, predicted_params, strength)

        # 결과 이미지 저장
        result_filename = f"corrected_{input_filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, corrected_img)

        return send_file(result_path, mimetype='image/jpeg')

    except Exception as e:
        # 에러 처리
        return {"error": str(e)}, 500

    finally:
        # 처리 후 임시 파일 삭제
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(target_path):
            os.remove(target_path)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001) # React 개발 서버와 다른 포트 사용
