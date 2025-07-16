from vectorLearn import *
import torch
import joblib

def infer_parameters(input_path, target_path):
    # 이미지 로드 및 벡터 추출
    current_img = cv2.imread(input_path)
    target_img = cv2.imread(target_path)
    current_vec = extract_image_vector(current_img)
    target_vec = extract_image_vector(target_img)
    combined_vec = np.concatenate([current_vec, target_vec])

    # 전처리기 불러오기
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    # 벡터 전처리
    vec_scaled = scaler.transform([combined_vec])
    vec_pca = pca.transform(vec_scaled)

    INPUT_DIM = 32  # same as PCA n_components
    OUTPUT_DIM = 6  # number of predicted parameters

    model = AdjustmentRegressor(INPUT_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load("adjustment_regressor.pth"))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(vec_pca, dtype=torch.float32)
        output_tensor = model(input_tensor)
        predicted_params = output_tensor.numpy()[0]

    print("예측된 파라미터 조정값:", predicted_params)
    return predicted_params

def apply_strength_to_parameters(predicted_params, strength):
    brightness = float(predicted_params[0]) * strength
    contrast = (float(predicted_params[1]) - 1) * strength + 1
    hue_shift = int(predicted_params[2]) * strength
    saturation_scale = (float(predicted_params[3]) - 1) * strength + 1
    value_scale = (float(predicted_params[4]) - 1) * strength + 1
    gamma = (float(predicted_params[5]) - 1) * strength + 1

    return brightness, contrast, hue_shift, saturation_scale, value_scale, gamma

def apply_correction_with_strength(input_path, predicted_params, strength):
    input_img = cv2.imread(input_path)
    brightness, contrast, hue_shift, saturation_scale, value_scale, gamma = apply_strength_to_parameters(predicted_params, strength)

    corrected_img = apply_adjustments(
        input_img,
        brightness=brightness,
        contrast=contrast,
        hue_shift=hue_shift,
        saturation_scale=saturation_scale,
        value_scale=value_scale,
        gamma=gamma
    )
    return corrected_img