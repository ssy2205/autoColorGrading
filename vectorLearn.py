import cv2
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. 이미지 보정 함수 정의 ---
def apply_adjustments(image, brightness=0, contrast=1.0, hue_shift=0, saturation_scale=1.0, value_scale=1.0, gamma=1.0):
    # 밝기 및 대비
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # HSV로 변환
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] *= saturation_scale
    hsv[..., 2] *= value_scale
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 감마 보정
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    adjusted = cv2.LUT(adjusted, table)

    return adjusted


# --- 2. 이미지 벡터 추출기 ---
def extract_image_vector(image):
    image = cv2.resize(image, (256, 256))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    rgb_mean = np.mean(rgb, axis=(0, 1))
    hsv_mean = np.mean(hsv, axis=(0, 1))
    lab_mean = np.mean(lab, axis=(0, 1))

    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    contrast = np.std(hsv[..., 2])
    
    return np.concatenate([rgb_mean, hsv_mean, lab_mean, hist_features, [contrast]])


# --- 3. 학습 데이터 생성 (입력: 원본+보정 벡터) ---
def generate_training_data(image_paths, num_samples_per_image=100):
    X = []
    y = []
    for image_path in image_paths:
        original = cv2.imread(image_path)
        for _ in range(num_samples_per_image):
            params = {
                "brightness": random.randint(-5, 5),
                "contrast": round(random.uniform(0.9, 1.1), 2),
                "hue_shift": random.randint(-3, 3),
                "saturation_scale": round(random.uniform(0.9, 1.1), 2),
                "value_scale": round(random.uniform(0.9, 1.1), 2),
                "gamma": round(random.uniform(0.9, 1.1), 2)
            }
            adjusted = apply_adjustments(original, **params)
            current_vector = extract_image_vector(original)
            target_vector = extract_image_vector(adjusted)
            vector = np.concatenate([current_vector, target_vector])
            X.append(vector)
            y.append(list(params.values()))
    return np.array(X), np.array(y)


# --- 4. 모델 학습 ---
class AdjustmentRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_regressor(X, y, num_epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = AdjustmentRegressor(input_dim=X.shape[1], output_dim=y.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return model


# --- 5. 실행 예시 ---
if __name__ == "__main__":
    image_folder = "train_images"
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.jpg', '.png'))]

    print("데이터 생성 중...")
    X, y = generate_training_data(image_paths, num_samples_per_image=100)

    print("스케일링 및 차원 축소 중...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=32)
    X_reduced = pca.fit_transform(X_scaled)

    print("모델 학습 중...")
    model = train_regressor(X_reduced, y, num_epochs=100)

    torch.save(model.state_dict(), "adjustment_regressor.pth")
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")
    print("모델 저장 완료 ✅")