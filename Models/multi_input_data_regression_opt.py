import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# 加载数据
data_path = 'processed_data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# 提取特征和目标
fingerprints = np.stack(data['MACCS_Normalized'])
images = np.stack(data['Image_Features_Normalized'])
labels = data['logBB'].values

# 定义数据集类
class MixedDataset(Dataset):
    def __init__(self, fingerprints, images, labels):
        self.fingerprints = fingerprints
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.fingerprints[idx], dtype=torch.float32),
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# 定义多模态神经网络
class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()
        # 分子指纹子网络
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 图像特征子网络
        self.image_fc = nn.Sequential(
            nn.Linear(image_feature_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 融合网络
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, fingerprint, image):
        fingerprint_out = self.fingerprint_fc(fingerprint)
        image_out = self.image_fc(image)
        combined = torch.cat((fingerprint_out, image_out), dim=1)
        output = self.fc(combined)
        return output

# 初始化变量
fingerprint_size = fingerprints.shape[1]
image_feature_size = images.shape[1]
criterion = nn.MSELoss()

# K 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
nn_predictions, rf_predictions, actuals, xgb_predictions = [], [], [], []

for train_idx, test_idx in kf.split(fingerprints):
    # 数据划分
    X_fingerprints_train, X_fingerprints_test = fingerprints[train_idx], fingerprints[test_idx]
    X_images_train, X_images_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # 创建数据加载器
    train_dataset = MixedDataset(X_fingerprints_train, X_images_train, y_train)
    test_dataset = MixedDataset(X_fingerprints_test, X_images_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化神经网络模型、优化器和学习率调度器
    model = MixedInputModel(fingerprint_size, image_feature_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 训练神经网络
    model.train()
    for epoch in range(50):
        total_loss = 0
        for fingerprints_batch, images_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(fingerprints_batch, images_batch).squeeze()
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # 测试神经网络
    model.eval()
    nn_preds, y_actual = [], []
    with torch.no_grad():
        for fingerprints_batch, images_batch, labels_batch in test_loader:
            preds = model(fingerprints_batch, images_batch).squeeze()
            nn_preds.extend(preds.cpu().numpy())
            y_actual.extend(labels_batch.cpu().numpy())

    nn_predictions.extend(nn_preds)
    actuals.extend(y_actual)

    # 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
    X_train_combined = np.hstack([X_fingerprints_train, X_images_train])
    X_test_combined = np.hstack([X_fingerprints_test, X_images_test])
    rf_model.fit(X_train_combined, y_train)
    rf_preds = rf_model.predict(X_test_combined)
    rf_predictions.extend(rf_preds)

    # XGBoost 模型
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
    xgb_model.fit(X_train_combined, y_train)
    xgb_preds = xgb_model.predict(X_test_combined)
    xgb_predictions.extend(xgb_preds)

# 集成模型预测
ensemble_predictions = (0.7 * np.array(nn_predictions) +
                        0.1 * np.array(rf_predictions) +
                        0.2 * np.array(xgb_predictions))

# 评估模型
mse = mean_squared_error(actuals, ensemble_predictions)
r2 = r2_score(actuals, ensemble_predictions)
print(f"Ensemble Model - Mean Squared Error: {mse:.4f}")
print(f"Ensemble Model - R² Score: {r2:.4f}")

# 可视化预测结果
plt.scatter(actuals, ensemble_predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ensemble Predicted vs Actual")
plt.savefig("ensemble_predict.png")
plt.show()
