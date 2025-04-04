import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import xgboost as xgb

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 22

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Multi-Head Attention Fusion
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128):
        super(MultiHeadAttentionFusion, self).__init__()
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)  # Combine fingerprint and image features
        attention_weights = torch.cat([head(combined).unsqueeze(1) for head in self.attention_heads], dim=1)
        attention_weights = self.softmax(attention_weights)
        combined_weighted = torch.sum(attention_weights * combined.unsqueeze(1), dim=1)
        return combined_weighted

# 定义模型
class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()
        nhead = max(1, fingerprint_size // 8)
        while fingerprint_size % nhead != 0:
            nhead -= 1

        self.fingerprint_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fingerprint_size, nhead=nhead),
            num_layers=6
        )
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, 128),
            nn.ReLU()
        )

        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (image_feature_size // 4) * (image_feature_size // 4), 128),
            nn.ReLU()
        )

        self.attention_fusion = MultiHeadAttentionFusion(256, num_heads=4)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),  # Adjusted input dimension from 128 to 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, fingerprint, image):
        fingerprint = fingerprint.unsqueeze(1)
        fingerprint_out = self.fingerprint_transformer(fingerprint).squeeze(1)
        fingerprint_out = self.fingerprint_fc(fingerprint_out)

        image = image.view(-1, 3, 128, 128)
        image_out = self.image_cnn(image)

        fused = self.attention_fusion(fingerprint_out, image_out)
        output = self.fc(fused)
        return output

# 输入文件列表和对应列名
input_files = [
    ("processed_data_maccs_opt_lso_fixed_1.pkl", "MACCS_Normalized"),
    #("processed_data_morgan_opt_lso_fixed_1.pkl", "Morgan_Normalized"),
    #("processed_data_rdkit_opt_lso_fixed_1.pkl", "rdkit_Normalized")
]
image_feature_col = "Image_Features_Normalized_PCA"


# 遍历文件进行训练
for data_path, fingerprint_key in input_files:
    print(f"Processing file: {data_path}, using fingerprint: {fingerprint_key}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 后续代码中可以使用 `fingerprint_key` 访问正确的列
    fingerprints = np.stack(data[fingerprint_key])
    images = np.stack(data['Image_Features_Normalized'])
    labels = data['logBB'].values

    fingerprint_size = fingerprints.shape[1]
    image_feature_size = 128
    criterion = nn.MSELoss()

    # K-fold 交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 创建预测和实际值数组
    num_samples = fingerprints.shape[0]
    nn_predictions = np.zeros(num_samples)
    rf_predictions = np.zeros(num_samples)
    xgb_predictions = np.zeros(num_samples)
    cat_predictions = np.zeros(num_samples)
    actuals = np.zeros(num_samples)

    # 在现有代码基础上添加训练过程的可视化

    # 修改训练部分代码，记录每个 epoch 的损失值
    for train_idx, test_idx in kf.split(fingerprints):
        # 数据拆分
        X_fingerprints_train, X_fingerprints_test = fingerprints[train_idx], fingerprints[test_idx]
        X_images_train, X_images_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # 数据加载器
        train_dataset = MixedDataset(X_fingerprints_train, X_images_train, y_train)
        test_dataset = MixedDataset(X_fingerprints_test, X_images_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化神经网络模型和优化器
        model = MixedInputModel(fingerprint_size, image_feature_size).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        # 记录损失值
        train_losses = []

        # 修改训练过程，增加验证损失记录
        val_losses = []
        model.train()
        for epoch in range(50):
            total_loss = 0
            val_loss = 0
            for fingerprints_batch, images_batch, labels_batch in train_loader:
                fingerprints_batch, images_batch, labels_batch = (
                    fingerprints_batch.to(device), images_batch.to(device), labels_batch.to(device)
                )
                optimizer.zero_grad()
                predictions = model(fingerprints_batch, images_batch).squeeze()
                loss = criterion(predictions, labels_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 计算验证损失
            model.eval()
            with torch.no_grad():
                for fingerprints_batch, images_batch, labels_batch in test_loader:
                    fingerprints_batch, images_batch, labels_batch = (
                        fingerprints_batch.to(device), images_batch.to(device), labels_batch.to(device)
                    )
                    predictions = model(fingerprints_batch, images_batch).squeeze()
                    loss = criterion(predictions, labels_batch)
                    val_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', label='Validation Loss', color='red')
        plt.xlabel("Epochs", fontsize=20, fontname="Times New Roman")
        plt.ylabel("Loss", fontsize=20, fontname="Times New Roman")
        plt.title(f"Training and Validation Loss Curve ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.legend(fontsize=18, loc="upper right", prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"loss_curve_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()


        # 测试神经网络
        # 在测试阶段记录实际值和预测值
        model.eval()
        nn_fold_predictions = []
        nn_actuals = []
        with torch.no_grad():
            for fingerprints_batch, images_batch, labels_batch in test_loader:
                fingerprints_batch, images_batch = fingerprints_batch.to(device), images_batch.to(device)
                preds = model(fingerprints_batch, images_batch).squeeze().cpu()
                nn_fold_predictions.extend(preds.numpy())
                nn_actuals.extend(labels_batch.numpy())

        # 更新预测值
        nn_predictions[test_idx] = nn_fold_predictions
        actuals[test_idx] = y_test
        # 保存模型
        with open("nn_model_maccs.pkl", "wb") as f:
            pickle.dump(model, f)

        # 绘制实际值与预测值对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(nn_actuals, nn_fold_predictions, alpha=0.6, edgecolor='k', label="Predictions")
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--', label="Perfect Fit")
        plt.xlabel("Actual Values", fontsize=20, fontname="Times New Roman")
        plt.ylabel("Predicted Values", fontsize=20, fontname="Times New Roman")
        plt.title(f"Actual vs Predicted ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.legend(fontsize=18, loc="upper left", prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"actual_vs_predicted_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()

        # 训练 Random Forest
        rf_model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
        X_train_combined = np.hstack([X_fingerprints_train, X_images_train])
        X_test_combined = np.hstack([X_fingerprints_test, X_images_test])
        rf_model.fit(X_train_combined, y_train)
        rf_preds = rf_model.predict(X_test_combined)
        rf_predictions[test_idx] = rf_preds
        # 保存模型
        with open("rf_model_maccs.pkl", "wb") as f:
            pickle.dump(rf_model, f)

        # 使用随机森林计算特征重要性
        feature_importances = rf_model.feature_importances_

        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(feature_importances)), feature_importances, color='skyblue', edgecolor='k')
        plt.xlabel("Feature Index", fontsize=20, fontname="Times New Roman")
        plt.ylabel("Importance", fontsize=20, fontname="Times New Roman")
        plt.title(f"Feature Importance ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"feature_importance_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()


        # 训练 XGBoost
        # XGBoost 训练过程可视化
        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=30,
            tree_method="hist",
            random_state=42,
            eval_metric="rmse"
        )

        xgb_eval_result = {}
        xgb_model.fit(
            X_train_combined, y_train,
            eval_set=[(X_train_combined, y_train), (X_test_combined, y_test)],
            verbose=False
        )

        # 获取评估结果
        xgb_eval_result = xgb_model.evals_result()

        # 提取训练和验证 RMSE
        train_rmse = xgb_eval_result['validation_0']['rmse']  # 直接访问列表
        test_rmse = xgb_eval_result['validation_1']['rmse']

        # 绘制训练过程曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_rmse, label="Training RMSE", marker='o')
        plt.plot(test_rmse, label="Validation RMSE", marker='o', color='red')
        plt.xlabel("Iterations", fontsize=20, fontname="Times New Roman")
        plt.ylabel("RMSE", fontsize=20, fontname="Times New Roman")
        plt.title("XGBoost Training Curve", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.legend(fontsize=18, loc="upper right", prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"xgboost_training_curve_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()

        xgb_preds = xgb_model.predict(X_test_combined)
        xgb_predictions[test_idx] = xgb_preds
        # 保存模型
        with open("xgb_model_maccs.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        # 训练 CatBoost
        # CatBoost 手动训练可视化
        cat_model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.01,
            depth=10,
            verbose=50,
            random_state=42
        )

        eval_result = cat_model.fit(
            X_train_combined,
            y_train,
            eval_set=(X_test_combined, y_test),
            use_best_model=True,
            plot=False  # 禁用默认绘图
        )

        feature_importances = cat_model.get_feature_importance(prettified=True)
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importances["Feature Id"], feature_importances["Importances"], color="skyblue", edgecolor="k")
        plt.xlabel("Importance", fontsize=20, fontname="Times New Roman")
        plt.ylabel("Feature Index", fontsize=20, fontname="Times New Roman")
        plt.title(f"CatBoost Feature Importance ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"catboost_feature_importance_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()


        # 提取训练过程
        train_error = cat_model.evals_result_['learn']['RMSE']
        test_error = cat_model.evals_result_['validation']['RMSE']

        # 手动绘制曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_error, label="Training RMSE", marker='o', color='blue')
        plt.plot(test_error, label="Validation RMSE", marker='o', color='red')
        plt.xlabel("Iterations", fontsize=20, fontname="Times New Roman")
        plt.ylabel("RMSE", fontsize=20, fontname="Times New Roman")
        plt.title("CatBoost Training Curve", fontsize=22, fontname="Times New Roman", fontweight="bold")
        plt.legend(fontsize=18, loc="upper right", prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"catboost_training_curve_manual_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
        plt.close()


        cat_preds = cat_model.predict(X_test_combined)
        cat_predictions[test_idx] = cat_preds
        # 保存模型
        with open("cat_model_maccs.pkl", "wb") as f:
            pickle.dump(cat_model, f)

    # 堆叠集成
    stacked_model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)),
            ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=30, random_state=42)),
            ("cat", CatBoostRegressor(iterations=300, learning_rate=0.01, depth=10, verbose=0, random_state=42))
        ],
        final_estimator=LinearRegression()
    )
    stacked_model.fit(np.vstack([nn_predictions, rf_predictions, xgb_predictions, cat_predictions]).T, actuals)
    stacked_preds = stacked_model.predict(np.vstack([nn_predictions, rf_predictions, xgb_predictions, cat_predictions]).T)
    import pickle

    # 保存模型
    with open("stacked_model_maccs.pkl", "wb") as f:
        pickle.dump(stacked_model, f)

    # 评估结果
    mse_stacked = mean_squared_error(actuals, stacked_preds)
    r2_stacked = r2_score(actuals, stacked_preds)
    print(f"File: {data_path}")
    print(f"Stacked Model - Mean Squared Error: {mse_stacked:.4f}")
    print(f"Stacked Model - R² Score: {r2_stacked:.4f}")

    from sklearn.model_selection import learning_curve

    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=stacked_model,
        X=np.vstack([nn_predictions, rf_predictions, xgb_predictions, cat_predictions]).T,
        y=actuals,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    # 计算平均分数和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1,
                    color="blue")
    plt.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1,
                    color="red")
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score", color="blue")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score", color="red")
    plt.xlabel("Training Size", fontsize=20, fontname="Times New Roman")
    plt.ylabel("R² Score", fontsize=20, fontname="Times New Roman")
    plt.title("Stacked Model Learning Curve", fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.legend(fontsize=18, loc="lower right", prop={'family': 'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.7)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"stacked_model_learning_curve_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(actuals, bins=30, alpha=0.5, label="Actual Values", color="blue", edgecolor="k")
    plt.hist(stacked_preds, bins=30, alpha=0.5, label="Predicted Values", color="orange", edgecolor="k")
    plt.xlabel("Value", fontsize=20, fontname="Times New Roman")
    plt.ylabel("Frequency", fontsize=20, fontname="Times New Roman")
    plt.title(f"Actual vs Predicted Distribution ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.legend(fontsize=18, loc="upper right", prop={'family': 'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.7)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"distribution_comparison_{os.path.basename(data_path).replace('.pkl', '')}.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 可视化结果
    plt.scatter(actuals, stacked_preds, alpha=0.5)
    plt.xlabel("Actual Values", fontsize=20, fontname="Times New Roman")
    plt.ylabel("Predicted Values", fontsize=20, fontname="Times New Roman")
    plt.title(f"Stacked Predicted vs Actual ({os.path.basename(data_path)})", fontsize=22, fontname="Times New Roman", fontweight="bold")
    # 设置刻度标签的字体大小
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig(f"stacked_predict_{os.path.basename(data_path).replace('.pkl', '')}_{r2_stacked:.4f}_{mse_stacked:.4f}.png", dpi=600)
    plt.close()
