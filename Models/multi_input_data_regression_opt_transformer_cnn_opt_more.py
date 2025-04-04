import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = 'processed_data_maccs_opt.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Extract features and targets
fingerprints = np.stack(data['MACCS_Normalized'])
images = np.stack(data['Image_Features_Normalized'])
labels = data['logBB'].values

# Apply PCA for dimensionality reduction
# Adjust PCA dimensions
fingerprint_pca = PCA(n_components=128)  # Try increasing components
image_pca = PCA(n_components=256)
fingerprints = fingerprint_pca.fit_transform(fingerprints)
images = image_pca.fit_transform(images)


# Define dataset class
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

# MultiHeadAttentionFusion module
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttentionFusion, self).__init__()
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ) for _ in range(num_heads)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        attention_weights = torch.cat([head(combined).unsqueeze(1) for head in self.attention_heads], dim=1)
        attention_weights = self.softmax(attention_weights)
        combined_weighted = torch.sum(attention_weights * combined.unsqueeze(1), dim=1)
        return combined_weighted

# Neural network model
# Neural network model updates
class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.image_fc = nn.Sequential(
            nn.Linear(image_feature_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.attention_fusion = MultiHeadAttentionFusion(512)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, fingerprint, image):
        fingerprint_out = self.fingerprint_fc(fingerprint)
        image_out = self.image_fc(image)
        fused = self.attention_fusion(fingerprint_out, image_out)
        output = self.fc(fused)
        return output

# Initialize variables
fingerprint_size = fingerprints.shape[1]
image_feature_size = images.shape[1]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Algorithms to evaluate
algorithms = {
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=15, learning_rate=0.1, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'CatBoost': CatBoostRegressor(iterations=500, depth=10, learning_rate=0.05, verbose=0),
    #'LightGBM': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
}

# Store results
model_results = {}
all_predictions = {name: [] for name in algorithms.keys()}
all_predictions['NeuralNetwork'] = []
actuals = []

for train_idx, test_idx in kf.split(fingerprints):
    X_fingerprints_train, X_fingerprints_test = fingerprints[train_idx], fingerprints[test_idx]
    X_images_train, X_images_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Train and evaluate each algorithm
    for name, model in algorithms.items():
        # Combine features for non-neural network models
        train_features = np.hstack([X_fingerprints_train, X_images_train])
        test_features = np.hstack([X_fingerprints_test, X_images_test])

        # Train model
        model.fit(train_features, y_train)

        # Predict and evaluate
        preds = model.predict(test_features)
        all_predictions[name].extend(preds)

    # Neural network
    train_dataset = MixedDataset(X_fingerprints_train, X_images_train, y_train)
    test_dataset = MixedDataset(X_fingerprints_test, X_images_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MixedInputModel(fingerprint_size, image_feature_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(50):  # 增加 Epoch 数量
        total_loss = 0
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
        scheduler.step(total_loss)

    # Test neural network
    model.eval()
    nn_preds, y_actual = [], []
    with torch.no_grad():
        for fingerprints_batch, images_batch, labels_batch in test_loader:
            fingerprints_batch, images_batch = fingerprints_batch.to(device), images_batch.to(device)
            preds = model(fingerprints_batch, images_batch).squeeze().cpu()
            labels_batch = labels_batch.cpu()
            nn_preds.extend(preds.numpy())
            y_actual.extend(labels_batch.numpy())

    all_predictions['NeuralNetwork'].extend(nn_preds)
    actuals.extend(y_actual)

# Select top-performing algorithms
average_results = {name: np.mean(all_predictions[name]) for name in all_predictions}
top_algorithms = sorted(average_results.items(), key=lambda x: x[1], reverse=True)[:3]

# Collect predictions from top algorithms
selected_predictions = [all_predictions[algo[0]] for algo in top_algorithms]
stacked_features = np.vstack(selected_predictions).T

# Validate consistency in dimensions
assert stacked_features.shape[0] == len(actuals), "Mismatch between features and actual values!"

stacking_model = Ridge(alpha=1.0)
stacking_model.fit(stacked_features, actuals)
stacked_predictions = stacking_model.predict(stacked_features)

# Evaluate final stacked model
mse = mean_squared_error(actuals, stacked_predictions)
r2 = r2_score(actuals, stacked_predictions)
print(f"Final Stacked Model - MSE: {mse:.4f}, R²: {r2:.4f}")

# Visualize results
plt.scatter(actuals, stacked_predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Stacked Model Predicted vs Actual")
plt.savefig("stacked_model_predictions_maccs_opt.png")
plt.show()

