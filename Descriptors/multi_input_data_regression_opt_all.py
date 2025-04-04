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

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = 'processed_data_maccs_opt_lso_fixed.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Extract features and targets
fingerprints = np.stack(data['MACCS_Normalized'])
images = np.stack(data['Image_Features_Normalized'])
labels = data['logBB'].values

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

# Define multimodal neural network with attention fusion
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


# Initialize variables
fingerprint_size = fingerprints.shape[1]
image_feature_size = 128
criterion = nn.MSELoss()

# K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
nn_predictions, rf_predictions, xgb_predictions, cat_predictions, actuals = [], [], [], [], []

for train_idx, test_idx in kf.split(fingerprints):
    # Data split
    X_fingerprints_train, X_fingerprints_test = fingerprints[train_idx], fingerprints[test_idx]
    X_images_train, X_images_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Create dataloaders
    train_dataset = MixedDataset(X_fingerprints_train, X_images_train, y_train)
    test_dataset = MixedDataset(X_fingerprints_test, X_images_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize neural network model and optimizer
    model = MixedInputModel(fingerprint_size, image_feature_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train neural network
    model.train()
    for epoch in range(50):  # Reduce epochs for faster convergence
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
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

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
    nn_predictions.extend(nn_preds)
    actuals.extend(y_actual)

    # Random forest model
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
    X_train_combined = np.hstack([X_fingerprints_train, X_images_train])
    X_test_combined = np.hstack([X_fingerprints_test, X_images_test])
    rf_model.fit(X_train_combined, y_train)
    rf_preds = rf_model.predict(X_test_combined)
    rf_predictions.extend(rf_preds)

    # XGBoost model
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.01, max_depth=30, tree_method="hist", random_state=42)
    xgb_model.fit(X_train_combined, y_train)
    xgb_preds = xgb_model.predict(X_test_combined)
    xgb_predictions.extend(xgb_preds)

    # CatBoost model
    cat_model = CatBoostRegressor(iterations=300, learning_rate=0.01, depth=10, verbose=0, random_state=42)
    cat_model.fit(X_train_combined, y_train)
    cat_preds = cat_model.predict(X_test_combined)
    cat_predictions.extend(cat_preds)

# Stacking ensemble
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

# Evaluate stacked model
mse_stacked = mean_squared_error(actuals, stacked_preds)
r2_stacked = r2_score(actuals, stacked_preds)
print(f"Stacked Model - Mean Squared Error: {mse_stacked:.4f}")
print(f"Stacked Model - R² Score: {r2_stacked:.4f}")

# Visualize results
plt.scatter(actuals, stacked_preds, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Stacked Predicted vs Actual")
plt.savefig("stacked_predict.png")
plt.show()
