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

# Apply PCA for dimensionality reduction
#fingerprint_pca = PCA(n_components=64)
#image_pca = PCA(n_components=128)
#fingerprints = fingerprint_pca.fit_transform(fingerprints)
#images = image_pca.fit_transform(images)

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

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, fingerprint_dim, image_dim, hidden_dim=128):
        super(MultiModalAttentionFusion, self).__init__()
        self.fingerprint_attention = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.image_attention = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.cross_modal_attention = nn.Sequential(
            nn.Linear(fingerprint_dim + image_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, fingerprint_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fingerprint, image):
        # Attention weights
        fingerprint_weight = self.fingerprint_attention(fingerprint).unsqueeze(1)  # [batch, 1]
        image_weight = self.image_attention(image).unsqueeze(1)  # [batch, 1]

        # Cross-modal attention
        combined = torch.cat((fingerprint, image), dim=1)  # [batch, fingerprint_dim + image_dim]
        cross_weight = self.cross_modal_attention(combined)  # [batch, fingerprint_dim]

        # 确保 cross_weight 是二维
        if cross_weight.dim() == 3:
            cross_weight = cross_weight.squeeze(1)

        # Normalize weights
        attention_weights = torch.cat([fingerprint_weight, image_weight], dim=1)  # [batch, 2]
        attention_weights = self.softmax(attention_weights)  # [batch, 2]

        # Weighted features
        fingerprint_weighted = attention_weights[:, 0:1] * fingerprint  # [batch, seq_len, fingerprint_dim]
        image_weighted = attention_weights[:, 1:2] * image  # [batch, seq_len, image_dim]

        # 汇总 seq_len 维度
        fingerprint_weighted = fingerprint_weighted.mean(dim=1)  # [batch, fingerprint_dim]
        image_weighted = image_weighted.mean(dim=1)              # [batch, image_dim]

        # 打印张量形状
        #print(f"After reducing seq_len: fingerprint_weighted shape: {fingerprint_weighted.shape}")
        #print(f"After reducing seq_len: image_weighted shape: {image_weighted.shape}")
        #print(f"cross_weight shape: {cross_weight.shape}")

        # Concatenate features
        fused_feature = torch.cat((fingerprint_weighted, image_weighted, cross_weight), dim=1)  # [batch, fingerprint_dim + image_dim + fingerprint_dim]

        #print(f"fused_feature shape: {fused_feature.shape}")
        return fused_feature



class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()
        nhead = 8
        while fingerprint_size % nhead != 0 and nhead > 1:
            nhead -= 1

        if fingerprint_size % nhead != 0:
            raise ValueError(f"fingerprint_size={fingerprint_size} must be divisible by nhead={nhead}.")

        self.fingerprint_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fingerprint_size, nhead=nhead),
            num_layers=12
        )
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * (image_feature_size // 8) * (image_feature_size // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.attention_fusion = MultiModalAttentionFusion(512, 512)

        self.fc = nn.Sequential(
            nn.Linear(512 + 512 + 512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, fingerprint, image):
        fingerprint = fingerprint.unsqueeze(1)  # [batch, seq_len, fingerprint_size]
        fingerprint_out = self.fingerprint_transformer(fingerprint).squeeze(1)  # [batch, fingerprint_size]
        fingerprint_out = self.fingerprint_fc(fingerprint_out)

        image = image.view(-1, 3, 128, 128)
        image_out = self.image_cnn(image)

        fused_features = self.attention_fusion(fingerprint_out, image_out)
        #print(f"fused_features shape: {fused_features.shape}")
        output = self.fc(fused_features)
        return output



# Initialize variables
fingerprint_size = fingerprints.shape[1]
image_feature_size = 128  # Assuming images are resized to 128x128
criterion = nn.MSELoss()

# K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
nn_predictions, rf_predictions, actuals, xgb_predictions = [], [], [], []

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Train neural network
    model.train()
    for epoch in range(50):
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
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Test neural network
    model.eval()
    nn_preds, y_actual, nn_correct = [], [], 0
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

    # XGBoost model with GPU acceleration
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=30,
        tree_method="hist",  # Enable GPU acceleration
        device="cuda",
        random_state=42
    )
    xgb_model.fit(X_train_combined, y_train)
    xgb_preds = xgb_model.predict(X_test_combined)
    xgb_predictions.extend(xgb_preds)

# Ensemble model prediction
ensemble_predictions = (0.4 * np.array(nn_predictions) +
                        0.3 * np.array(rf_predictions) +
                        0.3 * np.array(xgb_predictions))

# Evaluate ensemble model
mse = mean_squared_error(actuals, ensemble_predictions)
r2 = r2_score(actuals, ensemble_predictions)
print(f"Ensemble Model - Mean Squared Error: {mse:.4f}")
print(f"Ensemble Model - R² Score: {r2:.4f}")

# Visualize results
plt.scatter(actuals, ensemble_predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ensemble Predicted vs Actual")
plt.savefig("ensemble_predict.png")
plt.show()

