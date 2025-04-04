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

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = 'processed_data.pkl'
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

# Define multimodal neural network
class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()
        # Fingerprint subnetwork
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
        # Image feature subnetwork
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
        # Fusion network
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

# Initialize variables
fingerprint_size = fingerprints.shape[1]
image_feature_size = images.shape[1]
criterion = nn.MSELoss()

# Function to compare float values up to two decimal places
def is_match(pred, actual):
    return round(pred, 2) == round(actual, 2)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
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

            # Compare predictions and actual values (rounded to 2 decimals)
            for pred, actual in zip(preds.numpy(), labels_batch.numpy()):
                if is_match(pred, actual):
                    nn_correct += 1

    # Calculate accuracy for neural network
    nn_accuracy = nn_correct / len(y_actual)
    print(f"Neural Network Accuracy (matching to 2 decimals): {nn_accuracy:.2%}")

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
        learning_rate=0.05,
        max_depth=10,
        tree_method="gpu_hist",  # Enable GPU acceleration
        random_state=42
    )
    xgb_model.fit(X_train_combined, y_train)
    xgb_preds = xgb_model.predict(X_test_combined)
    xgb_predictions.extend(xgb_preds)

# Ensemble model prediction
ensemble_predictions = (0.4 * np.array(nn_predictions) +
                        0.3 * np.array(rf_predictions) +
                        0.3 * np.array(xgb_predictions))

# Evaluate ensemble model (rounded to 2 decimals)
ensemble_predictions_rounded = np.round(ensemble_predictions, 2)
actuals_rounded = np.round(actuals, 2)

# Calculate matching accuracy for ensemble model
ensemble_correct = sum(ensemble_predictions_rounded == actuals_rounded)
ensemble_accuracy = ensemble_correct / len(actuals)
print(f"Ensemble Model Accuracy (matching to 2 decimals): {ensemble_accuracy:.2%}")
# Evaluate model
mse = mean_squared_error(actuals, ensemble_predictions)
r2 = r2_score(actuals, ensemble_predictions)
print(f"Ensemble Model - Mean Squared Error: {mse:.4f}")
print(f"Ensemble Model - R² Score: {r2:.4f}")

# Additional metrics based on rounded values
mse_rounded = mean_squared_error(actuals_rounded, ensemble_predictions_rounded)
r2_rounded = r2_score(actuals_rounded, ensemble_predictions_rounded)
print(f"Ensemble Model (Rounded to 2 decimals) - Mean Squared Error: {mse_rounded:.4f}")
print(f"Ensemble Model (Rounded to 2 decimals) - R² Score: {r2_rounded:.4f}")

# Visualize results
plt.scatter(actuals_rounded, ensemble_predictions_rounded, alpha=0.5)
plt.xlabel("Actual Values (Rounded to 2 Decimals)")
plt.ylabel("Predicted Values (Rounded to 2 Decimals)")
plt.title("Ensemble Predicted vs Actual (Rounded)")
plt.savefig("ensemble_predict_rounded.png")
plt.show()
