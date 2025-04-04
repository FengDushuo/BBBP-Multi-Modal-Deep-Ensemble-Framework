import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

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
fingerprint_pca = PCA(n_components=64)
image_pca = PCA(n_components=128)
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

# Define multimodal neural network
class MixedInputModel(nn.Module):
    def __init__(self, fingerprint_size, image_feature_size):
        super(MixedInputModel, self).__init__()

        # Fingerprint subnetwork
        self.fingerprint_fc = nn.Sequential(
            nn.Linear(fingerprint_size, 128),
            nn.ReLU()
        )

        # Image feature subnetwork
        self.image_fc = nn.Sequential(
            nn.Linear(image_feature_size, 128),
            nn.ReLU()
        )

        # Attention fusion module
        self.attention_fusion = MultiHeadAttentionFusion(256)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
criterion = nn.MSELoss()

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
nn_predictions, rf_predictions, xgb_predictions, actuals = [], [], [], []

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

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    rf_model.fit(np.hstack([X_fingerprints_train, X_images_train]), y_train)
    rf_predictions.extend(rf_model.predict(np.hstack([X_fingerprints_test, X_images_test])))

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=200, max_depth=15, learning_rate=0.1, random_state=42)
    xgb_model.fit(np.hstack([X_fingerprints_train, X_images_train]), y_train)
    xgb_predictions.extend(xgb_model.predict(np.hstack([X_fingerprints_test, X_images_test])))

# Stacking using Ridge regression
stacked_features = np.vstack([nn_predictions, rf_predictions, xgb_predictions]).T
stacking_model = Ridge(alpha=1.0)
stacking_model.fit(stacked_features, actuals)
stacked_predictions = stacking_model.predict(stacked_features)

# Save the best neural network model
torch.save(model.state_dict(), "best_nn_model_maccs.pth")

# Save the stacking model
with open("stacked_model_maccs_opt.pkl", "wb") as f:
    pickle.dump(stacking_model, f)

# Load and test saved models
# Load the neural network model
loaded_nn_model = MixedInputModel(fingerprint_size, image_feature_size).to(device)
loaded_nn_model.load_state_dict(torch.load("best_nn_model.pth"))
loaded_nn_model.eval()

# Load the stacking model
with open("stacked_model_maccs_opt.pkl", "rb") as f:
    loaded_stacked_model = pickle.load(f)

# Example prediction
sample_fingerprint = torch.tensor(fingerprints[0], dtype=torch.float32).to(device)
sample_image = torch.tensor(images[0], dtype=torch.float32).to(device)
with torch.no_grad():
    nn_output = loaded_nn_model(sample_fingerprint.unsqueeze(0), sample_image.unsqueeze(0))
    print(f"Neural Network Prediction: {nn_output.item():.4f}")

stacked_input = np.array([nn_output.item(), rf_predictions[0], xgb_predictions[0]]).reshape(1, -1)
stacked_output = loaded_stacked_model.predict(stacked_input)
print(f"Stacked Model Prediction: {stacked_output[0]:.4f}")

# Evaluate
mse = mean_squared_error(actuals, stacked_predictions)
r2 = r2_score(actuals, stacked_predictions)
print(f"Stacked Model - MSE: {mse:.4f}, R²: {r2:.4f}")

# Visualize results
plt.scatter(actuals, stacked_predictions, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Stacked Model Predicted vs Actual")
plt.savefig("stacked_model_predictions_maccs_opt.png")
plt.show()

