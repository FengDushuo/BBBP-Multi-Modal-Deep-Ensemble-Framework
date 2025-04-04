import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Step 1: 定义 FlowDataset 类
class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 2: 定义流模型模块
class FlowLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(FlowLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def set_dropout_prob(self, prob):
        """动态设置 Dropout 概率"""
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x, reverse=False):
        if reverse:
            print(f"[Reverse] Input shape: {x.shape}")
            x = self.activation(x)
            x = self.fc2(x)
            print(f"[Reverse] Output shape: {x.shape}")
            return x
        else:
            print(f"[Forward] Input shape: {x.shape}")
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            print(f"[Forward] Output shape: {x.shape}")
            return x


# Step 3: 定义完整的流模型
class FlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=3, num_classes=2, dropout_prob=0.5):
        super(FlowModel, self).__init__()
        
        self.layers = nn.ModuleList([
            FlowLayer(input_dim if i == 0 else hidden_dim, hidden_dim, dropout_prob=dropout_prob)
            for i in range(n_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, reverse=False):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: Input shape {x.shape}")
            x = layer(x, reverse=reverse)
            print(f"Layer {i}: Output shape {x.shape}")
        return self.output_layer(x)



# Step 4: 绘制学习曲线函数
def plot_learning_curve(estimator, title, output, X, y, ylim=None, cv=5, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid()
    if ylim:
        plt.ylim(ylim)
    plt.savefig(output + title + '.jpg')
    print(f"Learning curve saved as {output + title + '.jpg'}")

# Step 5: 定义训练和评估流程
class FlowClassifier:
    def __init__(self, input_dim, hidden_dim=128, n_layers=3, epochs=10, batch_size=32, learning_rate=1e-3, dropout_prob=0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = FlowModel(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            n_layers=self.n_layers, 
            dropout_prob=self.dropout_prob
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train, y_train):
        # 检查输入数据维度与模型输入维度是否一致
        assert X_train.shape[1] == self.input_dim, \
            f"Input dimension mismatch: model expects {self.input_dim}, but got {X_train.shape[1]}"

        train_dataset = FlowDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)  # 正向传播
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, X, reverse=False):
        self.model.eval()
        dataset = FlowDataset(X, np.zeros(len(X)))  # 伪标签
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        predictions = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch, reverse=reverse)  # 可选反向传播
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(preds)
        return predictions

    def evaluate(self, X, y, reverse=False):
        preds = self.predict(X, reverse=reverse)
        accuracy = accuracy_score(y, preds)
        balanced_acc = balanced_accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        mcc = matthews_corrcoef(y, preds)
        kappa = cohen_kappa_score(y, preds)
        auc = roc_auc_score(y, preds) if len(set(y)) > 1 else "N/A"

        return {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "Cohen's Kappa": kappa,
            "ROC AUC": auc
        }

    def save_model(self, model_dir='saved_model'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "flow_model.pt"))
        print(f"Model saved to {model_dir}")

    def load_model(self, model_dir='saved_model'):
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "flow_model.pt"), map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {model_dir}")

    def score(self, X, y, reverse=False):
        return accuracy_score(y, self.predict(X, reverse=reverse))

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout_prob': self.dropout_prob
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # 动态重建模型
        self.model = FlowModel(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            n_layers=self.n_layers, 
            dropout_prob=self.dropout_prob
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self


# Step 6: 主流程
def do_flow_train(fps_file, input_file, output_dir):
    fingerprints = np.load(fps_file, allow_pickle=True)
    data = pd.read_csv(input_file, sep="\t")
    data["Morgan_Fingerprint"] = list(fingerprints)
    X = np.array([np.array(fp) for fp in data["Morgan_Fingerprint"]])
    y = data["BBB+/BBB-"].values

    # 使用 LabelEncoder 将字符串标签映射为整数
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # 将 y 映射为整数
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=100)  # 降维到 100 维
    X_reduced = pca.fit_transform(X_scaled)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    print('train_test_split successfully!')

    # 定义参数网格
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'n_layers': [2, 3, 4],
        'epochs': [10, 20],
        'batch_size': [16],
        'learning_rate': [1e-3, 5e-4]
    }

    # 使用 GridSearchCV 进行超参数搜索
    grid_search = GridSearchCV(
        estimator=FlowClassifier(input_dim=X_reduced.shape[1]),  # 动态设置输入维度
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        error_score='raise'
    )
    
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # # 初始化模型
    # classifier = FlowClassifier(
    #     input_dim=X_train.shape[1], 
    #     hidden_dim=128, 
    #     n_layers=3, 
    #     epochs=10, 
    #     batch_size=16, 
    #     learning_rate=1e-3,
    #     dropout_prob=0.5
    # )

    # # 训练模型
    # classifier.fit(X_train, y_train)

    plot_learning_curve(best_model, 'FlowModel_Learning_Curve', output_dir, X_train, y_train)

    metrics = best_model.evaluate(X_test, y_test)
    # 保存 metrics 到 CSV
    metrics_file = os.path.join(output_dir, "metrics.csv")
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics saved to {metrics_file}")

    print("Evaluation Metrics:", metrics)

    best_model.save_model(model_dir=output_dir+"flow_model")
    return metrics

# Example usage
metrics = do_flow_train(
    "D:/a_work/1-phD/project/5-VirtualScreening/morgan_fingerprints.npy",
    "D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_classification.tsv",
    "D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/output/"
)
