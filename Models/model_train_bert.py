import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

# Step 2: 定义 Dataset 类
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # 检查 texts 是否为 Pandas Series，如果是则重置索引
        if isinstance(texts, pd.Series):
            self.texts = texts.reset_index(drop=True)
        else:
            self.texts = texts

        # 检查 labels 是否为 Pandas Series，如果是则重置索引
        if isinstance(labels, pd.Series):
            self.labels = labels.reset_index(drop=True)
        else:
            self.labels = labels

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Step 3: 定义 BERT 分类器包装类，使其兼容 sklearn
class SklearnBertClassifier:
    def __init__(self, epochs=3, batch_size=4, learning_rate=2e-5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = None

    def fit(self, X, y):
        dataset = ReviewDataset(X, y, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * self.epochs)

        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        return self

    def save_model(self, model_dir='saved_model'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(f"Saving model to {model_dir}...")
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def load_model(self, model_dir='saved_model'):
        print(f"Loading model from {model_dir}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)

    def predict(self, X):
        dataset = ReviewDataset(X, [0] * len(X), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

    # 让该类兼容 scikit-learn 的 API
    def get_params(self, deep=True):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def evaluate(self, X, y):
        """
        评估模型性能并计算多个指标
        """
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        balanced_acc = balanced_accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        mcc = matthews_corrcoef(y, predictions)
        kappa = cohen_kappa_score(y, predictions)
        auc = roc_auc_score(y, predictions)

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

def do_bert_train(fps_file,input_file,output):
    # 加载描述符数据
    # morgan_fps = np.load("morgan_fingerprints.npy", allow_pickle=True)
    # maccs_fps = np.load("maccs_fingerprints.npy", allow_pickle=True)
    # rdkit_fps = np.load("rdkit_fingerprints.npy", allow_pickle=True)
    morgan_fps = np.load(fps_file, allow_pickle=True)

    # 目标标签
    data = pd.read_csv(input_file, sep="\t")  # 包含 SMILES 和目标值
    data["Morgan_Fingerprint"] = list(morgan_fps)
    # data["MACCS_Fingerprint"] = list(maccs_fps)
    # data["RDKit_Fingerprint"] = list(rdkit_fps)

    # 展平特征（以 Morgan 指纹为例）
    X = np.array([np.array(fp) for fp in data["Morgan_Fingerprint"]])  # 特征
    y = data["BBB+/BBB-"].values  # 标签

    #标准化特征可以加速训练并提高模型效果
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #特征维度过高（如 2048 位的 Morgan 指纹），可以通过 PCA 或特征选择降维
    pca = PCA(n_components=100)  # 降维到100个主成分
    X_reduced = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # 将标签编码为数值
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # 将 y_train 映射为 0 和 1
    y_test = label_encoder.transform(y_test)# 将标签编码为数值

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    print('train_test_split successfully!')

    # 定义BERT超参数调优网格
    param_grid_bert = {
        'epochs': [10, 50, 100],
        'batch_size': [2, 4],
        'learning_rate': [1e-5, 2e-5],
    }

    # 创建BERT模型对象
    bert_model = SklearnBertClassifier()

    # 使用GridSearchCV进行超参数调优
    grid_bert = GridSearchCV(bert_model, param_grid_bert, cv=3, scoring='f1', n_jobs=1)
    grid_bert.fit(X_train, y_train)

    # 获取最优的BERT模型
    best_bert_model = grid_bert.best_estimator_

    # 评估最优BERT模型
    bert_metrics = best_bert_model.evaluate(X_test, y_test)
    train_score = best_bert_model.score(X_train, y_train)
    test_score = best_bert_model.score(X_test, y_test)
    # 绘制学习曲线
    plot_learning_curve(best_bert_model, 'BERT_Learning_Curve', output, X_train, y_train)

    # 加载已保存的模型
    best_bert_model.save_model(model_dir=output+"bert_model")

    return bert_metrics,best_bert_model,train_score,test_score

do_bert_train("morgan_fingerprints.npy","B3DB_classification.tsv","./output/")
