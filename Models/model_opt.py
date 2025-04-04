import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# 评估模型性能
def evaluate_model(model, X_train, y_train, X_test, y_test, output, model_name):
    """
    评估模型性能并保存混淆矩阵
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # 计算性能指标
    test_accuracy = accuracy_score(y_test, test_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
    test_recall = recall_score(y_test, test_pred, average='binary', zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average='binary', zero_division=0)
    test_mcc = matthews_corrcoef(y_test, test_pred)
    test_cohen_kappa = cohen_kappa_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else roc_auc_score(y_test, test_pred)

    # 保存混淆矩阵
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name} Confusion Matrix", fontname='Times New Roman', fontsize=16)
    plt.xlabel("Predicted", fontname='Times New Roman', fontsize=14)
    plt.ylabel("True", fontname='Times New Roman', fontsize=14)
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_confusion_matrix.png")
    plt.close()

    return {
        "Accuracy": test_accuracy,
        "Balanced Accuracy": test_balanced_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1 Score": test_f1,
        "MCC": test_mcc,
        "Cohen's Kappa": test_cohen_kappa,
        "ROC AUC": test_auc
    }

def plot_performance(metrics, output_path):
    """
    绘制性能曲线
    """
    df = pd.DataFrame(metrics).T
    ax = df.plot(kind='bar', figsize=(12, 8), rot=45, legend=True)
    ax.set_title('Model Performance Metrics', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel('Score', fontname='Times New Roman', fontsize=14)
    ax.set_xlabel('Models', fontname='Times New Roman', fontsize=14)
    ax.tick_params(axis='x', labelsize=12, labelrotation=45, which='major', labelcolor='black')
    ax.tick_params(axis='y', labelsize=12, which='major', labelcolor='black')
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_path}/performance_metrics.png")
    #plt.show()

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring="f1", output_path="./", n_jobs=-1):
    """
    绘制学习曲线
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plt.xlabel("Training Examples", fontsize=14)
    plt.ylabel("Score", fontsize=14)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score", color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score", color="g")

    plt.legend(loc="best", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title.replace(' ', '_')}.png")
    plt.close()
    #plt.show()

def train_and_evaluate(input_file, fingerprint_file, output):
    # 加载数据
    fingerprints = np.load(fingerprint_file, allow_pickle=True)
    data = pd.read_csv(input_file, sep="\t")
    data["Morgan_Fingerprint"] = list(fingerprints)

    X = np.array([np.array(fp) for fp in data["Morgan_Fingerprint"]])
    y = data["BBB+/BBB-"].values

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用 PCA 降维
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)

    # 处理类别不平衡
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_reduced, y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    # 将标签编码为数值
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # 将 y_train 映射为 0 和 1
    y_test = label_encoder.transform(y_test)# 将标签编码为数值
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # 定义基模型
    models = {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVC": SVC(probability=True),
        "BernoulliNB": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP": MLPClassifier(max_iter=2000, learning_rate_init=0.001, batch_size=32)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        param_grid = {}
        if name == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        elif name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        elif name == "SVC":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        elif name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif name == "Gradient Boosting":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        elif name == "MLP":
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'batch_size': [32, 64, 128]
            }

        random_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, scoring='f1', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, output, name)
        print(f"{name} metrics: {metrics}")
        results[name] = metrics
        estimators.append((name, best_model))
        joblib.dump(best_model, f"{output}/{name}_model.pkl")

        # 绘制学习曲线
        plot_learning_curve(
            best_model, X_train, y_train, title=f"{name} Learning Curve", output_path=output
        )

    # 集成学习
    print("Training StackingClassifier...")
    estimators = [(name, joblib.load(f"{output}/{name}_model.pkl")) for name in results.keys()]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3), cv=5)
    stacking_model.fit(X_train, y_train)
    stacking_metrics = evaluate_model(stacking_model, X_train, y_train, X_test, y_test)
    print(f"StackingClassifier metrics: {stacking_metrics}")
    results["StackingClassifier"] = stacking_metrics
    joblib.dump(stacking_model, f"{output}/stacking_model.pkl")
    voting_model = VotingClassifier(estimators=estimators, voting='soft')
    voting_model.fit(X_train, y_train)
    voting_metrics = evaluate_model(voting_model, X_train, y_train, X_test, y_test)
    print(f"VotingClassifier metrics: {voting_metrics}")
    results["VotingClassifier"] = voting_metrics
    joblib.dump(voting_model, f"{output}/voting_model.pkl")

    # 绘制性能表格和曲线
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(f"{output}/model_performance_metrics.csv")
    plot_performance(results, output)

    return results

train_and_evaluate("B3DB_classification.tsv","morgan_fingerprints.npy","./output_opt/")

