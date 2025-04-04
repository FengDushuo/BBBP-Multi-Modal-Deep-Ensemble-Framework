import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.preprocessing import LabelEncoder
import shap
from matplotlib import rcParams
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
    plt.title(f"{model_name} Confusion Matrix", fontname='Times New Roman', fontsize=22)
    plt.xlabel("Predicted", fontname='Times New Roman', fontsize=20)
    plt.ylabel("True", fontname='Times New Roman', fontsize=20)
    plt.xticks(fontsize=18, fontname='Times New Roman')
    plt.yticks(fontsize=18, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_confusion_matrix.png",dpi=600)
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
    ax.set_title('Model Performance Metrics', fontname='Times New Roman', fontsize=22)
    ax.set_ylabel('Score', fontname='Times New Roman', fontsize=20)
    ax.set_xlabel('Models', fontname='Times New Roman', fontsize=20)
    ax.tick_params(axis='x', labelsize=18, labelrotation=45, which='major', labelcolor='black')
    ax.tick_params(axis='y', labelsize=18, which='major', labelcolor='black')

    # 将图例放在图表右侧
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(f"{output_path}/performance_metrics.png", dpi=600)
    plt.close()


def plot_learning_curve(estimator, X, y, title, output_path, cv=5, scoring="f1", n_jobs=-1):
    """
    绘制学习曲线并保存训练损失和验证得分到文件
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontname='Times New Roman', fontsize=22)
    plt.xlabel("Training Examples", fontname='Times New Roman', fontsize=20)
    plt.ylabel("Score", fontname='Times New Roman', fontsize=20)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # 计算训练和验证得分的平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘制曲线
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score", color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score", color="g")

    plt.legend(loc="best", fontsize=12, prop={'family': 'Times New Roman'})
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title.replace(' ', '_')}_learning_curve.png",dpi=600)
    plt.close()

    # 保存训练和验证得分到 CSV 文件
    scores_df = pd.DataFrame({
        "Training Examples": train_sizes,
        "Train Score Mean": train_scores_mean,
        "Train Score Std": train_scores_std,
        "Validation Score Mean": test_scores_mean,
        "Validation Score Std": test_scores_std
    })
    scores_df.to_csv(f"{output_path}/{title.replace(' ', '_')}_learning_scores.csv", index=False)

# 绘制 3D 超参数搜索可视化
def plot_3d_hyperparam_search(cv_results, param_x, param_y, score, title, output_path):
    """
    3D 超参数搜索结果可视化，确保 Z轴标签方向朝外，避免左侧裁剪，并实现 Z轴标签靠右显示。
    """

    # 设置全局字体为 Times New Roman
    from matplotlib import rcParams
    rcParams['font.family'] = 'Times New Roman'
    rcParams['axes.labelsize'] = 26  # 坐标轴标签字体大小
    rcParams['axes.titlesize'] = 26  # 标题字体大小

    # 获取参数值
    param_x_vals = cv_results[f"param_{param_x}"]
    param_y_vals = cv_results[f"param_{param_y}"]
    scores = cv_results[score]

    # 如果参数是非数值类型（如字符串），进行编码
    if not np.issubdtype(type(param_y_vals[0]), np.number):
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        param_y_vals = encoder.fit_transform(param_y_vals)

    # 创建 3D 图
    fig = plt.figure(figsize=(21, 18))
    ax = fig.add_subplot(111, projection='3d')

    # 设置视角，确保 Z 轴垂直
    ax.view_init(elev=20, azim=60)

    # 绘制 3D 散点图
    sc = ax.scatter(
        param_x_vals,
        param_y_vals,
        scores,
        c=scores,
        cmap='viridis',  # 使用适合科研的配色方案（例如 'viridis'）
        marker='o',
        edgecolor='k',  # 添加黑色边框，增强可读性
        s=200  # 设置标记的大小
    )

    # 设置坐标轴标签和标题
    ax.set_xlabel(param_x, fontsize=26, labelpad=20)
    ax.set_ylabel(param_y, fontsize=26, labelpad=20)
    ax.set_title(title, fontsize=26, fontweight='bold', pad=40)

    # 设置刻度字体大小
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)

    # 添加颜色条并调整尺寸
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, location='left', shrink=0.6, aspect=15)  # 调整 shrink 和 aspect
    cbar.set_label('Mean Test Score', fontsize=26, labelpad=15)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.yaxis.set_label_position('right')  # 将 label 放到右侧
    cbar.ax.yaxis.tick_right()  # 将刻度也调整到右侧

    # 手动扩展边界，避免任何裁剪
    fig.subplots_adjust(left=0.3, right=0.85, top=0.9, bottom=0.2)

    # 保存图像
    plt.savefig(f"{output_path}/{title.replace(' ', '_')}_3d_hyperparam_search.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_2d_hyperparam_search(cv_results, param_x, score, title, output_path):
    scores = cv_results[score]
    param_x_vals = cv_results[f"param_{param_x}"]

    plt.figure(figsize=(8, 6))
    plt.plot(param_x_vals, scores, marker='o', linestyle='-', color='b')
    plt.title(title, fontname='Times New Roman', fontsize=22)
    plt.xlabel(param_x, fontname='Times New Roman', fontsize=20)
    plt.ylabel(score, fontname='Times New Roman', fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title.replace(' ', '_')}_2D.png")
    plt.close()

# SHAP 分析
def shap_analysis(model, X_train, X_test, output, model_name):
    """
    对模型进行 SHAP 分析并生成符合科研标准的图表
    """
    # 根据模型类型选择适当的解释器
    explainer = shap.TreeExplainer(model) if isinstance(model, (GradientBoostingClassifier, RandomForestClassifier)) else shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)

    # 设置 Matplotlib 样式，满足科研绘图
    plt.style.use('seaborn-ticks')
    plt.rc('font', family='Times New Roman', size=12)
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # 绘制 SHAP summary 图
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{model_name} SHAP Summary Plot", fontsize=22, fontweight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_shap_summary.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 绘制 SHAP dependence 图（前两个重要特征）
    top_features = np.argsort(np.abs(shap_values).mean(axis=0))[-2:]
    for feature in top_features:
        shap.dependence_plot(
            feature, shap_values, X_test,
            show=False, interaction_index=None
        )
        plt.title(f"{model_name} SHAP Dependence Plot for Feature {feature}", fontsize=22, fontweight='bold')
        plt.xlabel(f"Feature {feature} Value", fontsize=20)
        plt.ylabel("SHAP Value (Impact on Model Output)", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output}/{model_name}_shap_dependence_{feature}.png", dpi=600, bbox_inches='tight')
        plt.close()

    # SHAP bar 图（特征重要性排名）
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"{model_name} SHAP Feature Importance (Bar)", fontsize=22, fontweight='bold')
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_shap_feature_importance.png", dpi=600, bbox_inches='tight')
    plt.close()

def shap_analysis_for_ensemble(model, X_train, X_test, output, model_name):
    """
    对集成模型进行 SHAP 分析，并保存符合科研要求的图表
    """
    # 针对 StackingClassifier 的最终学习器进行分析
    if isinstance(model, StackingClassifier):
        base_model = model.final_estimator_
        print(f"Performing SHAP analysis on StackingClassifier's final estimator: {type(base_model).__name__}")
    else:
        base_model = model
        print(f"Performing SHAP analysis on model: {type(base_model).__name__}")

    # 根据模型类型选择 TreeExplainer 或 KernelExplainer
    if isinstance(base_model, (GradientBoostingClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.KernelExplainer(base_model.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)

    # 设置 Matplotlib 样式
    plt.style.use('seaborn-ticks')
    plt.rc('font', family='Times New Roman', size=12)
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    # 绘制 SHAP Summary Plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{model_name} SHAP Summary Plot", fontsize=22, fontweight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_shap_summary.png", dpi=600, bbox_inches='tight')
    plt.close()

    # 绘制 SHAP Dependence Plot（前两个重要特征）
    top_features = np.argsort(np.abs(shap_values).mean(axis=0))[-2:]
    for feature in top_features:
        shap.dependence_plot(
            feature, shap_values, X_test,
            show=False, interaction_index=None
        )
        plt.title(f"{model_name} SHAP Dependence Plot for Feature {feature}", fontsize=22, fontweight='bold')
        plt.xlabel(f"Feature {feature} Value", fontsize=20)
        plt.ylabel("SHAP Value (Impact on Model Output)", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output}/{model_name}_shap_dependence_{feature}.png", dpi=600, bbox_inches='tight')
        plt.close()

    # 绘制 SHAP Bar Plot（特征重要性排名）
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"{model_name} SHAP Feature Importance (Bar)", fontsize=22, fontweight='bold')
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=20)
    plt.ylabel("Features", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output}/{model_name}_shap_feature_importance.png", dpi=600, bbox_inches='tight')
    plt.close()


def train_and_evaluate(input_file, fingerprint_file, feature, output):
    # 加载数据
    if feature == "Morgan":
        fingerprints = np.load(fingerprint_file, allow_pickle=True)
        data = pd.read_csv(input_file, sep="\t")
        data["Morgan_Fingerprint"] = list(fingerprints)
    
        X = np.array([np.array(fp) for fp in data["Morgan_Fingerprint"]])
        y = data["BBB+/BBB-"].values
    elif feature == "MACCS":
        # 加载数据
        fingerprints = np.load(fingerprint_file, allow_pickle=True)
        data = pd.read_csv(input_file, sep="\t")
        data["MACCS_Fingerprint"] = list(fingerprints)
    
        X = np.array([np.array(fp) for fp in data["MACCS_Fingerprint"]])
        y = data["BBB+/BBB-"].values
    elif feature == "RDKit":
        # 加载数据
        fingerprints = np.load(fingerprint_file, allow_pickle=True)
        data = pd.read_csv(input_file, sep="\t")
        data["RDKit_Fingerprint"] = list(fingerprints)
    
        X = np.array([np.array(fp) for fp in data["RDKit_Fingerprint"]])
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
        "CatBoost": CatBoostClassifier(iterations=500, random_seed=42, verbose=0),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
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
        param_x, param_y = None, None
        # 根据模型选择相应的超参数
        if name == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
            param_x, param_y = 'n_neighbors', 'weights'  # 选择最重要的两个参数
    
        elif name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
            param_x, param_y = 'C', 'penalty'  # 选择最重要的两个参数
    
        elif name == "SVC":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
            param_x, param_y = 'C', 'kernel'  # 选择最重要的两个参数
    
        elif name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'max_depth': [10, 20, 30, 40, 50, 60],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            param_x, param_y = 'n_estimators', 'max_depth'  # 选择最重要的两个参数
    
        elif name == "Gradient Boosting":
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300],
                'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            param_x, param_y = 'n_estimators', 'learning_rate'  # 选择最重要的两个参数
    
        elif name == "MLP":
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'learning_rate_init': [0.001, 0.005, 0.01, 0.05, 0.1],
                'batch_size': [4, 8, 16, 32, 64, 128]
            }
            param_x, param_y = 'learning_rate_init', 'batch_size'  # 选择最重要的两个参数
        elif name == "Decision Tree":
            param_grid = {
                'max_depth': [10, 20, 25, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 15, 20, 25],
                'min_samples_leaf': [1, 2, 4]
            }
            param_x, param_y = 'max_depth', 'min_samples_split'
        elif name == "BernoulliNB":
            param_grid = {
                'alpha': [0.1, 0.5, 1.0]
            }
            param_x = 'alpha'  # Only one parameter, so param_y is None

        elif name == "XGB":
            # 定义参数范围
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500, 600],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
            param_x, param_y = 'n_estimators', 'subsample'

        elif name == "CatBoost":
            param_grid = {
                'iterations': [100, 200, 300, 400, 500, 600],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8, 10, 12, 16],
                'l2_leaf_reg': [1, 3, 5, 7],
                'bagging_temperature': [0.0, 0.5, 1.0]
            }
            param_x, param_y = 'iterations', 'depth'


        random_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, scoring='precision', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        # 绘制 3D 超参数可视化
        # Plot 3D hyperparameter search if both param_x and param_y are defined
        if param_x and param_y:
            plot_3d_hyperparam_search(
                cv_results=random_search.cv_results_,
                param_x=param_x,
                param_y=param_y,
                score="mean_test_score",
                title=f"{name} Hyperparameter Search",
                output_path=output
            )
        elif param_x:  # For models with a single hyperparameter, plot a 2D visualization
            plot_2d_hyperparam_search(
                cv_results=random_search.cv_results_,
                param_x=param_x,
                score="mean_test_score",
                title=f"{name} Hyperparameter Search (2D)",
                output_path=output
            )
        best_model = random_search.best_estimator_
        metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, output, name)
        print(f"{name} metrics: {metrics}")
        #shap_analysis(best_model, X_train, X_test, output, name)
        results[name] = metrics
        joblib.dump(best_model, f"{output}/{name}_model.pkl")

        # 绘制学习曲线
        plot_learning_curve(
            best_model, X_train, y_train, title=f"{name} Learning Curve", output_path=output
        )

    # 集成学习
    print("Training StackingClassifier...")
    estimators = [(name, joblib.load(f"{output}/{name}_model.pkl")) for name in results.keys()]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=CatBoostClassifier(iterations=200, learning_rate=0.05, depth=8, verbose=0),passthrough=True)
    stacking_model.fit(X_train, y_train)
    stacking_metrics = evaluate_model(stacking_model, X_train, y_train, X_test, y_test, output, "StackingClassifier")
    print(f"StackingClassifier metrics: {stacking_metrics}")
    results["StackingClassifier"] = stacking_metrics
    joblib.dump(stacking_model, f"{output}/stacking_model.pkl")
    # SHAP 分析
    #shap_analysis_for_ensemble(stacking_model, X_train, X_test, output, "StackingClassifier")
    # 绘制学习曲线
    plot_learning_curve(
        stacking_model, X_train, y_train, title=f"Stacking Classifier Learning Curve", output_path=output
    )
    weights = [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for _, model in estimators]
    voting_model = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    voting_model.fit(X_train, y_train)
    voting_metrics = evaluate_model(voting_model, X_train, y_train, X_test, y_test, output, "VotingClassifier")
    print(f"VotingClassifier metrics: {voting_metrics}")
    results["VotingClassifier"] = voting_metrics
    joblib.dump(voting_model, f"{output}/voting_model.pkl")
    # SHAP 分析
    #shap_analysis_for_ensemble(voting_model, X_train, X_test, output, "VotingClassifier")
    # 绘制学习曲线
    plot_learning_curve(
        voting_model, X_train, y_train, title=f"Voting Classifier Learning Curve", output_path=output
    )

    # 绘制性能表格和曲线
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(f"{output}/model_performance_metrics.csv")
    plot_performance(results, output)

    return results

#train_and_evaluate("B3DB_classification.tsv","morgan_fingerprints.npy","Morgan","./output/")
# train_and_evaluate("B3DB_classification.tsv","maccs_fingerprints.npy","MACCS","./output_maccs/")
train_and_evaluate("B3DB_classification.tsv","rdkit_fingerprints.npy","RDKit","./output_rdkit_2/")
