import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC                            #引入支持向量机分类器
from sklearn.neighbors import KNeighborsClassifier     #引入KNN算法模型
from sklearn.linear_model import LogisticRegression    #引入LogisticRegression逻辑回归模型
from sklearn.tree import DecisionTreeClassifier        #引入决策树模型
from sklearn.naive_bayes import BernoulliNB            #引入伯努利贝叶斯模型
from sklearn.ensemble import RandomForestClassifier    #引入随机森林模型
from sklearn.ensemble import GradientBoostingClassifier#引入梯度提升分类树模型
from sklearn.neural_network import MLPClassifier       #引入神经网络多层感知机模型
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

#绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator, title, output, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 使用的分类器。
    title : 标题
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.savefig(output+title+'.jpg')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    评估模型性能并计算多个指标
    """
    # 训练集预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    # 准确率
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    # 平衡准确率
    test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
    # 精确率、召回率、F1分数
    test_precision = precision_score(y_test, test_pred, average='binary', zero_division=0)
    test_recall = recall_score(y_test, test_pred, average='binary', zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average='binary', zero_division=0)
    # Matthews 相关系数, Cohen’s κ
    test_mcc = matthews_corrcoef(y_test, test_pred)
    test_cohen_kappa = cohen_kappa_score(y_test, test_pred)

    # ROC AUC
    if hasattr(model, "predict_proba"):
        test_prob = model.predict_proba(X_test)
        # 检查 predict_proba 的返回值形状
        if test_prob.shape[1] == 2:
            test_auc = roc_auc_score(y_test, test_prob[:, 1])
        else:
            # 如果只返回一列，则使用预测结果计算 AUC
            test_auc = roc_auc_score(y_test, test_pred)
    else:
        test_auc = roc_auc_score(y_test, test_pred)

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

def plot_metrics(methods_list, metric_name, output):
    # 提前设置字体为 Times New Roman
    font_properties = {'family': 'Times New Roman', 'size': 16}
    algorithms = [m['Algorithm Name'] for m in methods_list]
    values = [m[metric_name] for m in methods_list]
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(algorithms, values, color='skyblue')
    # 设置 x 轴和 y 轴标签及标题
    plt.xlabel('Algorithms', fontdict=font_properties)
    plt.ylabel(metric_name, fontdict=font_properties)
    plt.title(f'Comparison of {metric_name} across models', fontdict=font_properties)
    # 设置 x 轴刻度字体
    plt.xticks(rotation=45, fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    # 网格线
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output+metric_name+'-performance.jpg')


def morgan_train_model(fps_file,input_file,output):
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

    methods_list=[]
    models_json={}

    # #### KNN算法模型的训练与评估
    print('KNN:')
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn_init = KNeighborsClassifier()                         #K近邻模型训练中，n邻居个数越多越欠拟合，越少越过拟合
    grid_knn = GridSearchCV(knn_init, param_grid_knn, cv=5, scoring='f1', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    knn = grid_knn.best_estimator_                                        #训练KNN模型
    plot_learning_curve(knn, 'KNN_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    knn_train_score = train_score
    knn_test_score = test_score
    models_json["KNN"]=knn
    knn_metrics = evaluate_model(knn, X_train, y_train, X_test, y_test)
    print(f"KNN metrics: {knn_metrics}")
    methods_list.append({
        'Algorithm Name': "KNN",
        'Train Score': knn_train_score,
        'Test Score': knn_test_score,
        **knn_metrics
    })
    joblib.dump(knn, output+'knn_model.pkl')

    # #### 逻辑回归模型的训练与评估
    print('Logic Regression:')
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'saga']
    }
    #弱正则化对应过拟合，强正则化对应欠拟合
    lr = LogisticRegression(max_iter=1000)
    grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    logistic_regression = grid_lr.best_estimator_                
    #在逻辑回归中，参数C控制正则化强弱，C越大正则化越弱，C越小正则化越强
    #训练逻辑回归模型
    plot_learning_curve(logistic_regression, 'logistic_regression_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = logistic_regression.score(X_train, y_train)
    test_score = logistic_regression.score(X_test, y_test)
    logistic_train_score = train_score
    logistic_test_score = test_score
    models_json["LR"]=logistic_regression
    logistic_metrics = evaluate_model(logistic_regression, X_train, y_train, X_test, y_test)
    print(f"LR metrics: {logistic_metrics}")
    methods_list.append({
        'Algorithm Name': "LR",
        'Train Score': logistic_train_score,
        'Test Score': logistic_test_score,
        **logistic_metrics
    })
    joblib.dump(logistic_regression, output+'logistic_regression_model.pkl')

    # #### 支持向量机分类器的训练与评估
    print('SVC:')
    param_grid_svc = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    svc_init = SVC()
    grid_svc = GridSearchCV(svc_init, param_grid_svc, cv=5, scoring='f1', n_jobs=-1)
    grid_svc.fit(X_train, y_train)
    svc = grid_svc.best_estimator_
    
    plot_learning_curve(svc, 'svc_Learning_Curve', output, X_train, y_train, ylim=None, cv=6,train_sizes=np.linspace(.1, 1.0, 5))
    train_score = svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    svc_train_score = train_score
    svc_test_score = test_score
    models_json["SVC"]=svc
    svc_metrics = evaluate_model(svc, X_train, y_train, X_test, y_test)
    print(f"SVC metrics: {svc_metrics}")
    methods_list.append({
        'Algorithm Name': "SVC",
        'Train Score': svc_train_score,
        'Test Score': svc_test_score,
        **svc_metrics
    })
    joblib.dump(svc, output+'svc_model.pkl')

    # #### 伯努利贝叶斯模型的训练与评估
    print('BernoulliNB:') 
    param_grid_nb = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'binarize': [0.0, 0.5, 1.0]
    } 
    nb = BernoulliNB()
    grid_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='f1', n_jobs=-1)
    grid_nb.fit(X_train, y_train)
    bernoulli_bayes = grid_nb.best_estimator_                  
    #训练伯努利贝叶斯模型
    plot_learning_curve(bernoulli_bayes, 'bernoulli_bayes_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = bernoulli_bayes.score(X_train, y_train)
    test_score = bernoulli_bayes.score(X_test, y_test)
    bernoulli_train_score = train_score
    bernoulli_test_score = test_score
    models_json["BB"]=bernoulli_bayes
    bernoulli_metrics = evaluate_model(bernoulli_bayes, X_train, y_train, X_test, y_test)
    print(f"BernoulliNB metrics: {bernoulli_metrics}")
    methods_list.append({
        'Algorithm Name': "BB",
        'Train Score': bernoulli_train_score,
        'Test Score': bernoulli_test_score,
        **bernoulli_metrics
    })
    joblib.dump(bernoulli_bayes, output+'bernoulli_bayes_model.pkl')

    # #### 决策树模型的训练与评估
    print('DecisionTree:')
    param_grid_dt = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    dt = DecisionTreeClassifier()
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
    grid_dt.fit(X_train, y_train)
    decision_tree = grid_dt.best_estimator_   
    #设置决策树的最大深度，避免出现过拟合现象           
    plot_learning_curve(decision_tree, 'decision_tree_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = decision_tree.score(X_train, y_train)
    test_score = decision_tree.score(X_test, y_test)
    DT_train_score = train_score
    DT_test_score = test_score
    models_json["DT"]=decision_tree
    DT_metrics = evaluate_model(decision_tree, X_train, y_train, X_test, y_test)
    print(f"DT metrics: {DT_metrics}")
    methods_list.append({
        'Algorithm Name': "DT",
        'Train Score': DT_train_score,
        'Test Score': DT_test_score,
        **DT_metrics
    })
    joblib.dump(decision_tree, output+'decision_tree_model.pkl')

    # #### 随机森林模型的训练与评估
    print('RandomForest:')
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf = RandomForestClassifier()
    random_rf = RandomizedSearchCV(rf, param_grid_rf, n_iter=20, cv=5, scoring='f1', n_jobs=-1, random_state=42)
    random_rf.fit(X_train, y_train)
    random_forest = random_rf.best_estimator_
    #训练随机森林模型
    plot_learning_curve(random_forest, 'random_forest_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = random_forest.score(X_train, y_train)
    test_score = random_forest.score(X_test, y_test)
    RF_train_score = train_score
    RF_test_score = test_score
    models_json["RF"]=random_forest
    RF_metrics = evaluate_model(random_forest, X_train, y_train, X_test, y_test)
    print(f"RF metrics: {RF_metrics}")
    methods_list.append({
        'Algorithm Name': "RF",
        'Train Score': RF_train_score,
        'Test Score': RF_test_score,
        **RF_metrics
    })
    joblib.dump(random_forest, output+'random_forest_model.pkl')

    # #### 梯度提升分类器模型的训练与评估
    print('GradientBoosting:')  
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0]
    } 
    gb = GradientBoostingClassifier()
    grid_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='f1', n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    gradient_boosting_tree = grid_gb.best_estimator_
    #训练梯度提升分类树模型
    plot_learning_curve(gradient_boosting_tree, 'gradient_boosting_tree_Learning_Curve', output, X_train, y_train, ylim=None, cv=6,train_sizes=np.linspace(.1, 1.0, 5))
    train_score = gradient_boosting_tree.score(X_train, y_train)
    test_score = gradient_boosting_tree.score(X_test, y_test)
    GBT_train_score = train_score
    GBT_test_score = test_score
    models_json["GBT"]=gradient_boosting_tree
    GBT_metrics = evaluate_model(gradient_boosting_tree, X_train, y_train, X_test, y_test)
    print(f"GBT metrics: {GBT_metrics}")
    methods_list.append({
        'Algorithm Name': "GBT",
        'Train Score': GBT_train_score,
        'Test Score': GBT_test_score,
        **GBT_metrics
    })
    joblib.dump(gradient_boosting_tree, output+'gradient_boosting_tree_model.pkl')


    # #### MLP神经网络多层感知机模型的训练与评估
    print('MLP:')  
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01]
    }  
    mlp_init = MLPClassifier(max_iter=1000)
    grid_mlp = GridSearchCV(mlp_init, param_grid_mlp, cv=5, scoring='f1', n_jobs=-1)
    grid_mlp.fit(X_train, y_train)
    mlp = grid_mlp.best_estimator_
    #训练神经网络MLP多层感知机模型
    plot_learning_curve(mlp, 'mlp_Learning_Curve', output, X_train, y_train, ylim=None, cv=6, train_sizes=np.linspace(.1, 1.0, 5))
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    MLP_train_score = train_score
    MLP_test_score = test_score
    models_json["MLP"]=mlp
    MLP_metrics = evaluate_model(mlp, X_train, y_train, X_test, y_test)
    print(f"MLP metrics: {MLP_metrics}")
    methods_list.append({
        'Algorithm Name': "MLP",
        'Train Score': MLP_train_score,
        'Test Score': MLP_test_score,
        **MLP_metrics
    })      
    joblib.dump(mlp, output+'mlp_model.pkl')

    labels = [item['Algorithm Name'] for item in methods_list]
    train_score_l = [item['Train Score'] for item in methods_list]
    test_score_l = [item['Test Score'] for item in methods_list]

    x = np.arange(len(labels))  # 标签位置
    width = 0.4  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig=plt.figure(dpi=600)
    ax=fig.add_subplot(111)
    max_y_lim = max(train_score_l+test_score_l) + 0.1
    min_y_lim = min(train_score_l+test_score_l) - 0.1
    plt.ylim(min_y_lim, max_y_lim)
    rects3 = ax.bar(x + 0.01, train_score_l, width, label='Train Score')
    rects4 = ax.bar(x + width+ 0.02, test_score_l, width, label='Test Score')
    # 为y轴、标题和x轴等添加一些文本。 
    ax.set_xlabel('Algorithms',fontproperties = 'Times New Roman', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.title("Algorithms Performance" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size' : 12})
    for rects in [rects3,rects4]:
        """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig(output+'algorithm_performance.jpg')

    data_file = open(output+'algorithm_data_file.csv', 'w')
    csv_writer = csv.writer(data_file)
    count = 0
    for emp in methods_list:
        if count == 0:
            header = emp.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(emp.values())
    data_file.close()

    max_fitted_algorithm_model = mlp
    max_fitted_algorithm_name = "MLP"
    max_fitted_algorithm_score = 0
    for item in methods_list:
        if item["Accuracy"]+item["ROC AUC"]+item["Balanced Accuracy"]>max_fitted_algorithm_score:
            max_fitted_algorithm_score = item["Accuracy"]+item["ROC AUC"]+item["Balanced Accuracy"]
            max_fitted_algorithm_name = item["Algorithm Name"]
            max_fitted_algorithm_model = models_json[max_fitted_algorithm_name]

    plot_metrics(methods_list, 'Accuracy', output)
    plot_metrics(methods_list, 'Balanced Accuracy', output)
    plot_metrics(methods_list, 'Precision', output)
    plot_metrics(methods_list, 'Recall', output)
    plot_metrics(methods_list, 'F1 Score', output)
    plot_metrics(methods_list, 'MCC', output)
    plot_metrics(methods_list, "Cohen's Kappa", output)
    plot_metrics(methods_list, 'ROC AUC', output)
    print("The Best Algorithm Name"+max_fitted_algorithm_name)

morgan_train_model("D:/a_work/1-phD/project/5-VirtualScreening/morgan_fingerprints.npy","D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_classification.tsv","D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/output/")