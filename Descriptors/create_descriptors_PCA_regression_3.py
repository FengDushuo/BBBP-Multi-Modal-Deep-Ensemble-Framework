import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import joblib

# 路径设置
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output_opt'
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data_rdkit_opt_lso_fixed_1.pkl'
log_file = 'data_cleaning_log_rdkit.txt'

# 创建输出目录
os.makedirs(image_dir, exist_ok=True)

# 加载数据
data = pd.read_csv(data_path, sep='\t')

# 检查必要列
required_columns = ['SMILES', 'logBB', 'NO.']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The file must contain the following columns: {required_columns}")

# 日志记录函数
def log_message(message):
    with open(log_file, 'a') as log:
        log.write(message + '\n')
    print(message)

# 过滤无效 SMILES
def compute_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(RDKFingerprint(mol), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        log_message(f"Error processing SMILES: {smiles}, Error: {e}")
        return np.zeros(2048, dtype=np.uint8)

data['rdkit'] = data['SMILES'].apply(compute_rdkit)
invalid_smiles = data[data['rdkit'].apply(lambda x: np.all(x == 0))]
log_message(f"Invalid SMILES count: {len(invalid_smiles)}")
data = data[data['rdkit'].apply(lambda x: not np.all(x == 0))]

# 图像预处理
def load_image_features(molecule_no):
    image_path = os.path.join(image_dir, f"{molecule_no}.png")
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            return transform(img).flatten().numpy().astype(np.float32)
        except Exception as e:
            log_message(f"Error loading image for {molecule_no}: {e}")
            return None
    else:
        log_message(f"Image not found for molecule NO.: {molecule_no}")
        return None

data['Image_Features'] = data['NO.'].apply(load_image_features)
missing_images = data[data['Image_Features'].isnull()]
log_message(f"Missing images count: {len(missing_images)}")

# 提供选项：继续使用标记数据或直接丢弃这些样本
use_missing_images = False
if not use_missing_images:
    data = data.dropna(subset=['Image_Features'])
    log_message("Dropped rows with missing images.")
else:
    log_message("Proceeding with missing images marked as None.")

# 特征标准化（按块处理）
def standardize_features(data, rdkit_col='rdkit', image_col='Image_Features', batch_size=100):
    scaler = StandardScaler()
    rdkit_normalized = []
    image_normalized = []

    for i in range(0, len(data), batch_size):
        batch_rdkit = np.vstack(data[rdkit_col].iloc[i:i + batch_size].values)
        batch_images = np.vstack(data[image_col].iloc[i:i + batch_size].values)
        batch_features = np.hstack([batch_rdkit, batch_images])
        batch_normalized = scaler.fit_transform(batch_features)
        num_rdkit = batch_rdkit.shape[1]
        rdkit_normalized.extend(batch_normalized[:, :num_rdkit])
        image_normalized.extend(batch_normalized[:, num_rdkit:])

    data['rdkit_Normalized'] = list(map(np.array, rdkit_normalized))
    data['Image_Features_Normalized'] = list(map(np.array, image_normalized))
    return data

data = standardize_features(data)

# PCA 降维
def apply_pca(data, column, n_components=30, batch_size=100):
    features = np.vstack(data[column].values)
    pca = PCA(n_components=n_components)
    reduced_features = []

    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        reduced_batch = pca.fit_transform(batch)
        reduced_features.extend(reduced_batch)

    data[f'{column}_PCA'] = list(np.array(reduced_features, dtype=np.float32))
    return data

data = apply_pca(data, 'rdkit_Normalized', n_components=30)
data = apply_pca(data, 'Image_Features_Normalized', n_components=30)

# 特征交互
def create_interaction_features(data, column1, column2):
    features1 = np.vstack(data[column1].values)
    features2 = np.vstack(data[column2].values)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(np.hstack([features1, features2]))
    data['Interaction_Features'] = list(interaction_features)
    return data

data = create_interaction_features(data, 'rdkit_Normalized_PCA', 'Image_Features_Normalized_PCA')

def apply_pca_and_plot(data, column, title, n_components=2, save_path=None):
    features = np.vstack(data[column].values)
    
    # 进行 PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    
    # 计算解释的方差比例
    explained_variance = pca.explained_variance_ratio_ * 100
    pc1_explained = f"{explained_variance[0]:.2f}"
    pc2_explained = f"{explained_variance[1]:.2f}"
    
    # 存储降维后的数据
    data[f'{column}_PCA'] = list(reduced_features)
    
    # 📌 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5, c='blue', edgecolors='k')
    
    plt.xlabel(f"PC1 ({pc1_explained}%)", fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.ylabel(f"PC2 ({pc2_explained}%)", fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.title(f"PCA Visualization of {title}", fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()
    
    return data

data = apply_pca_and_plot(data, 'rdkit_Normalized', 'Normalized Fingerprint', n_components=2, save_path="pca_regression_fingerprint_rdkit.png")

# 🔹 对图像特征进行 PCA 并可视化
data = apply_pca_and_plot(data, 'Image_Features_Normalized','Normalized Image Features', n_components=2, save_path="pca_regression_images_rdkit.png")
             
# 🔹 对交互特征进行 PCA 并可视化
data = apply_pca_and_plot(data, 'Interaction_Features', 'Interaction Features', n_components=2, save_path="pca_regression_interaction_rdkit.png")

