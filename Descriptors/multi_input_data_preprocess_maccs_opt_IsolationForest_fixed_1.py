import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torchvision import transforms
from PIL import Image
import os
import joblib

# 路径设置
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output_opt'
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data_maccs_opt_lso_fixed_1.pkl'
log_file = 'data_cleaning_log_maccs.txt'

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
def compute_MACCS(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)  # 使用较小的数据类型
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        log_message(f"Error processing SMILES: {smiles}, Error: {e}")
        return np.zeros(167, dtype=np.uint8)  # MACCS 指纹长度为 167

data['MACCS'] = data['SMILES'].apply(compute_MACCS)
invalid_smiles = data[data['MACCS'].apply(lambda x: np.all(x == 0))]
log_message(f"Invalid SMILES count: {len(invalid_smiles)}")
data = data[data['MACCS'].apply(lambda x: not np.all(x == 0))]

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
def standardize_features(data, maccs_col='MACCS', image_col='Image_Features', batch_size=100):
    scaler = StandardScaler()
    all_features_normalized = []

    for i in range(0, len(data), batch_size):
        batch_maccs = np.stack(data[maccs_col].iloc[i:i + batch_size].values)
        batch_images = np.stack(data[image_col].iloc[i:i + batch_size].values)
        batch_features = np.hstack([batch_maccs, batch_images])
        batch_normalized = scaler.fit_transform(batch_features)
        all_features_normalized.extend(batch_normalized)

    all_features_normalized = np.array(all_features_normalized, dtype=np.float32)
    num_maccs = batch_maccs.shape[1]
    data['MACCS_Normalized'] = list(all_features_normalized[:, :num_maccs])
    data['Image_Features_Normalized'] = list(all_features_normalized[:, num_maccs:])
    return data

data = standardize_features(data)

# PCA 降维
def apply_pca(data, column, n_components=30):
    features = np.vstack(data[column].values)
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    data[f'{column}_PCA'] = list(reduced_features)
    return data

data = apply_pca(data, 'MACCS_Normalized', n_components=30)
data = apply_pca(data, 'Image_Features_Normalized', n_components=30)

# 特征交互
def create_interaction_features(data, column1, column2):
    features1 = np.vstack(data[column1].values)
    features2 = np.vstack(data[column2].values)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(np.hstack([features1, features2]))
    data['Interaction_Features'] = list(interaction_features)
    return data

data = create_interaction_features(data, 'MACCS_Normalized_PCA', 'Image_Features_Normalized_PCA')

# 使用孤立森林检测异常值
def detect_outliers_with_isolation_forest(data, feature_columns):
    features = np.hstack([np.vstack(data[col].values) for col in feature_columns])
    model = IsolationForest(contamination=0.05, random_state=42)
    data['Outliers'] = model.fit_predict(features)
    return data

data = detect_outliers_with_isolation_forest(data, ['MACCS_Normalized_PCA', 'Image_Features_Normalized_PCA'])

# 删除 logBB 小于 -2.0 的数据点
logBB_threshold = -2.0
data = data[data['logBB'] >= logBB_threshold]

# 保存处理后的数据
data.to_pickle(output_path)
log_message(f"Final processed data saved to {output_path}")
