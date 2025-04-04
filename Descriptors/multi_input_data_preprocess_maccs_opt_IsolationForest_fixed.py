import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from torchvision import transforms
from PIL import Image
import os

# 路径设置
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output_opt'
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data_maccs_opt_lso_fixed_pca30_15.pkl'
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
            return list(MACCSkeys.GenMACCSKeys(mol))  # MACCS指纹
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        log_message(f"Error processing SMILES: {smiles}, Error: {e}")
        return [0] * 2048

data['MACCS'] = data['SMILES'].apply(compute_MACCS)
invalid_smiles = data[data['MACCS'].apply(lambda x: x == [0] * 2048)]
log_message(f"Invalid SMILES count: {len(invalid_smiles)}")
data = data[data['MACCS'].apply(lambda x: x != [0] * 2048)]

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
            return transform(img).numpy().flatten()
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

# 特征标准化
def standardize_features(data, maccs_col='MACCS', image_col='Image_Features'):
    maccs_features = np.vstack(data[maccs_col].values)
    image_features = np.vstack(data[image_col].values)
    all_features = np.hstack([maccs_features, image_features])
    scaler = StandardScaler()
    all_features_normalized = scaler.fit_transform(all_features)
    num_maccs = maccs_features.shape[1]
    data['MACCS_Normalized'] = list(all_features_normalized[:, :num_maccs])
    data['Image_Features_Normalized'] = list(all_features_normalized[:, num_maccs:])
    return data

data = standardize_features(data)

# PCA 降维
def apply_pca(data, column, n_components=10):
    features = np.vstack(data[column].values)
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    data[f'{column}_PCA'] = list(reduced_features)
    return data

data = apply_pca(data, 'MACCS_Normalized', n_components=30)
data = apply_pca(data, 'Image_Features_Normalized', n_components=30)

# 特征交互 (使用降维后的特征)
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
    data['outliers'] = model.fit_predict(features)
    return data

data = detect_outliers_with_isolation_forest(data, ['MACCS_Normalized_PCA', 'Image_Features_Normalized_PCA'])

# 删除 logBB 小于 -2.0 的数据点
logBB_threshold = -1.5
data = data[data['logBB'] >= logBB_threshold]

# 保存处理后的数据
data.to_pickle(output_path)
log_message(f"Final processed data saved to {output_path}")
