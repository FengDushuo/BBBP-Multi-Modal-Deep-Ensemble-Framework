import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from torchvision import transforms
from PIL import Image
import os

# 路径设置
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output_opt'
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data_maccs_opt_lso.pkl'
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

# 处理 logBB 异常值，使用 IQR 方法去除 logBB 的异常值
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

original_length = len(data)
data = remove_outliers(data, 'logBB')
log_message(f"Removed {original_length - len(data)} outliers from logBB column")

# 重新生成分子图像（确保每个 SMILES 对应一张图片）
def generate_molecule_image(smiles, molecule_no):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            image_path = os.path.join(image_dir, f"{molecule_no}.png")
            Draw.MolToFile(mol, image_path)
            return True
        else:
            log_message(f"Invalid SMILES for image generation: {smiles}")
            return False
    except Exception as e:
        log_message(f"Error generating image for SMILES {smiles}: {e}")
        return False

data['Image_Generated'] = data.apply(
    lambda row: generate_molecule_image(row['SMILES'], row['NO.']), axis=1
)
generated_images = data[data['Image_Generated']]
log_message(f"Generated images count: {len(generated_images)}")

# 确保数据一致性
invalid_indices = set(invalid_smiles.index) | set(missing_images.index)
data = data[~data.index.isin(invalid_indices)]
log_message(f"Final dataset length after ensuring consistency: {len(data)}")

# 标准化特征
fingerprint_scaler = StandardScaler()
data['MACCS_Normalized'] = data['MACCS'].apply(
    lambda x: fingerprint_scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
)

if use_missing_images:
    image_scaler = StandardScaler()
    data['Image_Features_Normalized'] = data['Image_Features'].apply(
        lambda x: image_scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten() if x is not None else None
    )
else:
    image_scaler = StandardScaler()
    data['Image_Features_Normalized'] = data['Image_Features'].apply(
        lambda x: image_scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
    )

# 使用孤立森林检测异常值并去除异常值
features = np.hstack([data['MACCS_Normalized'].values, data['Image_Features_Normalized'].values])
model = IsolationForest(contamination=0.05, random_state=42)

# 使用 Isolation Forest 训练模型并进行异常值预测
data['outliers'] = model.fit_predict(features)

# 1 表示正常点，-1 表示异常点
# 去除异常值
cleaned_data = data[data['outliers'] == 1]

# 打印移除异常值后的数据长度
log_message(f"Removed rows identified as outliers using Isolation Forest. New dataset size: {len(cleaned_data)}")

# 保存处理后的数据
cleaned_data.to_pickle(output_path)
log_message(f"Processed data saved to {output_path}")
