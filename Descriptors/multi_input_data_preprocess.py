import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# 加载数据
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output'

data = pd.read_csv(data_path, sep='\t')

# 检查数据是否包含必要列
required_columns = ['SMILES', 'logBB', 'NO.']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The file must contain the following columns: {required_columns}")

# 提取 MACCS 分子指纹
def compute_maccs(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return list(MACCSkeys.GenMACCSKeys(mol))
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return [0] * 167  # 返回全零指纹

data['MACCS'] = data['SMILES'].apply(compute_maccs)

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 加载图像并提取特征
def load_image_features(molecule_no):
    image_path = os.path.join(image_dir, f"{molecule_no}.png")
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return image_transform(img).numpy().flatten()
        except Exception as e:
            print(f"Error loading image for {molecule_no}: {e}")
            return np.zeros((128 * 128 * 3,))
    else:
        print(f"Image not found for molecule NO.: {molecule_no}")
        return np.zeros((128 * 128 * 3,))

data['Image_Features'] = data['NO.'].apply(load_image_features)

# 检查数据完整性
invalid_smiles = data[data['MACCS'].apply(lambda x: x == [0] * 167)]
print(f"无法解析的 SMILES 数量: {len(invalid_smiles)}")

missing_images = data[data['Image_Features'].apply(lambda x: np.all(x == 0))]
print(f"缺失图像的数量: {len(missing_images)}")

# 标准化特征
fingerprint_scaler = StandardScaler()
image_scaler = StandardScaler()

data['MACCS_Normalized'] = data['MACCS'].apply(
    lambda x: fingerprint_scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
)
data['Image_Features_Normalized'] = data['Image_Features'].apply(
    lambda x: image_scaler.fit_transform(x.reshape(-1, 1)).flatten()
)

# 保存处理后的数据
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data.pkl'
data.to_pickle(output_path)
print(f"数据已保存到 {output_path}")
