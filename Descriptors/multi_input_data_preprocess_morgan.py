import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

# Load the TSV file
file_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
data = pd.read_csv(file_path, sep='\t')

# Check if required columns exist
if 'SMILES' not in data.columns or 'NO.' not in data.columns:
    raise ValueError("The file must contain 'SMILES' and 'NO.' columns.")

# Output directory for images
output_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output'
os.makedirs(output_dir, exist_ok=True)

# Process each row and generate images
for index, row in data.iterrows():
    smiles = row['SMILES']
    molecule_name = row['NO.']
    
    # Convert SMILES to molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        # Generate image and save
        output_path = os.path.join(output_dir, f"{molecule_name}.png")
        Draw.MolToFile(molecule, output_path)
    else:
        print(f"Failed to process SMILES: {smiles}")

# 加载数据
data_path = 'D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_regression.tsv'
image_dir = 'D:/a_work/1-phD/project/5-VirtualScreening/rdkit-descriptor/img_output'

data = pd.read_csv(data_path, sep='\t')

# 检查数据是否包含必要列
required_columns = ['SMILES', 'logBB', 'NO.']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The file must contain the following columns: {required_columns}")

# 提取 Morgan 分子指纹
def compute_morgan(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))  # Morgan指纹 (ECFP)
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return [0] * 167  # 返回全零指纹

data['Morgan'] = data['SMILES'].apply(compute_morgan)

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
invalid_smiles = data[data['Morgan'].apply(lambda x: x == [0] * 167)]
print(f"无法解析的 SMILES 数量: {len(invalid_smiles)}")

missing_images = data[data['Image_Features'].apply(lambda x: np.all(x == 0))]
print(f"缺失图像的数量: {len(missing_images)}")

# 标准化特征
fingerprint_scaler = StandardScaler()
image_scaler = StandardScaler()


data['Morgan_Normalized'] = data['Morgan'].apply(
    lambda x: fingerprint_scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
)
data['Image_Features_Normalized'] = data['Image_Features'].apply(
    lambda x: image_scaler.fit_transform(x.reshape(-1, 1)).flatten()
)

# 保存处理后的数据
output_path = 'D:/a_work/1-phD/project/5-VirtualScreening/processed_data_morgan.pkl'
data.to_pickle(output_path)
print(f"数据已保存到 {output_path}")
