import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc
import torch

# 检查 GPU 是否可用
if not torch.cuda.is_available():
    print("警告：未检测到 GPU，程序将以 CPU 模式运行，可能会降低效率。")

# 数据文件路径（原始文件路径）
bbb_fpath = "D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_classification.tsv"
# 加载数据
df = pd.read_csv(bbb_fpath, sep="\t")

# 定义函数生成 GPU 加速特征
def generate_gpu_features(smiles):
    """
    使用 DeepChem 和 GPU 提取分子描述符
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 使用 DeepChem 的 GPU 加速特征生成器
        featurizer = dc.feat.ConvMolFeaturizer()
        features = featurizer.featurize([mol])
        if len(features) > 0 and features[0] is not None:
            return features[0].get_atom_features().tolist()  # 返回分子特征
        else:
            return None
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None

# 遍历 DataFrame 并生成所有 GPU 特征
results = []
for idx, row in df.iterrows():
    smiles = row["SMILES"]
    gpu_features = generate_gpu_features(smiles)
    results.append({
        "Compound Name": row["compound_name"],
        "SMILES": smiles,
        "GPU_Features": gpu_features,
    })

# 转换为 DataFrame
gpu_feature_df = pd.DataFrame(results)

# 保存 GPU 特征为 NumPy 数组文件
np.save("gpu_features.npy", gpu_feature_df["GPU_Features"].to_list())
print("GPU 特征数据已保存为 NumPy 文件")

# 保存为 CSV 文件
# gpu_feature_df.to_csv("gpu_fingerprint_results.csv", index=False)
# print("GPU 特征数据已保存为 CSV 文件")

# 查看结果
print(gpu_feature_df)
