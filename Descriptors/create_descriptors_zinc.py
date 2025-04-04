import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np

# 定义分子指纹生成函数
def generate_all_fingerprints(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None
        
        # 生成指纹
        morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))  # Morgan指纹
        # maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))  # MACCS指纹
        # rdkit_fp = list(Chem.RDKFingerprint(mol))  # RDKit拓扑指纹

        # Avalon 指纹（如果可用）
        # try:
        #     from rdkit.Avalon.pyAvalonTools import GetAvalonFP
        #     avalon_fp = list(GetAvalonFP(mol, nBits=2048))
        # except ImportError:
        #     avalon_fp = None

        # return morgan_fp, maccs_fp, rdkit_fp, avalon_fp
        return morgan_fp
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        # return None, None, None, None
        return None

# 定义函数处理整个目录
def process_directory(directory_path, output_path):
    all_results = []

    # 遍历目录中的每个 .smi 文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".smi"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            # 加载 .smi 文件
            feaa_data = pd.read_csv(file_path, sep=" ", names=["SMILES", "ZINC_ID"], skiprows=1)
            
            # 对每个分子生成指纹
            for idx, row in feaa_data.iterrows():
                smiles = row["SMILES"]
                compound_name = row["ZINC_ID"]
                # morgan_fp, maccs_fp, rdkit_fp, avalon_fp = generate_all_fingerprints(smiles)
                morgan_fp = generate_all_fingerprints(smiles)
                all_results.append({
                    "ZINC_ID": compound_name,
                    "SMILES": smiles,
                    "Morgan_Fingerprint": morgan_fp,
                    # "MACCS_Fingerprint": maccs_fp,
                    # "RDKit_Fingerprint": rdkit_fp,
                    # "Avalon_Fingerprint": avalon_fp,
                })

    # 转换为 DataFrame
    fingerprint_df = pd.DataFrame(all_results)
    
    # 保存指纹为 NumPy 文件
    np.save(os.path.join(output_path, "morgan_fingerprints.npy"), fingerprint_df["Morgan_Fingerprint"].to_list())
    # np.save(os.path.join(output_path, "maccs_fingerprints.npy"), fingerprint_df["MACCS_Fingerprint"].to_list())
    # np.save(os.path.join(output_path, "rdkit_fingerprints.npy"), fingerprint_df["RDKit_Fingerprint"].to_list())

    # 保存到 CSV 文件
    fingerprint_df.to_csv(os.path.join(output_path, "fingerprint_results.csv"), index=False)
    print("所有分子指纹已生成并保存！")

# 设置目录路径和输出路径
input_directory = "FE"  # 替换为实际目录路径
output_directory = "output"  # 替换为实际输出路径
os.makedirs(output_directory, exist_ok=True)

# 处理目录
process_directory(input_directory, output_directory)
