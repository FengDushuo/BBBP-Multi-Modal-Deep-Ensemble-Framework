import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import rdMolDescriptors

# data file name for BBB dataset with categorical data
bbb_fpath = "D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_classification.tsv"
# load data
df = pd.read_csv(bbb_fpath, sep="\t")

# 定义函数生成所有指纹
def generate_all_fingerprints(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None, None, None
        
        # 生成指纹
        morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))  # Morgan指纹 (ECFP)
        maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))  # MACCS指纹
        rdkit_fp = list(Chem.RDKFingerprint(mol))  # RDKit拓扑指纹
        # atom_pair_fp = list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048))  # Atom Pair指纹
        # torsion_fp = list(rdMolDescriptors.GetTopologicalTorsionFingerprintAsIntVect(mol))  # Torsion指纹
        
        # Avalon 指纹（如果 Avalon 可用）
        try:
            from rdkit.Avalon.pyAvalonTools import GetAvalonFP
            avalon_fp = list(GetAvalonFP(mol, nBits=2048))
        except ImportError:
            avalon_fp = None  # 如果 Avalon 未安装
        
        return morgan_fp, maccs_fp, rdkit_fp, avalon_fp
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None, None, None, None, None, None

# 遍历 DataFrame 并生成所有指纹
results = []
for idx, row in df.iterrows():
    smiles = row["SMILES"]
    morgan_fp, maccs_fp, rdkit_fp, avalon_fp = generate_all_fingerprints(smiles)
    results.append({
        "Compound Name": row["compound_name"],
        "SMILES": smiles,
        "Morgan_Fingerprint": morgan_fp,
        "MACCS_Fingerprint": maccs_fp,
        "RDKit_Fingerprint": rdkit_fp,
        "Avalon_Fingerprint": avalon_fp,
    })

# 转换为 DataFrame
fingerprint_df = pd.DataFrame(results)
# 将每种指纹保存为 NumPy 数组文件
np.save("morgan_fingerprints.npy", fingerprint_df["Morgan_Fingerprint"].to_list())
np.save("maccs_fingerprints.npy", fingerprint_df["MACCS_Fingerprint"].to_list())
print("指纹数据已保存为 NumPy 文件")
np.save("rdkit_fingerprints.npy", fingerprint_df["RDKit_Fingerprint"].to_list())
print("指纹数据已保存为 NumPy 文件")

# 查看结果
print(fingerprint_df)