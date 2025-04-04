import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 22

# 1️⃣ 读取数据
bbb_fpath = "D:/a_work/1-phD/project/5-VirtualScreening/B3DB/B3DB/B3DB_classification.tsv"
df = pd.read_csv(bbb_fpath, sep="\t")

# 2️⃣ 生成指纹
def generate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    morgan_fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))  # Morgan 指纹
    maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))  # MACCS 指纹（167 bits）
    rdkit_fp = list(Chem.RDKFingerprint(mol))  # RDKit 拓扑指纹

    return morgan_fp, maccs_fp, rdkit_fp

# 3️⃣ 处理数据
morgan_data, maccs_data, rdkit_data = [], [], []
labels = []  # BBB+ (1) or BBB- (0)

for idx, row in df.iterrows():
    smiles = row["SMILES"]
    bbb_label = row["BBB+/BBB-"]  # 读取分类标签

    # 转换为数值 (1: BBB+, 0: BBB-)
    if bbb_label == "BBB+":
        label = 1
    elif bbb_label == "BBB-":
        label = 0
    else:
        continue  # 跳过无效数据

    morgan_fp, maccs_fp, rdkit_fp = generate_fingerprints(smiles)

    if None in (morgan_fp, maccs_fp, rdkit_fp):
        continue  # 跳过错误数据

    # 分别存储不同指纹
    morgan_data.append(morgan_fp)
    maccs_data.append(maccs_fp)
    rdkit_data.append(rdkit_fp)
    labels.append(label)

# 转换为 NumPy 数组
morgan_matrix = np.array(morgan_data)
maccs_matrix = np.array(maccs_data)
rdkit_matrix = np.array(rdkit_data)
labels = np.array(labels)

# 4️⃣ PCA 降维 & 归一化
def perform_pca_and_plot(matrix, labels, title, filename):
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)  # 归一化

    pca = PCA(n_components=2)  # 降到 2D
    pca_result = pca.fit_transform(matrix_scaled)

    # 计算解释的方差比例
    explained_variance = pca.explained_variance_ratio_ * 100
    pc1_explained = round(explained_variance[0], 2)
    pc2_explained = round(explained_variance[1], 2)

    # 5️⃣ 绘制并保存 PCA 结果
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], c='blue', label="BBB+")
    plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], c='red', label="BBB-")
    
    plt.xlabel(f"PC1 ({pc1_explained}%)", fontsize=22, fontname="Times New Roman", fontweight="bold")  # 显示 PC1 解释的方差
    plt.ylabel(f"PC2 ({pc2_explained}%)", fontsize=22, fontname="Times New Roman", fontweight="bold")  # 显示 PC2 解释的方差
    plt.title(title, fontsize=22, fontname="Times New Roman", fontweight="bold")
    plt.legend(fontsize=22, loc="upper left", prop={'family': 'Times New Roman'})
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig(filename, dpi=600)  # 保存图片
    plt.show()

# 分别进行 PCA 并保存图像
perform_pca_and_plot(morgan_matrix, labels, "PCA of Morgan Fingerprint (BBB+ vs BBB-)", "pca_morgan_classification.png")
perform_pca_and_plot(maccs_matrix, labels, "PCA of MACCS Fingerprint (BBB+ vs BBB-)", "pca_maccs_classification.png")
perform_pca_and_plot(rdkit_matrix, labels, "PCA of RDKit Fingerprint (BBB+ vs BBB-)", "pca_rdkit_classification.png")

print("PCA 结果已保存！")
