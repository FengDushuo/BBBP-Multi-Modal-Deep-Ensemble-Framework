import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator

# 定义 SMILES 结构
smiles = "CN1C(=NN=N1)SCC2=C(N3C(C(C3=O)(NC(=O)C(C4=CC=C(C=C4)O)C(=O)O)OC)OC2)C(=O)O"

# 转换为 RDKit 分子对象
mol = Chem.MolFromSmiles(smiles)

### 计算 Morgan 指纹
bitInfo = {}  # 存储贡献原子的位置信息
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bitInfo)

# 获取 Morgan 指纹贡献的原子（蓝色）
highlight_atoms_morgan = {}
for bit, atoms in bitInfo.items():
    for atom, radius in atoms:
        highlight_atoms_morgan[atom] = (0.0, 0.0, 1.0)  # 蓝色

### 计算 MACCS 指纹
maccs_fp = MACCSkeys.GenMACCSKeys(mol)

# MACCS 使用 SMARTS 匹配结构
maccs_smarts = ['[OH]', '[C=O]', 'c1ccccc1']  # 羟基、羰基、苯环
highlight_atoms_maccs = {}
for smarts in maccs_smarts:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            for atom in match:
                highlight_atoms_maccs[atom] = (0.0, 1.0, 0.0)  # 绿色

### 计算 RDKit 指纹
rdkit_fp = GetRDKitFPGenerator().GetFingerprint(mol)

# RDKit 使用 SMARTS 识别环状结构
rdkit_smarts = ['[r5]', '[r6]']  # 五元环、六元环
highlight_atoms_rdkit = {}
for smarts in rdkit_smarts:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            for atom in match:
                highlight_atoms_rdkit[atom] = (1.0, 0.0, 0.0)  # 红色

### 绘制 Morgan 指纹高亮
drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms_morgan.keys()), highlightAtomColors=highlight_atoms_morgan)
drawer.FinishDrawing()
with open("morgan_fingerprint.png", "wb") as f:
    f.write(drawer.GetDrawingText())

### 绘制 MACCS 指纹高亮
drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms_maccs.keys()), highlightAtomColors=highlight_atoms_maccs)
drawer.FinishDrawing()
with open("maccs_fingerprint.png", "wb") as f:
    f.write(drawer.GetDrawingText())

### 绘制 RDKit 指纹高亮
drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms_rdkit.keys()), highlightAtomColors=highlight_atoms_rdkit)
drawer.FinishDrawing()
with open("rdkit_fingerprint.png", "wb") as f:
    f.write(drawer.GetDrawingText())

# 显示所有图片
from IPython.display import display
display(drawer.GetDrawingText())
