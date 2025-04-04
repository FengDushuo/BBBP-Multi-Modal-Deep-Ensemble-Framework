# 加载新的 SMILES 数据
new_data = pd.read_csv("new_molecule_library.csv")  # 包含 SMILES 列

# 生成指纹
new_morgan_fps = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=2048)) for smiles in new_data["SMILES"]]
X_new = np.array(new_morgan_fps)

# 标准化特征并降维
X_new_scaled = scaler.transform(X_new)
X_new_reduced = pca.transform(X_new_scaled)

# 预测分子活性
new_predictions = rf_model.predict(X_new_reduced)
new_probabilities = rf_model.predict_proba(X_new_reduced)[:, 1]

# 保存筛选结果
new_data["Prediction"] = new_predictions
new_data["Probability"] = new_probabilities
new_data.to_csv("virtual_screening_results.csv", index=False)
