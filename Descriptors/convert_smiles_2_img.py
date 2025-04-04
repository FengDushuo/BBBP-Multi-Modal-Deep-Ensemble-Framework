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

