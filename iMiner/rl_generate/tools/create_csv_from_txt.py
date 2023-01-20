import numpy as np
import selfies as sf
from tqdm import tqdm

def train_test_split(indices, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_indices = indices[:split]
    test_indices = indices[split:]
    return train_indices, test_indices

data_folder = "../data"
format = "selfies"

chembl_txt_file = f"{data_folder}/chembl.txt"
with open(chembl_txt_file, "r") as f:
    all_mols = [item.strip() for item in f.readlines()]

if format == "selfies":
    converted_mols = []
    for mol in tqdm(all_mols):
        try:
            converted_mols.append(sf.encoder(mol))
        except sf.EncoderError:
            print("Cannot convert", mol)
    all_mols = converted_mols

indices = np.arange(len(all_mols))
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

with open(f"{data_folder}/chembl_train.csv", "w") as f:
    f.write(format + ",length\n")
    for index in train_indices:
        mol = all_mols[index]
        mol_tokens = mol.split("][")
        f.write(f"{mol},{len(mol_tokens)}\n")

with open(f"{data_folder}/chembl_val.csv", "w") as f:
    f.write(format + ",length\n")
    for index in test_indices:
        f.write(f"{all_mols[index]},{len(all_mols[index])}\n")

print("Dataset process complete!")