import os
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
from scipy.spatial import distance_matrix

from iMiner.docking.base import BaseDocking
from iMiner.pathlib import RF_model_path, RF_col_mask

NELEMTS = 54   # maximum number of chemical elements considered

atomic_number_to_name = {
    6 : ["C" , "CA" , "CB" , "CD" , "CD1" , "CD2" , "CE" , "CE1" , 
         "CE2", "CE3", "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3"],
    8 : ["O" , "OD1" , "OD2" , "OE1" , "OE1A" , "OE1B" , "OE2" , "OG" , "OG1", "OH", "OXT"],
    7 : ["N" , "NE" , "NE1" , "NE2" , "NE2A" , "NE2B" , "ND1" , "ND2" , "NH1" , "NH2" , "NZ"],
    9 : ["F"],
    15 : ["P"],
    16 : ["S" , "SD" , "SG"],
    17 : ["Cl", "CL"],
    35 : ["Br", "BR"],
    53 : ["I"],
}

name_to_atomic_number = { }
for k, v in atomic_number_to_name.items():
    for i in v:
        name_to_atomic_number[i] = k
        

def read_pdb_file(filename):
    coords = []
    atomnumbers = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                indices = [0, 6, 12, 17, 20, 22, 26, 30, 38, 46, 54, 60, 66, 78]
                items = [line.strip()[i:j] for i, j in zip(indices, indices[1:]+[None])]
                if items[-2].strip() == "H":
                    continue
                if name_to_atomic_number.get(items[2].strip()) is None:
                    continue
                atomnumbers.append(name_to_atomic_number.get(items[2].strip()))
                coords += items[7:10]

    assert len(coords)//3 == len(atomnumbers)
    return np.array(atomnumbers), np.reshape(coords, (-1, 3)).astype(float)

def read_ligand_sdf(filename):
    coords = []
    atomnumbers = []
    with open(filename, "r") as f:
        lines = f.readlines()
        natoms = int(lines[3][:3].strip())
        for i in range(natoms):
            line = lines[i+4].split()
            atomnumber = name_to_atomic_number.get(line[3])
            if atomnumber is None:
                continue
            atomnumbers.append(atomnumber)
            coords += line[:3]
        assert len(coords)//3 == len(atomnumbers)
        return np.array(atomnumbers), np.reshape(coords, (-1, 3)).astype(float)  
        
def RF_descriptor(ligands, pocket, dcutoff=12, verbose=0):
    lig_descriptors = []
    
    if not os.path.exists(pocket):
        if verbose: print(pocket, "does not exists")
        return
    pocket_a, pocket_c = read_pdb_file(pocket)
    
    for ligand_file in ligands: 
        if not os.path.exists(ligand_file):
            if verbose: print(ligand_file, "does not exists")
            continue

        ligand_a, ligand_c = read_ligand_sdf(ligand_file)
        features = np.zeros((NELEMTS, NELEMTS))

        # Calculate distances between current ligand and its binding site
        d = distance_matrix(pocket_c, ligand_c)
        dmask = d < dcutoff
        lgrid, pgrid = np.meshgrid(ligand_a, pocket_a)
        assert pgrid.shape == dmask.shape
        p_hits = pgrid[dmask]
        l_hits = lgrid[dmask]
        for u in zip(p_hits, l_hits):
            features[int(u[0]), int(u[1])] += 1
            
        col_mask = list(atomic_number_to_name.keys())
        lig_descriptors.append(features[col_mask][:, col_mask])

    return np.array(lig_descriptors)
 
class RFscoring(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path=None, logger=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        super().__init__(protein_pdb, docking_box, temp_path, **kwargs)   
    
    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["original_names", "smiles", "score"]
        '''
        
         # prepare lists to record results
        ligand_smiles = [self.convert_sdf_to_smiles(l) for l in ligands]
        
        # generate RFscore descriptors
        descriptors = RF_descriptor(ligands, self.protein_path)
        Xs = np.reshape(descriptors, (-1, 81))[:, RF_col_mask]
        
        with open(RF_model_path, 'rb') as f:
            RFmodel = pickle.load(f)
        RF_pred = RFmodel.predict(Xs) * -1.36 #convert pkd to kcal/mol
        
        # generate the final pandas dataframe and return
        df = pd.DataFrame({"original_names": ligands, "smiles": ligand_smiles, "score": RF_pred})
        return df
