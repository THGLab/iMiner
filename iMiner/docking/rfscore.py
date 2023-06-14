import os
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
from scipy.spatial import distance_matrix

from iMiner.docking.base import BaseDocking
from iMiner.pathlib import RF_model_path, RF_col_mask
from iMiner.utils import read_pdb_file, read_ligand_sdf, atomic_number_to_name

NELEMTS = 54   # maximum number of chemical elements considered
        
def RF_descriptor(ligand, pocket, dcutoff=12):
    pocket_a, pocket_c = pocket
    ligand_a, ligand_c = read_ligand_sdf(ligand)
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
    return features[col_mask][:, col_mask]
 
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

        :return: pd.DataFrame with columns ["smiles", "score"]
        '''
        if not os.path.exists(self.protein_path):
            print(self.protein_path, "does not exist")
            return
        pocket_info = read_pdb_file(self.protein_path)
        
        ligand_names = []
        ligand_smiles = []
        ligand_paths = []
        descriptors = []
        #score_mask = np.zeros(len(ligands), dtype=bool)
    
        for ligand_file in ligands: 
            if not os.path.exists(ligand_file):
                continue
            ligand_names.append(ligand_file.split("/")[-1].split(".")[0])
            ligand_smiles.append(self.convert_sdf_to_smiles(ligand_file))
            ligand_paths.append(ligand_file)
            # generate RFscore descriptors
            descriptors.append(RF_descriptor(ligand_file, pocket_info))
            
        Xs = np.reshape(descriptors, (-1, 81))[:, RF_col_mask]
        
        with open(RF_model_path, 'rb') as f:
            RFmodel = pickle.load(f)
        RF_pred = RFmodel.predict(Xs) * -1.36 #convert pkd to kcal/mol
        #RF_pred[score_mask] = np.nan
        
        # generate the final pandas dataframe and return
        df = pd.DataFrame({"ligand_names": ligand_names, "smiles": ligand_smiles, "rfscore_score": RF_pred,
                           "rfscore_path": ligand_paths})
        return df
