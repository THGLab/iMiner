from iMiner.docking.vina import VinaDocking
#from rdkit import Chem
import os
import pandas as pd
import numpy as np
from glob import glob

docking = VinaDocking(protein_pdb="/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdbqt", 
    temp_path="/tmp", docking_box=[42.5, 34, 63, 66.5, 58, 87],
    #num_modes=5, exhaustiveness=64)
    weight_gauss1 = 0.003372, weight_gauss2 = -0.008098, weight_repulsion = 0.014212, weight_hydrophobic = -0.008361,
    weight_hydrogen = -0.227928, weight_rot = 0.0584600
    )


#df = pd.read_csv("../../MPro/WJ_2D/version_2/filtered.csv")
#df = df[df.docking < -7]
liglist = glob(f"/global/scratch/users/ozhang/covid/MPro/frag_test/redocking/vina/*.sdf")
result = docking.rescore_parallel(liglist, n_jobs=1)
result.to_csv(f"../../MPro/frag_test/redocking/vina_rescore.csv", index=False)
