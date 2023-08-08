from iMiner.docking import IGNscoring
from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv("../../MPro/WJ_2D_derive/redock/vina/results.csv")
ligands = df[np.isnan(df["ign_score"])].vina_path.values

ign = IGNscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
results = ign.rescore(ligands, n_jobs=16)
results.to_csv("../../MPro/WJ_2D_derive/redock/ign.csv", index=False)