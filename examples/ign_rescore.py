from iMiner.docking import IGNscoring, RFscoring
from pathlib import Path
from glob import glob
import pandas as pd

df = pd.read_csv("../../MPro/interaction/filtered.csv")

#ign = IGNscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
#results = ign.rescore(["../../MPro/interaction/docking/vina-gpu/%s.sdf"%n for n in ligands.names], n_jobs=16)
#results.to_csv("../../MPro/interaction/ign.csv", index=False)

rf = RFscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
results = rf.rescore(["../../MPro/interaction/docking/vina-gpu/%s.sdf"%n for n in df.names], n_jobs=16)
results.to_csv("../../MPro/interaction/rf.csv", index=False)
