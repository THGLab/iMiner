from iMiner.docking import IGNscoring, RFscoring
from pathlib import Path
from glob import glob
import pandas as pd

ign = IGNscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
lig_path = [f"/global/scratch/users/ozhang/covid/MPro/WJ_2D/version_{i}/docking/vina-gpu/{n}.sdf" for n in df.names]
results = ign.rescore(lig_path, n_jobs=16)
results.to_csv(f"../../MPro/WJ_2D/version_{i}/docking/ign.csv", index=False)

#rf = RFscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
#results = rf.rescore(["../../MPro/frag_test/docking/vina-gpu/%s.sdf"%n for n in df.names], n_jobs=16)
#results.to_csv("../../MPro/frag_test/rf.csv", index=False)
