from iMiner.docking import IGNscoring, RFscoring
from pathlib import Path
from glob import glob
import pandas as pd

df = pd.read_csv("../../MPro/frag_test/version_1/results.csv")

ign = IGNscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
results = ign.rescore(["../../MPro/frag_test/version_1/docking/vina-gpu/%s.sdf"%n for n in df.names], n_jobs=16)
results.to_csv("../../MPro/frag_test/version_1/docking/ign.csv", index=False)

#rf = RFscoring("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", None, temp_path="/tmp")
#results = rf.rescore(["../../MPro/frag_test/docking/vina-gpu/%s.sdf"%n for n in df.names], n_jobs=16)
#results.to_csv("../../MPro/frag_test/rf.csv", index=False)
