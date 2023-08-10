from iMiner.docking import IGNscoring, RFscoring
from pathlib import Path
from glob import glob

ligands = glob("/global/scratch/users/ozhang/covid/ZikvPro/active_select/*.sdf")
ign = IGNscoring("/global/scratch/users/ozhang/covid/ZikvPro/ns3_protease.pdb", None, temp_path="/tmp")
results = ign.rescore(ligands, n_jobs=16)
results.to_csv("../../ZikvPro/active_select/ign.csv", index=False)
