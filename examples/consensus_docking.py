#from iMiner.docking.vina import VinaGPUDocking
from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import numpy as np
import pandas as pd

#box size in unit of Angstrom; 
#vina version >=1.2.0 (0.375 Ang grid spacing by default)

binding_site = box_from_center_and_size(center=(54.5, 46, 75), size=(24, 24, 24))
df = pd.read_csv("../../MPro/interaction/results.csv")
f1 = df.docking < -7
f2 = df.drug_likeliness > 2.8
f3 = df.interaction > 0
ligands = df[f1 & f2 & f3]

project = ConsensusDocking("WJmpro", project_path="../../MPro/interaction/docking", docking_protocols=["vina", "ign"])
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdbqt", "mpro", binding_site=binding_site)
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", "mpro_", binding_site=binding_site)
project.add_multiple_ligands(ligands.smiles, names=ligands.ligand_names.values.astype(str))
project.run_consensus_docking(protein_name="mpro", n_jobs=8) #, exhaustiveness=64, single_job_timeout=600)
