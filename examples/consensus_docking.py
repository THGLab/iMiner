#from iMiner.docking.vina import VinaGPUDocking
from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import numpy as np
import pandas as pd
import os

binding_site = box_from_center_and_size(center=(54.5, 46, 75), size=(24, 24, 24))
df = pd.read_csv("../../MPro/frag_test/merge.csv").loc[[8,12,71,99,147,259,260,270,271,299,314,320,321,405,503,550]]

project = ConsensusDocking("WJmpro", project_path="../../MPro/frag_test/redocking", docking_protocols=["vina", "ign"])
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdbqt", "mpro", binding_site=binding_site)
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", "mpro_", binding_site=binding_site)
project.add_multiple_ligands(df.smiles, names=df.names)
project.run_consensus_docking(protein_name="mpro", n_jobs=8, exhaustiveness=64, single_job_timeout=300)
