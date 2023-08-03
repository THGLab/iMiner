#from iMiner.docking.vina import VinaGPUDocking
from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import pandas as pd

#box size in unit of Angstrom; 
#vina version >=1.2.0 (0.375 Ang grid spacing by default)

binding_site = box_from_center_and_size(center=(54.5, 46, 75), size=(24, 24, 24))
df = pd.read_csv("vina_redock.csv")
ligands = df["smiles"].values

project = ConsensusDocking("./test", docking_protocols=["vina", "rfscore", "ign"])
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdbqt", "mpro", binding_site=binding_site)
project.add_protein("/global/scratch/users/ozhang/covid/MPro/WJ_mpro.pdb", "mpro_", binding_site=binding_site)
project.add_multiple_ligands(ligands, names=df["name"].values)
project.run_consensus_docking(protein_name="mpro", n_jobs=8)
