#from iMiner.docking.vina import VinaGPUDocking
from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import pandas as pd

#box size in unit of Angstrom; 
#vina version >=1.2.0 (0.375 Ang grid spacing by default)

binding_site = box_from_center_and_size(center=(56, 72, 23), size=(18, 18, 22))
df = pd.read_csv("vina_redock.csv")
ligands = df["smiles"].values

project = ConsensusDocking("./test", docking_protocols=["vina", "rfscore"])
project.add_protein("/global/scratch/users/ozhang/covid/rdkit_vina/ns3_ns2brm.pdbqt", "zikvopen", binding_site=binding_site)
project.add_multiple_ligands(ligands, names=df["name"].values)
project.run_consensus_docking(protein_name="zikvopen", n_jobs=8)
