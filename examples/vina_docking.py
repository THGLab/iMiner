#from iMiner.docking.vina import VinaGPUDocking
from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path

#ligands = ["/global/scratch/users/ozhang/covid/rdkit_vina/outputs/cmpd9+.sdf"]

#box size in unit of Angstrom; 
#vina version >=1.2.0 (0.375 Ang grid spacing by default)

#docking = VinaGPUDocking(protein_pdb="/global/scratch/users/ozhang/covid/rdkit_vina/ns3_protease.pdbqt", 
#docking_box=box_from_center_and_size(center=(-13, 15, -13), size=(20, 25, 25)), temp_path=Path("./docking"))
#print(docking.dock(ligands=ligands, output_dir="./docking"))


binding_site = box_from_center_and_size(center=(-13, 15, -13), size=(20, 25, 25))
ligands = ["COC=CC1=C[C@H]2C(=O)N[CH][C@@H]3CC[C@@H](C)C[C@@H]3C[C@@H]1C(=O)N(c1cccc(O)c1)C2=O",
"C[C@@H]1/C=C\c2nonc2C2=CN1C=C[C@H]2Nc1cccc2cnccc12"]


project = ConsensusDocking("test_consensus_docking", docking_protocols=["vina-gpu"])
project.add_protein("/global/scratch/users/ozhang/covid/rdkit_vina/ns3_protease.pdbqt", "zikvpro", binding_site=binding_site)
project.add_multiple_ligands(ligands)
project.run_consensus_docking(n_jobs=2)