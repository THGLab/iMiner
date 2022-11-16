from iMiner.docking.vina import VinaDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
from glob import glob

ligands = glob("./helicase/ligands/*.sdf")[:3]


#box size in unit of Angstrom; 
#assumes vina version >=1.2.0 (0.375 Ang spacing by default); otherwise rescale box size

docking = VinaDocking(protein_pdb="./helicase/helicase-holo.pdb", 
docking_box=box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35)), temp_path=Path("/tmp"))

print(docking.dock(ligands=ligands, output_dir="./docking"))
