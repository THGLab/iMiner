from iMiner.docking.vina import VinaDocking
from iMiner.utils import box_from_center_and_size
from pathlib import Path

docking = VinaDocking(protein_pdb="./helicase/helicase-holo.pdbqt", 
docking_box=box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35)), temp_path=Path("./docking"))

print(docking.dock(ligands=["./helicase/ligands/UNC-0379-holo.sdf"], output_dir="./docking"))
