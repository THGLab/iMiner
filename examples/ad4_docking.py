from iMiner.docking.autodock4 import AD4Docking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import os

ligands = glob("./helicase/ligands/*.sdf")[:3]
docking = AD4Docking(protein_pdb="./helicase/5rob.pdb",
docking_box=box_from_center_and_size(center=(-18,14.5,-33), size=(24,14,20)), temp_path=Path("/tmp"))

print(docking.dock(ligands, output_dir = "./docking"))