from iMiner.docking.autodock4 import AD4Docking
from iMiner.utils import box_from_center_and_size
from pathlib import Path

docking = AD4Docking(protein_pdb="./helicase/helicase-holo.pdb", ligand_fd="./helicase/ligands",
docking_box=box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35)))

print(docking.ad4result_analysis())