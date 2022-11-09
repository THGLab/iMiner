from iMiner.docking.autodock4 import AD4Docking
from iMiner.utils import box_from_center_and_size

docking = AD4Docking(protein_pdb="./helicase/5rob.pdb", ligand_fd="./helicase/ligands",
protein_ad4_fd="./helicase/ad4-folder",
docking_box=box_from_center_and_size(center=(-18,14.5,-33), size=(24,14,20)))

print(docking.ad4result_analysis())