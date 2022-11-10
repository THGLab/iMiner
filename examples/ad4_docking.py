from iMiner.docking.autodock4 import AD4Docking
from iMiner.utils import box_from_center_and_size
from pathlib import Path
import os

ligands = [str(Path("./protein-example/ligands") / file) \
    for file in os.listdir("./helicase/ligands") if file.endswith("sdf")]
docking = AD4Docking(protein_pdb="./helicase/5rob.pdb", protein_ad4_fd="./helicase/ad4-folder",
docking_box=box_from_center_and_size(center=(-18,14.5,-33), size=(24,14,20)))

print(docking.dock(ligands, "./helicase/ad4-out-poses"))