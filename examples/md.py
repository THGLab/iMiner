import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print(sys.path[0])
from iMiner.core.md import MDProject


if __name__ == "__main__":
    proj = MDProject(project_name="helicase")
    proj.add_ligand("./helicase/UNC-0379-holo.sdf", name="UNC-0379-holo")
    proj.add_protein("./helicase/helicase-holo.pdb", name='helicase-holo')
    proj.read_params_json("./helicase/param.json")
    proj.run(lig_name="UNC-0379-holo", prot_name="helicase-holo", task_name="unc_holo")
