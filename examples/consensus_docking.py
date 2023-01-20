from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size

binding_site = box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35))
ligands = ["COC=CC1=C[C@H]2C(=O)N[CH][C@@H]3CC[C@@H](C)C[C@@H]3C[C@@H]1C(=O)N(c1cccc(O)c1)C2=O",
"C[C@@H]1/C=C\c2nonc2C2=CN1C=C[C@H]2Nc1cccc2cnccc12"]


project = ConsensusDocking("test_consensus_docking", docking_protocols=["vina-gpu"])
project.add_protein("./helicase/helicase-holo.pdb", "helicase_holo_structure", binding_site=binding_site)
project.add_multiple_ligands(ligands)
project.run_consensus_docking(n_jobs=4)
