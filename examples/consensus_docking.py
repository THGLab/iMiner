from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size

binding_site = box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35))
ligands = ["O=C(Nc1ccc2c(c1)N(C(=O)C1CCCC1)CC2)c1ccc2ccccc2n1",
"O=C(Nc1ccc2c(c1)N(C(=O)C1CCCO1)CC2)C1CC1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)C(C)Oc1ccc(C)cc1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)C([NH3+])(CC)CC",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)N1CCN(c2ncc(C(F)(F)F)cc2Cl)CC1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)NC1CCCc2ccc(F)cc21"]


project = ConsensusDocking("test_consensus_docking", docking_protocols=["vina"])
project.add_protein("./helicase/helicase-holo.pdb", "helicase_holo_structure", binding_site=binding_site)
project.add_multiple_ligands(ligands)
project.run_consensus_docking(n_jobs=3)
