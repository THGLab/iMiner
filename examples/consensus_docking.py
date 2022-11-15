from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.utils import box_from_center_and_size

binding_site = box_from_center_and_size(center=(-23, 18, -33), size=(30, 25,35))
ligands = ["O=C(Nc1ccc2c(c1)N(C(=O)C1CCCC1)CC2)c1ccc2ccccc2n1",
"O=C(Nc1ccc2c(c1)N(C(=O)C1CCCO1)CC2)C1CC1",
"O=C(Nc1ccc2c(c1)N(C(=O)c1ccco1)CC2)c1cccnc1OC",
"O=C(Nc1ccc2c(c1)N(C(=O)c1cccs1)CC2)c1ccc(C)cc1",
"O=C(Nc1ccc2c(c1)N(C(=O)c1cccs1)CC2)c1ccco1",
"O=C(Nc1ccc2c(c1)N(C(=O)c1cccs1)CC2)c1cnccn1",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)C1(c2ccccc2)CC1",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)c1cc2cccc(OC)c2o1",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)c1cc2cc(OC)ccc2o1",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)c1cc(-c2cccs2)on1",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)c1cc(C)nc2onc(C)c12",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)C1=NC(C)C(C(=O)OC(C)C)=C1C",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)c1oc2cc(C)cc(C)c2c1C",
"O=C(Nc1ccc2c(c1)NC(=O)C(C)O2)NC1CCCc2ccccc21",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)C1(c2ccccc2)CCC1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)c1c(O)cc(F)cc1F",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)C(C)Oc1ccc(C)cc1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)C([NH3+])(CC)CC",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)N1CCN(c2ncc(C(F)(F)F)cc2Cl)CC1",
"O=C(Nc1ccc2c(c1)NC(=O)CO2)NC1CCCc2ccc(F)cc21"]


project = ConsensusDocking("test_consensus_docking", docking_protocols=["vina"])
project.add_protein("./helicase/helicase-holo.pdb", "helicase_holo_structure", binding_site=binding_site))
project.add_multiple_ligands(ligands)
project.run_consensus_docking()
