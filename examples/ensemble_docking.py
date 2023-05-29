from iMiner.core.ensemble_docking import EnsembleDocking


confs = ["7M8P","7M91","7RMB"]
ligands = ["COC=CC1=C[C@H]2C(=O)N[CH][C@@H]3CC[C@@H](C)C[C@@H]3C[C@@H]1C(=O)N(c1cccc(O)c1)C2=O",
"C[C@@H]1/C=C\c2nonc2C2=CN1C=C[C@H]2Nc1cccc2cnccc12"]
protein_confs = ["/home/jerry/data/iMiner/covid_project/mpro_exp_strucs/non-covalent/representative_crystal_proteins/{}.pdb".format(conf) for conf in confs]


project = EnsembleDocking("test_ensemble_docking", docking_protocol="vina")
project.add_protein_confs("mpro", protein_confs, protein_conf_names=confs, 
                          binding_site_dataframe="/home/jerry/data/iMiner/covid_project/mpro_exp_strucs/non-covalent/representative_crystal_proteins/centers.csv")

project.add_multiple_ligands(ligands)
project.create_docking_objects("mpro")
project.run_ensemble_docking(n_jobs=8)
