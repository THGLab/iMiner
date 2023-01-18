'''
Author: Jie Li
Date created: Aug 22, 2022

Doing local Autodock Vina docking to obtain the scores
'''

import numpy as np
import pandas as pd
import multiprocessing

from iMiner.core.consensus_docking import ConsensusDocking
from iMiner.rl_generate.utils import get_gpu_count

VINA_DOCKING_PY_DIRPATH = '/home/jerry/data/covid_project/rdkit_vina'
import sys
import os
from pathlib import Path
# sys.path.append(VINA_DOCKING_PY_DIRPATH)
# from docking import docksmile
# from utils import get_free_gpu, 

# def return_func_delegate(func, *args, send_end=None, **kwargs):
#     results = func(*args, **kwargs)
#     send_end.send(results)

# def docking_with_timeout(*args, timeout=45):
    
#     if args[-1]: # use_gpu
#         gpu = get_free_gpu()
#     else:
#         gpu = None
#     receive_end, send_end = multiprocessing.Pipe(duplex=False)
#     p = multiprocessing.Process(target=return_func_delegate,
#          args=(docksmile,) + args[:-1], kwargs=dict(gpu=gpu, send_end=send_end, return_docked_file=True))
#     p.daemon = True
#     p.start()
#     send_end.close() # child must be the only one with it opened
#     p.join(timeout)
    
#     if p.is_alive():
#         p.terminate()
#         print("docking for SMILES {} timed out".format(args[0]))
#         return args[0], np.nan, None
#     else:
#         return receive_end.recv()  # get value from the child


class vina_score_assigner():
    def __init__(self, protein_file, box, path, use_gpu=False, timeout=45) -> None:
        '''
        use_gpu: whether or not use the gpu version of vina for docking
        timeout: the timeout for each docking job
        '''
        protein_name = os.path.basename(protein_file).split('.')[0]
        self.protein_name = protein_name
        self.docking_project = ConsensusDocking(protein_name, path, ["vina" if not use_gpu else "vina-gpu"])
        self.docking_project.add_protein(protein_file_path=protein_file, name=protein_name, binding_site=box)
        self.output_dir = self.docking_project.project_path / Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        if use_gpu:
            self.n_jobs = get_gpu_count() - 1
        else:
            self.n_jobs = int(multiprocessing.cpu_count() * 0.9)
        self.timeout = timeout


    def get_scores(self, smiles, new_names):
        added_new_names = self.docking_project.add_multiple_ligands(smiles, names=new_names)
        self.docking_project.run_consensus_docking(ligand_names=added_new_names, n_jobs=self.n_jobs, 
            output_csv=self.output_dir / Path(f"{self.iter}.csv"), single_job_timeout=self.timeout)
        result_df = pd.read_csv(self.output_dir / Path(f"{self.iter}.csv"))
        result_df.index = result_df.ligand_names
        result_dict = result_df["score"].to_dict()
        results = [result_dict.get(name, np.nan) for name in new_names]
        return results

if __name__ == '__main__':
    import time
    assigner = vina_score_assigner(use_gpu=True)
    start_time = time.time()
    returns = assigner.get_scores(["O=C(Nc1ccc2c(c1)N(C(=O)C1CCCC1)CC2)c1ccc2ccccc2n1",
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
"O=C(Nc1ccc2c(c1)NC(=O)CO2)NC1CCCc2ccc(F)cc21"])
    print(returns)
    print('Time elapsed: %.2f seconds' % (time.time() - start_time))
