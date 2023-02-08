'''
Author: Jie Li
Date created: Aug 22, 2022

Doing local Autodock Vina docking to obtain the scores
'''

import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

from iMiner.core.consensus_docking import ConsensusDocking, docking_protocol_map
from iMiner.rl_generate.utils import get_gpu_count
from iMiner.rl_generate.evaluators.fragment_similarity import calc_fragment_position

from rdkit import Chem
#from rdkit.Chem import Draw
import os
from pathlib import Path

def unpack_helper(func, args):
    '''
    Helper function to unpack arguments for multiprocessing
    '''
    return func(*args)

def calc_average_df(df, column_name, key="smiles"):
    """
    calculate average of data column sharing the same key;
    update new column values
    """
    sorted_df = df.sort_values(key, ignore_index=False)
    data = sorted_df[column_name].values
    dkey = sorted_df[key].values
    new_scores = []
    st = 0
    i = 1
    while i < df.shape[0]:
        if i == df.shape[0] - 1:
            score = np.mean(data[st:])
            new_scores += [score]*(i + 1 - st)
        elif dkey[i] != dkey[st]:
            score = np.mean(data[st : i])
            new_scores += [score]*(i - st)
            st = i
        i += 1
    sorted_df["adjusted_"+column_name] = new_scores
    return sorted_df.sort_index()
        
    
class vina_score_assigner():
    def __init__(self, protein_file, box, path, use_gpu=False, timeout=45, fragment=None) -> None:
        '''
        use_gpu: whether or not use the gpu version of vina for docking
        timeout: the timeout for each docking job
        fragment: dock with fragmebt restraints
        '''
        protein_name = os.path.basename(protein_file).split('.')[0]
        self.protein_name = protein_name
        self.protocol_name = "vina" if not use_gpu else "vina-gpu"
        self.docking_project = ConsensusDocking(protein_name, path, [self.protocol_name])
        self.docking_project.add_protein(protein_file_path=protein_file, name=protein_name, binding_site=box)
        self.output_dir = self.docking_project.project_path / Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.frag_restrain = fragment
        if use_gpu:
            self.n_jobs = get_gpu_count()
        else:
            self.n_jobs = int(multiprocessing.cpu_count() * 0.9)
        self.timeout = timeout


    def get_scores(self, smiles, new_names, iteration=0):
        added_new_names = self.docking_project.add_multiple_ligands(smiles, names=new_names)
        self.docking_project.run_consensus_docking(ligand_names=added_new_names, n_jobs=self.n_jobs, 
            output_csv=self.output_dir / Path(f"{iteration}.csv"), single_job_timeout=self.timeout)
        result_df = self.get_result_df(iteration)
        
        if self.frag_restrain is not None:
            # untested
            new_score = self.dock_restrain_parallel(result_df["path"].values, 
                result_df["score"].values)
            result_df["score"] =  new_score
                
        # average vina scores for the same smiles 
        avg_result = calc_average_df(result_df, "score")
        self.update_result_df(avg_result, iteration)
        avg_result.index = avg_result.ligand_names
        result_dict = avg_result["adjusted_score"].to_dict()
                        
        # save the best 50 molecules this iteration in an image
        """
        selected = result_df.sort_values("score", ascending=True).head(50)
        img = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in selected["smiles"]], 
            legends=list(selected["ligand_names"]+",vina:"+selected["score"].astype(str)),
            molsPerRow=5, subImgSize=(200,200))
        img.save(f'{self.output_dir}/{iteration}.png')
        """
        return [result_dict.get(name, np.nan) for name in new_names]
        
    
    def fragment_position_restrain(self, sdf_path):
        if sdf_path is None or not sdf_path.endswith(".sdf"):
            return np.nan
        scores = docking_protocol_map[self.protocol_name].read_energy_from_sdf(sdf_path)
        i = calc_fragment_position(sdf_path, self.frag_restrain[0], self.frag_restrain[1])
        if i is None:
            return 0.
        else:
            s = scores[i]
            docking_protocol_map[self.protocol_name].extract_pose(sdf_path, sdf_path, i)
            return s
        
    def dock_restrain_parallel(self, result_paths, old_scores):
        pool = multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.9))
        new_score = old_scores.copy()
        mask = old_scores < 0.
        sdf_paths = [[p] for p in result_paths[mask]]
        valid_scores = []
        for score in pool.imap(partial(unpack_helper, self.fragment_position_restrain), sdf_paths):
            valid_scores.append(score)
        new_score[mask] = np.array(valid_scores)
        pool.close()
        pool.join()
        return new_score
        
    def update_interaction(self, iteration, new_names, interaction_score):
        result_df = self.get_result_df(iteration)
        result_df["interaction"] = interaction_score
        avg_result = calc_average_df(result_df, "interaction")
        self.update_result_df(avg_result, iteration)
        avg_result.index = avg_result.ligand_names
        result_dict = avg_result["adjusted_interaction"].to_dict()
        return [result_dict.get(name, np.nan) for name in new_names]
    
    def get_result_df(self, iteration):
        return pd.read_csv(self.output_dir / Path(f"{iteration}.csv"))
    
    def update_result_df(self, df, iteration):
        df.to_csv(self.output_dir / Path(f"{iteration}.csv"), index=False)

if __name__ == '__main__':
    import time
    assigner = vina_score_assigner("/global/scratch/users/ozhang/covid/rdkit_vina/ns3_protease.pdbqt", 
            [-22., 4.5, -25., -4., 22.5, -1.], 
            "/global/scratch/users/ozhang/covid/ZikvPro/vina_test", timeout=300, 
            fragment=["Cc2cc(N)c1cccc(Cl)c1n2", [-13.79580109,  15.63052044, -17.64702725]])
    start_time = time.time()
    returns = assigner.get_scores(["C1NC1(C(C#N)COC5=CC(F)=CC3(CC=2C=CC(Cl)=CC=C))[NH]C=2CN3C=C4C=C(Cl)C=CC4=N5",
        "Cc1cc(NCc2cccc3cc(-c4ccccc4C(=O)O)ccc23)c2cccc(Cl)c2n1",
        "Cc1cc(NCc2ccccc2-c2nc(-c3cccc4ccccc34)no2)c2ccccc2n1", 
        "Cc1cc(NC(=O)c2cc3cc4c(cc3s2)NCCCC4)c2cccc(Cl)c2n1", 
        "Cc1cc(-c2ccc(N3Cc4cccc(Cl)c4C3=O)cc2F)c2cccc(C#N)c2n1"], ["1-1", "2-2", "1-2", "2-1", "1-0"])
    print(returns)
    print('Time elapsed: %.2f seconds' % (time.time() - start_time))
