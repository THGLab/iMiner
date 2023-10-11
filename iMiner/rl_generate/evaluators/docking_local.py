'''
Author: Jie Li, Oufan Zhang
Date created: Aug 22, 2022

Doing local Autodock Vina docking to obtain the scores
'''

import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import TimeoutError

from iMiner.core.consensus_docking import ConsensusDocking, docking_protocol_map
from iMiner.rl_generate.utils import get_gpu_count
from iMiner.rl_generate.evaluators.fragment_similarity import calc_fragment_position

from rdkit import Chem
#from rdkit.Chem import Draw
import os
from pathlib import Path

def nan_average(x):
    """
    return the average of an array excluding the nan/zero entries; 
    return nan if x is nan
    """
    if len(x) == 1:
        return x[0]
    cx = np.nan_to_num(x)
    nnan = (cx == 0).sum()
    if nnan == 0:
        return np.mean(x)
    if nnan == len(x):
        return np.nan
    return cx.sum()/(len(x) - nnan)

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
    for i in range(1, df.shape[0]):
        if i == df.shape[0] - 1:
            score = nan_average(data[st:]) 
            new_scores += [score]*(i + 1 - st)
        elif dkey[i] != dkey[st]:
            score = nan_average(data[st:i])
            new_scores += [score]*(i - st)
            st = i
    sorted_df["adjusted_"+column_name] = new_scores
    return sorted_df.sort_index()
        
    
class docking_score_assigner():
    def __init__(self, protein_file, box, path, protocols, timeout=60, fragment=None, **kwargs) -> None:
        '''
        protocols: list of docking programs
        timeout: the timeout for each docking job
        fragment: path to fragment restraints .sdf
        '''
        protein_name = os.path.basename(protein_file).split('.')[0]
        self.protein = protein_file
        self.protocol_name = [protocols] if isinstance(protocols, str) else protocols
        self.docking_project = ConsensusDocking(protein_name, path, self.protocol_name)
        self.docking_project.add_protein(protein_file_path=protein_file, name=protein_name, binding_site=box)
        self.output_dir = self.docking_project.project_path / Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frag_restrain = fragment
        if "vina-gpu" in protocols:
            self.n_jobs = get_gpu_count()
        else:
            self.n_jobs = int(multiprocessing.cpu_count() * 0.9)
        self.timeout = timeout
        self.docking_args = kwargs


    def get_scores(self, smiles, new_names, iteration=0):
        added_new_names = self.docking_project.add_multiple_ligands(smiles, names=new_names)
        self.docking_project.run_consensus_docking(ligand_names=added_new_names, n_jobs=self.n_jobs, 
            output_csv=self.output_dir / Path(f"{iteration}.csv"), 
            single_job_timeout=self.timeout, **self.docking_args)
        result_df = self.get_result_df(iteration)
        
        if self.frag_restrain is not None:
            # decay of fragment restraints
            #frag_similarity = self.frag_restrain.get("similarity_threshold", 0.2) + \
            #        self.frag_restrain.get("similarity_increment", 0.02) * int(iteration % 10 == 0 and iteration > 0)
            #self.frag_restrain["similarity_threshold"] = np.minimum(frag_similarity, self.frag_restrain.get("similarity_clip", 0.4))
            
            #frag_position = self.frag_restrain.get("distance_threshold", 0.8) * \
            #        self.frag_restrain.get("distance_decay", 0.99) ** int(iteration % 10 == 0 and iteration > 0)
            #self.frag_restrain["distance_threshold"] = np.maximum(frag_position, self.frag_restrain.get("distance_clip", 0.6))
            #print("fragment restraints:", self.frag_restrain["similarity_threshold"], self.frag_restrain["distance_threshold"])
                
            for protocol in self.protocol_name:
                protocol_name = protocol
                if protocol == "vina-gpu":
                    protocol_name = "vina"
                new_scores = self.dock_restrain_parallel(
                    result_df.smiles.values,
                    result_df[protocol_name+"_path"].values, 
                    result_df[protocol_name+"_score"].values, protocol)
   
                # enforce 0 score for fragment poses not in the specified position
                if protocol in ["rfscore", "ign"]:
                    new_scores[result_df["vina_score"].values == 0] = 0
                result_df[protocol_name+"_score"] = new_scores
        
        # sum all docking scores (TODO: arithmetic mean, geometric mean, weights)
        colname = ["vina_score" if p=="vina-gpu" else p+"_score" for p in self.protocol_name]
        result_df["score"] = result_df[colname].sum(axis=1)
                
        # average docking scores for the same smiles 
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
        
    
    def fragment_position_restrain(self, smile, sdf_path, protocol):
        if sdf_path is None or not sdf_path.endswith(".sdf") or not os.path.exists(sdf_path):
            return np.nan
    
        if protocol in ["vina", "vina-gpu"]:
            scores = docking_protocol_map[protocol].read_energy_from_sdf(sdf_path)
            assert len(scores) == self.docking_args.get("num_modes", 1)
            i = calc_fragment_position(smile, sdf_path, **self.frag_restrain)
            if i is None:
                return 0.
            else:
                docking_protocol_map[protocol].extract_pose(sdf_path, sdf_path, i)
                return scores[i]
                
        elif protocol in ["rfscore", "ign"]:
            RFscore = docking_protocol_map[protocol](self.protein, [])
            return RFscore.rescore([sdf_path])[f"{protocol}_score"][0]
        
    def dock_restrain_parallel(self, result_smiles, result_paths, old_score, protocol):
        pool = ProcessPoolExecutor(self.n_jobs)
        new_score = old_score.copy()
        mask = old_score < 0.
        zipped_args = zip(result_smiles[mask], result_paths[mask], [protocol] * np.sum(mask))

        valid_scores = []
        futures = [pool.submit(self.fragment_position_restrain, *args) for args in zipped_args]
        for future in futures:
            try:
                score = future.result(timeout=self.timeout)
            except TimeoutError:
                score = 0.
            valid_scores.append(score)
        new_score[mask] = np.array(valid_scores)
        del pool
        return new_score
        
        
    def update_interaction(self, iteration, new_names, interaction_score):
        result_df = self.get_result_df(iteration)
        result_df["interaction"] = interaction_score
        # update interaction scores for the same smiles
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
    assigner = docking_score_assigner("/global/scratch/users/ozhang/covid/rdkit_vina/ns3_protease.pdbqt", 
            [-22., 4.5, -25., -4., 22.5, -1.], 
            "/global/scratch/users/ozhang/covid/ZikvPro/vina_test", timeout=300, 
            fragment="/global/scratch/users/ozhang/covid/test.sdf")
    start_time = time.time()
    returns = assigner.get_scores(["C1NC1(C(C#N)COC5=CC(F)=CC3(CC=2C=CC(Cl)=CC=C))[NH]C=2CN3C=C4C=C(Cl)C=CC4=N5",
        "Cc1cc(NCc2cccc3cc(-c4ccccc4C(=O)O)ccc23)c2cccc(Cl)c2n1",
        "Cc1cc(NCc2ccccc2-c2nc(-c3cccc4ccccc34)no2)c2ccccc2n1", 
        "Cc1cc(NC(=O)c2cc3cc4c(cc3s2)NCCCC4)c2cccc(Cl)c2n1", 
        "Cc1cc(-c2ccc(N3Cc4cccc(Cl)c4C3=O)cc2F)c2cccc(C#N)c2n1"], ["1-1", "2-2", "1-2", "2-1", "1-0"])
    print(returns)
    print('Time elapsed: %.2f seconds' % (time.time() - start_time))
