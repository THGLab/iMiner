import numpy as np
import os.path 
import multiprocessing
from functools import partial
#from pathlib import Path

from iMiner.utils import random_id
from iMiner.cmd import run_command
from iMiner.md.analysis.interaction import analyze_single_frame

    
def make_complex(ligand_path, protein_path, idx):
    # with multiple models, idx defines the index of model
    with open(protein_path, "r+") as p:
        protein = p.read()
        protein = protein.replace("END\n", "")
        
    with open(ligand_path, "r+") as l:
        content = l.read()
    ligfile = content.split("ENDMDL")[idx]
    if "HETATM" not in ligfile: return None
    
    hetlines = ligfile.split("HETATM")[1:]
    ligand = "HETATM" + "HETATM".join(hetlines[:-1] + [hetlines[-1].split("\n")[0]])
    #print(idx+1, ligand_path, ligand)
    comp = ligand_path.replace(".pdb", f"_{idx+1}_complex.pdb")
    with open(comp, "w+") as c:
        c.write(protein)
        c.write("TER\n")
        c.write(ligand)
    return comp
    
def sdf_to_pdb(sdf_path, pdb_path):
    cmd_sdf_2_pdb = "obabel -isdf {} -opdb -O{}".format(sdf_path, pdb_path)
    code, out, err = run_command(cmd_sdf_2_pdb, raise_error=False) 
    if code != 0:
        print(err)
        return False
    return True

def unpack_helper(func, args):
    '''
    Helper function to unpack arguments for multiprocessing
    '''
    return func(*args)    

class InteractionScorer():
    """
    protein_path: str or os.PathLike, path to protein pdb
    key_res: list of str, residue format amino acid 3 letter code/resID/chainID
    
    ToDo: interaction types
    """
    def __init__(self, protein_path, key_residues) -> None:
        self.protein = protein_path
        self.residue = key_residues
        self.n_jobs = int(multiprocessing.cpu_count() * 0.9)
    
    def calc_score(self, ligand):
        """
        calculate score based on protein ligand interactions involving specified key residues
        
        ligand: str or os.PathLike, path to sdf file
        
        return:
        an interaction score np.float or None if calculation errors,
        index of the model/pose with the highest interaction score
        """
        if ligand is None or not ligand.endswith(".sdf"): 
            return 0
        pdbname = f"/tmp/{os.path.basename(ligand).replace('.sdf', '_'+random_id()+'.pdb')}"
        if not sdf_to_pdb(ligand, pdbname) or not os.path.exists(pdbname):
            return 0
        # number of models in pdb
        with open(pdbname, "r") as f:
            pdbfile = f.read()
        npdbs = len(pdbfile.split("ENDMDL\n")) 
        if npdbs > 1: 
            npdbs -= 1
        ridx = range(npdbs)
        if "SELECTED MODELS: " in pdbfile:
            ridx = [int(i) for i in pdbfile.split("SELECTED MODELS: ")[1].split("\n")[0].split()]
        
        scorelist = []
        for n in ridx:
            score = 0
            complx = make_complex(pdbname, self.protein, n)
            if complx is None:
                return 0
            interact = analyze_single_frame(complx, "UNL")
            for key in interact:
                if "/".join(key.split("/")[1:]) in self.residue:
                    score += 1
            scorelist.append(score)
            # clean temporary files
            os.remove(complx)
        max_score = np.mean(scorelist)
        # clean temporary files
        os.remove(pdbname)
        return max_score 
    
    def calc_score_parallel(self, ligands):
        ligands = [[ligand] for ligand in ligands]
        results = []
        #print("parallel interaction calculation")
        with multiprocessing.Pool(self.n_jobs) as pool:
            for result in pool.map(self.calc_score, ligands):
                results.append(result)
        
        return np.array(results)
    
# for testing
if __name__ == '__main__':
    
    IS = InteractionScorer("/global/scratch/users/ozhang/covid/ZikvPro/ns3_protease.pdb", 
                    ["SER/81/A", "ASP/83/A", "ASP/75/B", "HIS/51/B", "GLY/153/B", "GLY/151/B", 
                   "SER/135/B", "TYR/161/B", "TYR/130/B", "ASP/129/B", "GLY/159/B", 
                   "VAL/154/B", "VAL/155/B"])
    s = IS.calc_score_parallel([f"/global/scratch/users/ozhang/covid/rdkit_vina/outputs/{n}_out.sdf" for n in range(3)])
    print(s)