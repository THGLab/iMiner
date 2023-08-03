from iMiner.docking.base import BaseDocking
from iMiner.cmd import run_command, set_directory
from iMiner.utils import random_id
from iMiner.pathlib import (
    ign_dir_path, ign_python_path, 
    ign_src_path, ign_lib_path
)
import os
import shutil
from pathlib import Path
 
my_env = os.environ.copy()
my_env["PATH"] = f"{ign_dir_path}:" + my_env["PATH"]
os.environ.update(my_env)

class IGNscoring(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path=None, logger=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        super().__init__(protein_pdb, docking_box, temp_path, **kwargs)   
        self.working_path = Path(temp_path) / "{}-ign".format(self.protein_name)
        os.makedirs(self.working_path, exist_ok = True)
        
        self.protein_path = self.working_path / "{}.pdb".format(self.protein_name)
        # work around for not installing ADFR; adds *.pdb as second protein with name *_
        if protein_pdb.endswith(".pdbqt"):
            shutil.copy(protein_pdb.replace(".pdbqt", "_.pdb"), self.protein_path)
        elif protein_pdb.endswith(".pdb"):
            shutil.copy(protein_pdb, self.protein_path)
    
    def run_prediction(self, ligand_txt, out_csv, device, n_jobs):
        ign_dic_path = self.working_path / f"ign_dic"
        os.makedirs(ign_dic_path, exist_ok = True)
        
        with set_directory(self.working_path):
            # path_modifier = f"PATH={tankbind_dir_path}:$PATH"
            cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{ign_lib_path}"
            run_command(cmd, shell=True)
            cmd = f"{ign_python_path} {ign_src_path}/rescore.py" + \
                f" --protein {self.protein_path} --ligands {ligand_txt}" + \
                f" --graph_dic_path {ign_dic_path}" + \
                f" --device {device} --output_dir {out_csv} --num_process {n_jobs}"
            run_command(cmd)
    
    def rescore(self, ligands, n_jobs=8, gpu=None):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["ligand_names", "smiles", "ign_score"]
        '''
        unique_id = random_id()
        
        if not os.path.exists(self.protein_path):
            print(self.protein_path, "does not exist")
            return
        ligand_dir = ligands[0].split("/")[-1]
        with open(self.working_path / f"ligand_content_{unique_id}.txt", "w") as f:
            f.write("\n".join(ligands))
        if gpu is None:
            device = "cpu"
        else:
            device = gpu
        self.run_prediction(self.working_path / f"ligand_content_{unique_id}.txt", 
            self.working_path / f"ign_rescore_{unique_id}.csv", device, n_jobs)
        
        df = pd.read_csv(self.working_path / f"ign_rescore_{unique_id}.csv")
        df["ign_score"] *= -1.36 #convert pkd to kcal/mol
        df["ign_path"] = df["ligand_names"].apply(lambda x : ligand_dir + "/" + x)
        df["smiles"] = df["ign_path"].apply(self.convert_sdf_to_smiles)
        
        return df
        

        