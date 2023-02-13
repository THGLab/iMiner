'''
Author: Jie Li
Date Created: Feb 3, 2023

Defines the docking class using TankBind
'''

from pathlib import Path
from typing import Optional
import os
from iMiner.docking.base import BaseDocking
from iMiner.cmd import run_command, set_directory
from iMiner.utils import random_id
from iMiner.pathlib import tankbind_python_path, tankbind_dir_path, tankbind_src_path, p2rank_path
from rdkit import Chem
import shutil
import pandas as pd

my_env = os.environ.copy()
my_env["PATH"] = f"{tankbind_dir_path}:" + my_env["PATH"]
os.environ.update(my_env)
class TankBindDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path: Optional[os.PathLike] = None, logger=None, **kwargs) -> None:
        super().__init__(protein_pdb, docking_box)
        self.working_path = temp_path / "{}-tankbind".format(self.protein_name)
        os.makedirs(self.working_path, exist_ok = True)
        self.docking_box = docking_box
        if kwargs.get("ignore_docking_box", False):
            self.docking_box = None
        self.prepare_protein(protein_pdb, self.docking_box)


    def prepare_protein(self, protein_pdb, docking_box=None):
        protein_pdb = os.path.abspath(protein_pdb)
        cmd = [tankbind_python_path, f"{tankbind_src_path}/prepare_protein.py", f"--protein_pdb={protein_pdb}",
             f'--p2rank_cmd=bash {p2rank_path}']
        if docking_box is not None: # when docking box is provided, only use that center position. otherwise, consider all pockets
            center = [(docking_box[0] + docking_box[3]) / 2, (docking_box[1] + docking_box[4]) / 2, (docking_box[2] + docking_box[5]) / 2]
            center = ",".join([str(x) for x in center])
            cmd.append(f"--center={center}")
        with set_directory(self.working_path):
            run_command(cmd)
            assert os.path.exists("protein.pkl"), "Failed to prepare protein"

    def run_docking(self, ligand_content_csv, unique_id, device):
        with set_directory(self.working_path):
            # path_modifier = f"PATH={tankbind_dir_path}:$PATH"
            cmd = f"{tankbind_python_path} {tankbind_src_path}/dock_ligands.py --protein_data protein.pkl " + \
                 f"--ligands {ligand_content_csv} --device {device} --output_dir {unique_id}"
            run_command(cmd)
            assert os.path.exists(os.path.join(unique_id, "prediction_info.csv")), "Failed to generate docking results"


    def dock(self, ligands, output_dir, gpu=None):
        '''
        Run actual docking with ICM-pro

        :param ligands: list of ligands, each ligand is a path (str or os.PathLike) to the corresponding .sdf file
        :param output_dir: str, path to the output directory where sdf files for the docked conformations will be saved
        :param gpu: int, the gpu id to use for docking. If None, use CPU
        '''
        # prepare lists to record results
        ligand_smiles = []
        ligand_scores = []
        ligand_conformation_paths = []

        # First make sure output_dir exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)

        # generate a unique id for this batch of docking, so that files will not overwrite each other if multiprocessing
        unique_id = random_id()

        # prepare ligand content csv file
        ligand_names = [Path(ligand).stem for ligand in ligands]
        ligand_paths = [str(Path(ligand).absolute()) for ligand in ligands]
        ligand_content = pd.DataFrame({"compound_name": ligand_names, "sdf_path": ligand_paths})
        ligand_content.to_csv(self.working_path / f"ligand_content_{unique_id}.csv", index = False)

        # run TankBind docking
        if gpu is None:
            device = "cpu"
        else:
            device = f"cuda:{gpu}"
        self.run_docking(self.working_path / f"ligand_content_{unique_id}.csv", unique_id, device)

        # print(ligands)
        # generate final results and copy docked conformations to output_dir
        docked_results = pd.read_csv(self.working_path / f"{unique_id}/prediction_info.csv")
        docked_results.index = docked_results.compound_name.astype(str)
        pockets = docked_results.pocket_name.to_dict()
        scores = (-docked_results.affinity).to_dict()  # keep in mind the original affinity is traned from -logK
        for name, ligand in zip(ligand_names, ligands):
            # save the ligand smiles
            ligand_smiles.append(self.convert_sdf_to_smiles(ligand))
            # print(scores)
            ligand_scores.append(scores[name])
            # copy the docked conformation to output_dir
            ligand_pocket = pockets[name]
            docked_sdf = self.working_path / f"{unique_id}/{ligand_pocket}_{name}.sdf"
            shutil.copy(docked_sdf, output_dir / f"{name}.sdf")
            ligand_conformation_paths.append(str(output_dir / f"{name}.sdf"))

        df = pd.DataFrame({"original_names": ligands, "smiles": ligand_smiles, "score": ligand_scores, "path": ligand_conformation_paths})
        df["index"] = df.index
        return df

    def _get_parallel_docking_args(self, ligands, output_dir, single_job_timeout, n_jobs):
        devices = [None] * len(ligands)
        zipped_args = zip(ligands, [output_dir] * len(ligands), devices)
        return zipped_args