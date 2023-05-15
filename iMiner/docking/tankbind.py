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
from iMiner.pathlib import rcPath
from rdkit import Chem
import shutil
import pandas as pd


class TankBindDocking(BaseDocking):
    def __init__(self, protein, docking_box, temp_path: Optional[os.PathLike] = None, logger=None, **kwargs) -> None:
        super().__init__(protein, docking_box)
        # prep working path
        self.working_path = Path(temp_path) / "{}-tankbind".format(self.protein_name)
        self.working_path.mkdir(exist_ok=True, parents=True)
        self.working_path = self.working_path.resolve()
        
        # docking box
        self.docking_box = docking_box
        
        # setup TankBind envs
        self.tankbind_python_path = rcPath.get('tankbind_python_path', 'python')
        self.tankbind_src_path = rcPath['tankbind_src_path']
        self.p2rank_path = rcPath['p2rank_path']
        if self.tankbind_python_path != "python":
            tankbind_lib_path = Path(self.tankbind_python_path).parent.parent / 'lib'
            os.environ['LD_LIBRARY_PATH'] = f"{tankbind_lib_path}:" + os.environ['LD_LIBRARY_PATH']
            os.environ['PATH'] = str(Path(self.tankbind_python_path).parent) + ":" + os.environ['PATH']
        
        # prepare protein
        self.prepare_protein(protein, docking_box)
        
        # setup TankBind model path
        self.model_path = kwargs.get("model_path", None)
        self.model_path = Path(self.model_path).resolve() if self.model_path else None

    def prepare_protein(self, protein, docking_box):
        center = [(docking_box[0] + docking_box[3]) / 2, (docking_box[1] + docking_box[4]) / 2, (docking_box[2] + docking_box[5]) / 2]
        center = ",".join([str(x) for x in center])
        protein = Path(protein).resolve()
        
        if protein.suffix == '.pkl':
            shutil.copyfile(protein, self.working_path / "protein.pkl")
        elif protein.suffix == '.pdb':
            with set_directory(self.working_path):
                cmd = [self.tankbind_python_path, f"{self.tankbind_src_path}/prepare_protein.py", f"--protein_pdb={protein}",
                    f'--center={center}', f'--p2rank_cmd="bash {self.p2rank_path}"']
                run_command(cmd)
                assert os.path.exists("protein.pkl"), "Failed to prepare protein"
        else:
            raise RuntimeError(f"Unsupported format: {protein.suffix}")

    def run_docking(self, ligand_content_csv, output_dir, device):
        ligand_content_csv = Path(ligand_content_csv).resolve()
        with set_directory(self.working_path):
            cmd = [
                self.tankbind_python_path, f"{self.tankbind_src_path}/dock_ligands.py",
                "--protein_data", "protein.pkl",
                "--ligands", ligand_content_csv,
                "--device", device,
                "--output_dir", output_dir
            ]
            if self.model_path is not None:
                cmd += ['--model_path', self.model_path]
            run_command(cmd)
            assert os.path.exists(os.path.join(output_dir, "prediction_info.csv")), "Failed to generate docking results"


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

        # generate final results and copy docked conformations to output_dir
        docked_results = pd.read_csv(self.working_path / f"{unique_id}/prediction_info.csv")
        docked_results.index = docked_results.compound_name
        pockets = docked_results.pocket_name.to_dict()
        scores = (-docked_results.affinity).to_dict()  # keep in mind the original affinity is traned from -logK
        for name, ligand in zip(ligand_names, ligands):
            # save the ligand smiles
            ligand_smiles.append(self.convert_sdf_to_smiles(ligand))
            ligand_scores.append(scores[name])
            # copy the docked conformation to output_dir
            ligand_pocket = pockets[name]
            docked_sdf = self.working_path / f"{unique_id}/{ligand_pocket}_{name}.sdf"
            shutil.copy(docked_sdf, output_dir / f"{name}.sdf")
            ligand_conformation_paths.append(str(output_dir / f"{name}.sdf"))

        df = pd.DataFrame({"original_names": ligands, "smiles": ligand_smiles, "score": ligand_scores, "path": ligand_conformation_paths})
        df["index"] = df.index
        return df
