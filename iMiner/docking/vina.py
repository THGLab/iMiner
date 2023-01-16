'''
Author: Jie Li, Oufan Zhang
Date Created: Nov 3, 2022

Defines the docking class for AutoDock Vina and Autodock Vina GPU
'''

from iMiner.docking.autodock import AutoDockBaseDocking
from iMiner.cmd import run_command, set_directory
from iMiner.utils import random_id
from pathlib import Path
from typing import Optional
import itertools
import os
from os import path
import numpy as np
import re
import pandas as pd

VINA_BINARY = Path(path.abspath(path.dirname(__file__))) / 'bins/vina'
VINA_GPU_SCRIPT = Path(path.abspath(path.dirname(__file__))) / 'run_vina_gpu.sh'

class VinaDocking(AutoDockBaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path: Optional[os.PathLike] = None, logger=None, **kwargs) -> None:
        super().__init__(protein_pdb, docking_box, logger=logger)
        self.working_path = Path(temp_path) / "{}-vina".format(self.protein_name)
        os.makedirs(self.working_path, exist_ok = True)
        if not os.path.exists(self.working_path / "{}.pdbqt".format(self.protein_name)):
            self.convert_pdb_to_pdbqt(protein_pdb, self.working_path / "{}.pdbqt".format(self.protein_name))
        self.docking_box = docking_box

        self.write_config(**kwargs)

    def write_config(self, exhaustiveness=8, num_modes=1, energy_range=30, **kwargs):
        '''
        Write the config file for AutoDock Vina docking

        :param exhaustiveness: int, the exhaustiveness of the docking
        :param num_modes: int, the number of modes (conformations) to be generated
        :param energy_range: int, the energy range of the docking

        '''

        config_fp = self.working_path / "config.txt"
        lines = ["receptor = ./{}.pdbqt".format(self.protein_name),
                 "",
                 "center_x = {}".format((self.docking_box[0] + self.docking_box[3]) / 2),
                 "center_y = {}".format((self.docking_box[1] + self.docking_box[4]) / 2),
                 "center_z = {}".format((self.docking_box[2] + self.docking_box[5]) / 2),
                 "",
                 "size_x = {}".format(self.docking_box[3] - self.docking_box[0]),
                 "size_y = {}".format(self.docking_box[4] - self.docking_box[1]),
                 "size_z = {}".format(self.docking_box[5] - self.docking_box[2]),
                 "",
                 "num_modes = {}".format(num_modes),
                 "energy_range = {}".format(energy_range),
        ]

        # exhaustiveness may be None to accomondate Vina-GPU config
        if exhaustiveness is not None:
            lines.append("exhaustiveness = {}".format(exhaustiveness))
        with open(config_fp, "w") as f:
            f.write("\n".join(lines))

            
    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf/.pdbqt file

        :return: pd.DataFrame with columns ["original_names", "smiles", "score"]
        '''
        
         # prepare lists to record results
        ligand_smiles = []
        ligand_scores = []
        ligand_conformation_paths = []

        for ligand in ligands:
            ligand_name = Path(ligand).stem
            ligand_work_name = ligand_name + "_" + random_id()
            succ = self.convert_sdf_to_pdbqt(ligand, self.working_path / "{}.pdbqt".format(ligand_work_name))
            if not (succ and os.path.exists(self.working_path / "{}.pdbqt".format(ligand_work_name))):
                continue
            # save the ligand smiles
            ligand_smiles.append(self.convert_sdf_to_smiles(ligand))

            # execute vina docking under the working directory
            with set_directory(self.working_path):
                cmd = f"{VINA_BINARY} --config config.txt --ligand {ligand_work_name}.pdbqt --score_only"
                code, out, err = run_command(cmd, timeout=100)

            # special handling if calculation job times out
            if code == 999:
                ligand_scores.append(np.nan)
                ligand_conformation_paths.append("calculation timed out!")
                continue

            # obtain docking score from the results
            strings = re.split('Estimated Free Energy of Binding   :', out)
            line = strings[1].split('\n')[0]
            energy = float(line.strip().split()[0])
            ligand_scores.append(energy)
        
        # generate the final pandas dataframe and return
        df = pd.DataFrame({"original_names": ligands, "smiles": ligand_smiles,
             "score": ligand_scores})
        return df

    def _get_cmd(self, ligand_work_name):
        cmd = f"{VINA_BINARY} --config config.txt --ligand {ligand_work_name}.pdbqt " + \
                    f"--out {ligand_work_name}_out.pdbqt"
        return cmd
        

    def dock(self, ligands, output_dir, single_job_timeout=None):
        '''
        Run actual Autodock Vina docking

        :param ligands: list of ligands, each ligand is a path (str or os.PathLike) to the corresponding .sdf file
        :param output_dir: str, path to the output directory where sdf files for the docked conformations will be saved
        :param single_job_timeout: int, timeout for each docking job in seconds. When it is None, no timeout will be set
        '''
        # prepare lists to record results
        ligand_smiles = []
        ligand_scores = []
        ligand_conformation_paths = []

        # First make sure output_dir exists
        os.makedirs(output_dir, exist_ok = True)

        # convert output_dir to Path object
        output_dir = Path(output_dir)

        for ligand in ligands:
            ligand_name = Path(ligand).stem
            ligand_work_name = ligand_name + "_" + random_id()
            succ = self.convert_sdf_to_pdbqt(ligand, self.working_path / "{}.pdbqt".format(ligand_work_name))
            if not (succ and os.path.exists(self.working_path / "{}.pdbqt".format(ligand_work_name))):
                continue
            # save the ligand smiles
            ligand_smiles.append(self.convert_sdf_to_smiles(ligand))

            # execute vina docking under the working directory
            with set_directory(self.working_path):
                cmd = self._get_cmd(ligand_work_name)
                code, out, err = run_command(cmd, timeout=single_job_timeout, raise_error=False)

            # special handling if calculation job times out
            if code == 999:
                ligand_scores.append(np.nan)
                ligand_conformation_paths.append("calculation timed out!")
                continue

            if code != 0:
                ligand_scores.append(np.nan)
                ligand_conformation_paths.append(err)
                continue

            # obtain docking score from the results
            energy = np.nan
            strings = re.split('\n', out)
            for line in strings:
                if line[0:4] == '   1':
                    energy = float(re.split(' +', line)[2])
            ligand_scores.append(energy)

            # save the conformation
            # first check if the conformation is generated
            if os.path.exists(self.working_path / f"{ligand_work_name}_out.pdbqt"):
                # find the first available name
                if os.path.exists(output_dir / f"{ligand_name}.sdf"):
                    i = 1
                    while os.path.exists(output_dir / f"{ligand_name}_{i}.sdf"):
                        i += 1
                    ligand_name = f"{ligand_name}_{i}"
                succ = self.convert_adresult_to_sdf(self.working_path / "{}_out.pdbqt".format(ligand_work_name),
                        output_dir / "{}.sdf".format(ligand_name))
                if succ and os.path.exists(output_dir / "{}.sdf".format(ligand_name)):
                    ligand_conformation_paths.append(str(output_dir / "{}.sdf".format(ligand_name)))
                else:
                    ligand_conformation_paths.append(None)
            else:
                ligand_conformation_paths.append(None)
        
        # generate the final pandas dataframe and return
        df = pd.DataFrame({"original_names": ligands, "smiles": ligand_smiles,
             "score": ligand_scores, "path": ligand_conformation_paths})
        return df
            


class VinaGPUDocking(VinaDocking):
    def __init__(self, protein_pdb, docking_box, temp_path: Optional[os.PathLike] = None, logger=None, **kwargs) -> None:
        AutoDockBaseDocking.__init__(self, protein_pdb, docking_box, logger=logger)
        self.working_path = Path(temp_path) / "{}-vina-gpu".format(self.protein_name)
        os.makedirs(self.working_path, exist_ok = True)
        if not os.path.exists(self.working_path / "{}.pdbqt".format(self.protein_name)):
            self.convert_pdb_to_pdbqt(protein_pdb, self.working_path / "{}.pdbqt".format(self.protein_name))
        self.docking_box = docking_box

        self.write_config(**kwargs)

    def write_config(self, num_modes=1, energy_range=30, **kwargs):
        '''
        Write the config file for AutoDock Vina docking

        :param num_modes: int, the number of modes (conformations) to be generated
        :param energy_range: int, the energy range of the docking

        '''
        super().write_config(exhaustiveness=None, num_modes=num_modes, 
            energy_range=energy_range, **kwargs)

    def dock(self, ligands, output_dir, single_job_timeout=None, gpu=0):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return super().dock(ligands, output_dir, single_job_timeout)

    def _execute_docking(self, ligand_work_name):
        cmd = f"sh {VINA_GPU_SCRIPT} {ligand_work_name}"
        return cmd

    def _get_parallel_docking_args(self, ligands, output_dir, single_job_timeout, n_jobs):
        iterator = itertools.cycle(range(n_jobs))
        job_cpus = [next(iterator) for _ in range(len(ligands))]
        zipped_args = zip(ligands, [output_dir] * len(ligands), [single_job_timeout] * len(ligands), job_cpus)
        return zipped_args

    # def dock(self, ligands, output_dir):
    #     '''
    #     Run actual Autodock Vina GPU docking

    #     :param ligands: list of ligands, each ligand is a path (str or os.PathLike) to the corresponding .sdf file
    #     :param output_dir: str, path to the output directory where sdf files for the docked conformations will be saved
    #     '''
    #     # prepare lists to record results
    #     ligand_smiles = []
    #     ligand_scores = []
    #     ligand_conformation_paths = []

    #     # First make sure output_dir exists
    #     os.makedirs(output_dir, exist_ok = True)

    #     # convert output_dir to Path object
    #     output_dir = Path(output_dir)

    #     for ligand in ligands:
    #         ligand_name = Path(ligand).stem
    #         ligand_work_name = ligand_name + "_" + timestamp(hashed=True)
    #         succ = self.convert_sdf_to_pdbqt(ligand, self.working_path / "{}.pdbqt".format(ligand_work_name))
    #         if not (succ and os.path.exists(self.working_path / "{}.pdbqt".format(ligand_work_name))):
    #             continue
    #         # save the ligand smiles
    #         ligand_smiles.append(self.convert_sdf_to_smiles(ligand))

    #         # execute vina docking under the working directory
    #         with set_directory(self.working_path):
    #             cmd = f"{VINA_GPU_BINARY} --config config.txt --ligand {ligand_work_name}.pdbqt " + \
    #                 "--out {ligand_work_name}_out.pdbqt"
    #             code, out, err = run_command(cmd)

    #         # obtain docking score from the results
    #         energy = np.nan
    #         strings = re.split('\n', out)
    #         for line in strings:
    #             if line[0:4] == '   1':
    #                 energy = float(re.split(' +', line)[2])
    #         ligand_scores.append(energy)

    #         # save the conformation
    #         # first check if the conformation is generated
    #         if os.path.exists(self.working_path / f"{ligand_work_name}_out.pdbqt"):
    #             # find the first available name
    #             if os.path.exists(output_dir / f"{ligand_name}.sdf"):
    #                 i = 1
    #                 while os.path.exists(output_dir / f"{ligand_name}_{i}.sdf"):
    #                     i += 1
    #                 ligand_name = f"{ligand_name}_{i}"
    #             succ = self.convert_adresult_to_sdf(self.working_path / "{}_out.pdbqt".format(ligand_work_name),
    #                     output_dir / "{}.sdf".format(ligand_name))
    #             if succ and os.path.exists(output_dir / "{}.sdf".format(ligand_name)):
    #                 ligand_conformation_paths.append(str(output_dir / "{}.sdf".format(ligand_name)))
    #             else:
    #                 ligand_conformation_paths.append(None)
    #         else:
    #             ligand_conformation_paths.append(None)
        
    #     # generate the final pandas dataframe and return
    #     df = pd.DataFrame({"smiles": ligand_smiles, "score": ligand_scores, "path": ligand_conformation_paths})
    #     return df

if __name__ == "__main__":
    print(VINA_BINARY)
