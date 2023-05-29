'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseDocking class with interfaces to be realized by different docking protocols
'''

from pathlib import Path
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures._base import TimeoutError
import numpy as np
import math


def unpack_helper(func, args):
    '''
    Helper function to unpack arguments for multiprocessing
    '''
    return func(*args)
    
    
class BaseDocking:
    def __init__(self, protein_pdb, docking_box, temp_path=None, logger=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        :param logger: a logger object to log some results
        :param temp_path: str, path to the temporary directory
        '''
        self.protein_path = Path(protein_pdb).resolve()
        self.protein_folder = self.protein_path.parent
        self.protein_name = self.protein_path.stem
        self.logger = logger


    def dock(self, ligands, output_dir, single_job_timeout=120):
        '''
        Dock a list of ligands to the pocket in the protein

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param single_job_timeout: int, timeout for each job in seconds

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        pass

    def _get_parallel_docking_args(self, ligands, output_dir, single_job_timeout, n_jobs):
        zipped_args = zip(ligands, [output_dir] * len(ligands), [single_job_timeout] * len(ligands))
        return zipped_args


    def dock_parallel(self, ligands, output_dir, n_jobs=1, single_job_timeout=120, verbose=True, save_df_freq=500, **kwargs):
        '''
        Dock a list of ligands to the pocket in the protein in parallel

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param n_jobs: int, number of jobs to run in parallel
        :param single_job_timeout: int, timeout for each job in seconds
        :param verbose: bool, whether to show progress bar
        :param save_df_freq: int, frequency to save the results to the disk (to prevent losing results)

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        
        pool = ProcessPoolExecutor(n_jobs)
        ligands = [[ligand] for ligand in ligands]
        zipped_args = self._get_parallel_docking_args(ligands, output_dir, single_job_timeout, n_jobs)
        counter = 0
        results = []
        if verbose:
            pbar = tqdm(total=len(ligands))
        futures = [pool.submit(self.dock, *args) for args in zipped_args]
        for ligand, future in zip(ligands, futures):
            try:
                result = future.result(timeout=single_job_timeout)
            except TimeoutError:
                result = pd.DataFrame({"original_names": ligand, "smiles": ["timeout"],
                               "score": [np.nan], "path": [""]})
            counter += 1
            results.append(result)
            if verbose:
                pbar.update(1)
            if counter % save_df_freq == 0:
                df = pd.concat(results)
                df.to_csv(Path(output_dir) / "results.csv", index=False)
                if self.logger is not None:
                    self.logger.info(f"Saved checkpoint results to {output_dir}/results.csv")

        # clean up
        # pool.close()
        # pool.join()
        del pool

        final_results = pd.concat(results)
        final_results.reset_index(inplace=True)
        return final_results
        

    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["ligand_name", "smiles", "score"]
        '''
        pass

    def rescore_parallel(self, ligands, output_dir, n_jobs=1, n_chunks=5, single_job_timeout=500, verbose=True, save_df_freq=100, **kwargs):
        '''
        Rescore a list of ligands in their provided poses with the protein in parallel

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param n_jobs: int, number of jobs to run in parallel
        :param n_chunks: int, number of chunks to split the ligands into
        :param single_job_timeout: int, timeout for each job in seconds
        :param verbose: bool, whether to show progress bar
        :param save_df_freq: int, frequency to save the results to the disk (to prevent losing results)

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the original conformation
        '''
        
        pool = ProcessPoolExecutor(n_jobs)
        chunk_size = math.ceil(len(ligands) / n_chunks)
        ligands = [[ligands[i * chunk_size: (i + 1) * chunk_size]] for i in range(n_chunks)]

        counter = 0
        results = []
        if verbose:
            pbar = tqdm(total=n_chunks)
        futures = [pool.submit(self.rescore, *args) for args in ligands]
        for ligand, future in zip(ligands, futures):
            try:
                result = future.result(timeout=single_job_timeout)
            except TimeoutError:
                len_lig = len(ligand)
                result = pd.DataFrame({"ligand_name": ligand, "smiles": ["timeout"] * len_lig,
                               "score": [np.nan] * len_lig})
            counter += 1
            results.append(result)
            if verbose:
                pbar.update(1)
            if counter % save_df_freq == 0:
                df = pd.concat(results)
                df.to_csv(Path(output_dir) / "results.csv", index=False)
                if self.logger is not None:
                    self.logger.info(f"Saved checkpoint results to {output_dir}/results.csv")

        # clean up
        # pool.close()
        # pool.join()
        del pool

        final_results = pd.concat(results)
        final_results.reset_index(inplace=True)
        return final_results

    @staticmethod
    def convert_sdf_to_smiles(sdf_path):
        '''
        Convert a ligand sdf file to a smiles string, to be used by any of the docking protocols

        :param sdf_path: str, path to the sdf file

        :return: str, the smiles string
        '''
        sdmol = Chem.SDMolSupplier(sdf_path)
        mol = sdmol[0]
        try:
            return Chem.MolToSmiles(mol)
        except RuntimeError:
            return None

