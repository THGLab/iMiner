'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseDocking class with interfaces to be realized by different docking protocols
'''

from pathlib import Path
from rdkit import Chem
import multiprocessing
import pandas as pd

class BaseDocking:
    def __init__(self, protein_pdb, docking_box, temp_path=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        :param temp_path: str, path to the temporary directory
        '''
        self.protein_path = Path(protein_pdb).resolve()
        self.protein_folder = self.protein_path.parent
        self.protein_name = self.protein_path.stem


    def dock(self, ligands, output_dir, single_job_timeout=120):
        '''
        Dock a list of ligands to the pocket in the protein

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param single_job_timeout: int, timeout for each job in seconds

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        raise NotImplementedError()

    def dock_parallel(self, ligands, output_dir, n_jobs=1, single_job_timeout=120):
        '''
        Dock a list of ligands to the pocket in the protein in parallel

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param n_jobs: int, number of jobs to run in parallel
        :param single_job_timeout: int, timeout for each job in seconds

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        pool = multiprocessing.Pool(n_jobs)
        zipped_args = zip(ligands, [output_dir] * len(ligands), [single_job_timeout] * len(ligands))
        results = pool.starmap(self.dock, zipped_args)
        final_results = pd.concat(results)
        final_results.reset_index(inplace=True)
        return final_results
        

    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["index", "smiles", "score"]
        '''
        pass

    def convert_sdf_to_smiles(self, sdf_path):
        '''
        Convert a ligand sdf file to a smiles string, to be used by any of the docking protocols

        :param sdf_path: str, path to the sdf file

        :return: str, the smiles string
        '''
        sdmol = Chem.SDMolSupplier(sdf_path)
        mol = sdmol[0]
        return Chem.MolToSmiles(mol)

