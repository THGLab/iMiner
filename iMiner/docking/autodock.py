'''
Author: Jie Li, Oliver Sun, Eric Wang
Date Created: Oct 31, 2022

Implementation of the autodock docking protocol, including Autodock4, Autodock Vina, and Autodock Vina GPU
'''
from pathlib import Path
import subprocess
import os
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem

from iMiner.utils import dist_mat, random_id
from iMiner.docking.base import BaseDocking
from iMiner.pathlib import *


class AutoDockBaseDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        super().__init__(protein_pdb, docking_box, temp_path, **kwargs)


    def convert_pdb_to_pdbqt(self, pdb_path, output_path, add_h = True):
        '''
        Convert a pdb file to a pdbqt file that can be used for AutoDock docking

        :param pdb_path: str, path to the pdb file
        :param output_path: str, path to the output pdbqt file

        :return: True if the run is successful
        '''
        protein_path = Path(pdb_path).resolve()
        
        # run protein preparation depending on if there's need to add H
        if add_h:
            try:
                out = subprocess.run([protein_prep_path, '-r', protein_path, '-o', output_path,\
                '-A', 'checkhydrogens'])
            except subprocess.CalledProcessError as e:
                return e.output
        else:
            try:
                out = subprocess.run([protein_prep_path, '-r', protein_path, '-o', output_path,])
            except subprocess.CalledProcessError as e:
                return e.output
        
        return True
    
    def convert_pdbqt_to_flex_rigid(self, pdbqt_path, flex_res):
        """
        Convert a prepared pdbqt to a rigid file and flexible file for 
        flexible residue docking.

        :param pdbqt_path: str, relative path to the prepared pdbqt file
        :param flex_res: str, prepared string for flexible residues
        """
        try:
            out = subprocess.run([pythonsh_path, flexrec_prep_path, '-r', pdbqt_path, '-s', flex_res])
        except subprocess.CalledProcessError as e:
            print(e.output)
        
        return True

    @staticmethod
    def convert_sdf_to_pdbqt(sdf_path, output_path):
        '''
        Convert a ligand sdf file to a pdbqt file that can be used for AutoDock docking

        :param sdf_path: str, path to the sdf file
        :param output_path: str, path to the output pdbqt file

        :return: True, if the run is successful
        '''
        try:
            out = subprocess.run([meeko_ligprep_path, '-i', sdf_path, '-o', output_path])
        except subprocess.CalledProcessError as e:
            print('Bad molecule: '+ sdf_path)
            return False
        
        return True

    @staticmethod
    def convert_adresult_to_sdf(adresult_path, output_path, idx=-1):
        '''
        Convert a pdbqt/dlg file to a sdf file for standard file formatting

        :param adresult_path: str, path to the pdbqt/dlg file
        :param output_path: str, path to the output sdf file
        :param idx: int, the index of model to export

        :return: True, if the run is successful
        '''
        comment = ""
        if isinstance(idx, list): 
            comment = " ".join(idx)
            idx = -1
        temp_path = output_path if idx==-1 else f"/tmp/{random_id()}_{os.path.basename(output_path)}" 
        try:
            out = subprocess.run([meeko_ligconv_path, adresult_path, '-o', temp_path])
        except subprocess.CalledProcessError as e:
            return False
        if len(comment) > 0: 
            with open(temp_path, "a") as f:
                f.write("\n>  <REMARK>\nSELECTED MODELS: "+comment)
        if idx == -1: return True
        return AutoDockBaseDocking.extract_pose(temp_path, output_path, idx)
        
        
    @staticmethod
    def extract_pose(sdf_path, output_path, idx=0):
        '''
        extract the pose with idx in sdf containing multiple models
        
        :param sdf_path: str, path to the sdf result file
        :param output_path: str, path to the output sdf file
        :param idx: int, index of pdbqt model to extract

        :return: True, if the run is successful
        '''
        #assert not Path(sdf_path) == Path(output_path)
        with open(sdf_path, "r") as f:
            all_sdf = f.read().split("$$$$\n")
        # single model in pdbqt
        if len(all_sdf) == 1:
            with open(output_path, "w+") as f1:
                f1.write(all_sdf[0])
            return True
        elif len(all_sdf) - 1 <= idx:
            idx = 0
        # save the selected model
        with open(output_path, "w+") as f1:
            f1.write(all_sdf[idx])
        return True
    
    @staticmethod
    def read_smiles_from_pdbqt(pdbqt_path):
        '''
        Read a smiles string from pdbqt file, only for pdbqt generated by meeko
    
        :param pdbqt_path: str, path to the pdbqt file
    
        :return: str, the smiles string
        '''
        with open(pdbqt_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK SMILES'):
                    return line.strip().split()[-1]
    
    @staticmethod
    def read_energy_from_pdbqt(pdbqt_path):
        '''
        Read vina score from pdbqt file, only for pdbqt generated by meeko
    
        :param pdbqt_path: str, path to the pdbqt file
    
        :return: float, vina score
        '''
        with open(pdbqt_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK VINA RESULT:'):
                    return np.float(line.strip().split()[3])
    
    @staticmethod
    def read_smiles_from_dlg(dlg_path):
        '''
        Read a smiles string from a ligand dlg file, only for docking with pdbqt generated by meeko
    
        :param dlg_path: str, path to the dlg file
    
        :return: str, the smiles string
        '''
        with open(dlg_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('INPUT-LIGAND-PDBQT: REMARK SMILES'):
                    return line.strip().split()[-1]
