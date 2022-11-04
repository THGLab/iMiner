'''
Author: Jie Li, Oliver Sun
Date Created: Oct 31, 2022

Implementation of the autodock docking protocol, including Autodock4, Autodock Vina, and Autodock Vina GPU
'''

from iMiner.docking.base import BaseDocking
from pathlib import Path
import subprocess
import os

protein_prep_path = '/global/home/groups/co_armada2/local/ADFRsuite/bin/prepare_receptor'
meeko_ligprep_path = "/global/home/groups/co_armada2/local/Meeko/scripts/mk_prepare_ligand.py"

class AutoDockBaseDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box):
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        pass


    def convert_pdb_to_pdbqt(self, pdb_path, output_path, add_h = True):
        '''
        Convert a pdb file to a pdbqt file that can be used for AutoDock docking

        :param pdb_path: str, path to the pdb file
        :param output_path: str, path to the output pdbqt file

        :return: the message for completing the conversion
        '''
        protein_path = Path(pdb_path).resolve()
        protein_name = protein_path.stem
        protein_folder = protein_path.parent

        # preprocess the protein by removing water & heteroatoms
        # save the processed protein in the same folder as original pdb file
        with open(pdb_path, "r") as f:
            protein_file = f.read().split("\n")
        new_file = [i for i in protein_file if not i.startswith('HETATM')]
        processed_fp = os.path.join(protein_folder, "{}-processed.pdb".format(protein_name))
        with open(processed_fp, "w") as f1:
            f1.write("\n".join(new_file))
        
        # run protein preparation depending on if there's need to add H
        if add_h:
            try:
                out = subprocess.run([protein_prep_path, '-r', processed_fp, '-o', output_path,\
                '-A', 'checkhydrogens'])
            except subprocess.CalledProcessError as e:
                return e.output
        else:
            try:
                out = subprocess.run([protein_prep_path, '-r', processed_fp, '-o', output_path,])
            except subprocess.CalledProcessError as e:
                return e.output
        
        return True

    def convert_sdf_to_pdbqt(self, sdf_path, output_path):
        '''
        Convert a ligand sdf file to a pdbqt file that can be used for AutoDock docking

        :param sdf_path: str, path to the sdf file
        :param output_path: str, path to the output pdbqt file
        '''
        try:
            out = subprocess.run([meeko_ligprep_path, '-i', sdf_path, '-o', output_path])
        except subprocess.CalledProcessError as e:
            return e.output
        
        return True