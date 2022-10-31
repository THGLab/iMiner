'''
Author: Jie Li
Date Created: Oct 31, 2022

Implementation of the autodock docking protocol, including Autodock4, Autodock Vina, and Autodock Vina GPU
'''

from iMiner.docking.base import BaseDocking

class AutoDockBaseDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''


    def convert_pdb_to_pdbqt(self, pdb_path, output_path):
        '''
        Convert a pdb file to a pdbqt file that can be used for AutoDock docking

        :param pdb_path: str, path to the pdb file
        :param output_path: str, path to the output pdbqt file
        '''
        pass

    def convert_sdf_to_pdbqt(self, sdf_path, output_path):
        '''
        Convert a ligand sdf file to a pdbqt file that can be used for AutoDock docking

        :param sdf_path: str, path to the sdf file
        :param output_path: str, path to the output pdbqt file
        '''
        pass