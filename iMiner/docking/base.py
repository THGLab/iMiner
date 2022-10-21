'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseDocking class with interfaces to be realized by different docking protocols
'''

class BaseDocking:
    def __init__(self, protein_pdb, docking_box) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        pass

    def dock(self, ligands, output_dir):
        '''
        Dock a list of ligands to the pocket in the protein

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory

        :return: pd.DataFrame with columns ["index", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        pass

    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["index", "smiles", "score"]
        '''
        pass

