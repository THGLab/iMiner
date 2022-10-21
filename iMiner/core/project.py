'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseProject class with associated logic for project initialization
The protein is supposed to be a pdb file and the ligand is supposed to be a sdf file
'''

class BaseProject:
    def __init__(self, project_name, project_path=None) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name
        '''
        pass

    def add_protein(self, protein_file_path, binding_site=None):
        '''
        Add a protein with corresponding binding site definition to the project

        :param protein_file_path: str, path to the protein file
        :param binding_site: (xmin, ymin, zmin, xmax, ymax, zmax), the binding site definition
        '''
        pass

    def add_ligand(self, smiles_or_path, format='inferred'):
        '''
        Add a ligand to the project

        :param smiles_or_path: str, either the smiles string of the ligand or the path to the ligand file
        :param format: one of ['inferred', 'smiles', 'sdf', 'pdb']. If format is 'inferred', the format will be inferred from the file extension
        '''
        pass