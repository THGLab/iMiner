'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseProject class with associated logic for project initialization
The protein is supposed to be a pdb file and the ligand is supposed to be a sdf file
'''

import os
import shutil
from rdkit.Chem import MolFromSmiles, AddHs, AllChem, SDWriter

class BaseProject:
    def __init__(self, project_name, project_path=None) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name
        '''
        # setup project folders
        if project_path is None:
            project_path = os.path.join(os.getcwd(), project_name)
        self.project_path = project_path
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        os.mkdir(os.path.join(project_path, 'proteins'))
        os.mkdir(os.path.join(project_path, 'ligands'))

        # prepare temp path
        self.temp_path = os.path.join(project_path, 'tmp')
        os.mkdir(self.temp_path)

        # prepare protein (with binding sites) mapping dicts that map names to the corresponding file paths
        # because ligands do not have names, the ligand paths are saved as a list
        self.proteins = {}
        self.binding_sites = {}
        self.ligands = []

        # provide a cache for processed protein pdb files (in case multiple binding sites are defined for the same protein)
        self._protein_processed_cache = set()

    def add_protein(self, name, protein_file_path, binding_site=None):
        '''
        Add a protein with corresponding binding site definition to the project

        :param name: str, the name of the protein + pocket
        :param protein_file_path: str, path to the protein file
        :param binding_site: (xmin, ymin, zmin, xmax, ymax, zmax), the binding site definition
        '''
        self.binding_sites[name] = binding_site
        # preprocess the protein file and save the clean file to project_path/proteins
        # if the protein has been processed before, skip the preprocessing step

    def add_ligand(self, smiles_or_path, format='inferred'):
        '''
        Add a ligand to the project

        :param smiles_or_path: str, either the smiles string of the ligand or the path to the ligand file
        :param format: one of ['inferred', 'smiles', 'sdf', 'pdb']. If format is 'inferred', the format will be inferred from the file extension
        '''

        # decide the file format
        if format == 'inferred':
            if smiles_or_path.endswith('.sdf'):
                format = 'sdf'
            elif smiles_or_path.endswith('.pdb'):
                format = 'pdb'
            elif "." not in smiles_or_path:
                format = 'smiles'

        # get ligand name according to its index in the current project
        ligand_name = f'{len(self.ligands) + 1}.sdf'

        # process the ligand file according to the format
        if format == 'smiles':
            self.ligands.append(_process_smiles(smiles_or_path, os.path.join(self.project_path, 'ligands', ligand_name)))
        elif format == 'sdf':
            # directly copy the sdf file to the corresponding position
            shutil.copy(smiles_or_path, os.path.join(self.project_path, 'ligands', ligand_name))
            self.ligands.append(os.path.join(self.project_path, 'ligands', ligand_name))
        elif format == 'pdb':
            self.ligands.append(_process_pdb(smiles_or_path, os.path.join(self.project_path, 'ligands', ligand_name)))

    def _process_smiles(self, smiles, save_path):
        '''
        Process a smiles string and save the corresponding ligand file to project_path/ligands

        :param smiles: str, the smiles string of the ligand
        :param save_path: str, the path to save the processed ligand file

        :return: str, the path to the ligand file
        '''
        mol = MolFromSmiles(smiles)
        # assert valid smiles
        if m is None:
            raise RuntimeError(smiles + ' is not a valid smile string')
        mh = AddHs(m)
        embed = AllChem.EmbedMolecule(mh, useRandomCoords=False)

        # make sure embedding is successful
        if embed! = 0:
            raise RuntimeError('RDkit fails to embed molecule ' + smiles)

        # save the ligand file to the corresponding position
        with SDWriter(save_path) as writer:
            writer.write(mh)


    def _process_pdb(self, pdb, save_path):
        '''
        Process a pdb file and save the corresponding ligand file to project_path/ligands

        :param pdb: str, the path to the pdb file
        :param save_path: str, the path to save the processed ligand file

        :return: str, the path to the ligand file
        '''
        raise NotImplementedError()