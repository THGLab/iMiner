'''
Author: Jie Li, Eric Wang
Date Created: Oct 21, 2022

This file defines the BaseProject class with associated logic for project initialization
The protein is supposed to be a pdb file and the ligand is supposed to be a sdf file
'''

import os
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Optional

from rdkit.Chem import MolFromSmiles, AddHs, AllChem, SDWriter


class BaseProject:
    def __init__(self, project_name: str, project_path: Optional[os.PathLike] = None) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name
        '''
        # setup project folders
        if project_path is None:
            project_path = Path.cwd() / project_name
        self.project_path = Path(project_path).resolve()
        self.project_path.mkdir(exist_ok=True, parents=True)
        
        # setup ligands and proteins directory
        self.ligands_path = self.project_path  / "ligands"
        self.ligands_path.mkdir(exist_ok=True, parents=True)
        self.proteins_path = self.project_path / "proteins"
        self.proteins_path.mkdir(exist_ok=True, parents=True)

        # prepare temp path
        self.temp_path = Path('/tmp') / project_name
        self.temp_path.mkdir(exist_ok=True, parents=True)

        # prepare protein (with binding sites) mapping dicts that map names to the corresponding file paths
        # because ligands do not have names, the ligand paths are saved as a list
        self.proteins = OrderedDict()
        self.binding_sites = OrderedDict()
        self.ligands = OrderedDict()

        # provide a cache for processed protein pdb files (in case multiple binding sites are defined for the same protein)
        self._protein_processed_cache = set()

    def add_protein(self, protein_file_path, name=None, preprocess=False, binding_site=None):
        '''
        Add a protein with corresponding binding site definition to the project

        :param protein_file_path: path to the protein file
        :param name: the name of the protein + pocket, default None
        :param preprocess: if preprocess the protein pdb with `pdbfixer`
        :param binding_site: (xmin, ymin, zmin, xmax, ymax, zmax), the binding site definition
        :type protein_file_path: str
        :type name: str
        :type preprocess: bool
        :type binding_site: tuple
        '''
        self.binding_sites[name] = binding_site
        name = str(len(self.proteins.items()) + 1) if name is None else name
        protein_path = os.path.join(self.project_path, "proteins", f"{name}.pdb")
        if preprocess:
            raise NotImplementedError()
        else:
            shutil.copyfile(protein_file_path, protein_path)
        self.proteins[name] = protein_path

    def add_ligand(self, smiles_or_path, name=None, format='inferred'):
        '''
        Add a ligand to the project

        :param name: Name of the ligand
        :param smiles_or_path: Either the smiles string of the ligand or the path to the ligand file
        :param format: One of ['inferred', 'smiles', 'sdf', 'pdb']. If format is 'inferred', the format will be inferred from the file extension
        :type name: str
        :type smiles_or_path: str
        :type format: str
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
        if name is None:
            name = str(len(self.ligands.items()) + 1)
        ligand_name = f"{name}.sdf"
        ligand_path = os.path.join(self.project_path, 'ligands', ligand_name)

        # process the ligand file according to the format
        if format == 'smiles':
            self._process_smiles(smiles_or_path, ligand_path)
        elif format == 'sdf':
            # directly copy the sdf file to the corresponding position
            shutil.copy(smiles_or_path, ligand_path)
        elif format == 'pdb':
            self._process_pdb(smiles_or_path, ligand_path)
        
        self.ligands[name] = ligand_path

    def add_multiple_ligands(self, smiles_or_paths, names=None, format='inferred'):
        '''
        Add multiple ligands to the project

        :param smiles_or_paths: A list of smiles strings or paths to the ligand files
        :param names: A list of names for the ligands. When None, the ligands will be named with their index in the list
        :param format: One of ['inferred', 'smiles', 'sdf', 'pdb']. If format is 'inferred', the format will be inferred from the file extension
        :type smiles_or_paths: list
        :type names: list
        :type format: str
        '''
        if names is None:
            names = [str(i) for i in range(len(smiles_or_paths))]
        for smiles_or_path, name in zip(smiles_or_paths, names):
            self.add_ligand(smiles_or_path, name, format)
    
    def get_ligand_with_name(self, name) -> Path:
        """
        Get ligand file path with its name

        :param name: Name of the query ligand
        :type name: str
        :return: Path of the query ligand
        :rtype: pathlib.Path
        """
        return Path(self.ligands[name]).resolve()
    
    def get_ligand_with_index(self, idx) -> Path:
        """
        Get ligand file path with its index

        :param idx: Index of the query ligand
        :type idx: int
        :return: Path of the query ligand
        :rtype: pathlib.Path
        """
        return Path(self.ligands.items()[idx][-1]).resolve()
    
    def get_protein_with_name(self, name) -> Path:
        """
        Get ligand file path with its name

        :param name: Name of the query ligand
        :type name: str
        :return: Path of the query ligand
        :rtype: pathlib.Path
        """
        return Path(self.proteins[name]).resolve()
    
    def get_protein_with_index(self, idx) -> Path:
        """
        Get ligand file path with its index

        :param idx: Index of the query ligand
        :type idx: int
        :return: Path of the query ligand
        :rtype: pathlib.Path
        """
        return Path(self.proteins.items()[idx][-1]).resolve()
        
    def _process_smiles(self, smiles, save_path):
        '''
        Process a smiles string and save the corresponding ligand file to project_path/ligands

        :param smiles: str, the smiles string of the ligand
        :param save_path: str, the path to save the processed ligand file

        :return: str, the path to the ligand file
        '''
        mol = MolFromSmiles(smiles)
        # assert valid smiles
        if mol is None:
            raise RuntimeError(smiles + ' is not a valid smile string')
        mh = AddHs(mol)
        embed = AllChem.EmbedMolecule(mh, useRandomCoords=False)

        # make sure embedding is successful
        if embed != 0:
            raise RuntimeError('RDkit fails to embed molecule ' + smiles)

        # save the ligand file to the corresponding position
        writer = SDWriter(save_path)
        writer.write(mh)


    def _process_pdb(self, pdb, save_path):
        '''
        Process a pdb file and save the corresponding ligand file to project_path/ligands

        :param pdb: str, the path to the pdb file
        :param save_path: str, the path to save the processed ligand file

        :return: str, the path to the ligand file
        '''
        raise NotImplementedError()