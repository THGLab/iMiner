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

from iMiner.utils import dist_mat
from iMiner.docking.base import BaseDocking


protein_prep_path = '/global/home/users/jerry-li1996/src/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor'
# need to first go in the virtual env named vina
meeko_ligprep_path = "/global/home/users/jerry-li1996/.conda/envs/iMiner/bin/mk_prepare_ligand.py"
meeko_ligconv_path = "/global/home/users/jerry-li1996/.conda/envs/iMiner/bin/mk_copy_coords.py"


class LigandPDBQT:
    def __init__(self, template: Optional[Union[Chem.rdchem.Mol, os.PathLike]] = None):
        self.data = {
            "ID": [],
            "resname": [],
            "element": [],
            "resid": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
            "vdw": [],
            "elec": [],
            "q": [],
            "type": []
        }
        self._rdmol = None
        if template is None:
            self.template = template
        elif isinstance(template, Chem.rdchem.Mol):
            self.template = template
        elif Path(template).suffix == ".mol":
            self.template = Chem.MolFromMolFile(str(template), removeHs=False)
        elif Path(template).suffix == ".sdf":
            self.template = Chem.SDMolSupplier(str(template), removeHs=False)[0]
        else:
            raise ValueError(f"Invaild template input/type: {template}")
            
    @property
    def df(self):
        return pd.DataFrame(self.data)
    
    @property
    def coords(self):
        return np.array([self.data['x_coord'], self.data['y_coord'], self.data['z_coord']]).T
    
    def parse_line(self, line: str):
        if line.startswith("ATOM"):
            ls = line.split()
            self.data['ID'].append(int(ls[1]))
            self.data['resname'].append(ls[2])
            self.data['element'].append(ls[3])
            self.data['resid'].append(int(ls[4]))
            self.data['x_coord'].append(float(ls[5]))
            self.data['y_coord'].append(float(ls[6]))
            self.data['z_coord'].append(float(ls[7]))
            self.data['vdw'].append(float(ls[8]))
            self.data['elec'].append(float(ls[9]))
            self.data['q'].append(float(ls[10]))
            self.data['type'].append(ls[11])

    def read_file(self, fname: os.PathLike):
        with open(fname, 'r') as f:
            for line in f:
                self.parse_line(line)

    def map_template_coords(self, tcrd: np.ndarray):
        dmat = dist_mat(self.coords, tcrd)
        mapping = {"pdbqt_atom_id": [], "template_atom_id": []}
        for i in range(dmat.shape[0]):
            argmin = np.argmin(dmat[i])
            min_dist = dmat[i][argmin]
            if min_dist > 1e-3:
                raise ValueError(f"No template atom mapped for atom {i}")
            else:
                mapping['pdbqt_atom_id'].append(i)
                mapping['template_atom_id'].append(argmin)
        
        return pd.DataFrame(mapping)
    
    def get_mapping(self):
        return self.map_template_coords(self.template.GetConformer(0).GetPositions())
    
    def get_rdmol(self, mapping: pd.DataFrame, reset_h: bool = True):
        assert self.template is not None, "No template found"
        # mapping = self.map_template_coords(self.template.GetConformer(0).GetPositions())
        new_mol = deepcopy(self.template)
        conf = new_mol.GetConformer(0)
        for pdbqt_id, tmpl_id in zip(mapping['pdbqt_atom_id'], mapping['template_atom_id']):
            conf.SetAtomPosition(tmpl_id, self.coords[pdbqt_id])
        if reset_h:
            self._rdmol = Chem.AddHs(Chem.RemoveHs(new_mol), addCoords=True)
        return self._rdmol
    
    def write_sdf(self, fname: os.PathLike, mapping: pd.DataFrame, reset_h: bool = True):
        writer = Chem.SDWriter(str(fname))
        writer.write(self.get_rdmol(mapping, reset_h), confId=0)
        writer.close()
    
    def write_mol(self, fname: os.PathLike, mapping: pd.DataFrame, reset_h: bool = True):
        Chem.MolToMolFile(self.get_rdmol(mapping, reset_h), str(fname), confId=0)


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

        :return: True, if the run is successful
        '''
        try:
            out = subprocess.run([meeko_ligprep_path, '-i', sdf_path, '-o', output_path])
        except subprocess.CalledProcessError as e:
            print('Bad molecule: '+ sdf_path)
            return False
        
        return True

    def convert_adresult_to_sdf(self, adresult_path, output_path):
        '''
        Convert a pdbqt/dlg file to a sdf file for standard file formatting

        :param adresult_path: str, path to the pdbqt/dlg file
        :param output_path: str, path to the output sdf file

        :return: True, if the run is successful
        '''
        try:
            out = subprocess.run([meeko_ligconv_path, adresult_path, '-o', output_path])
        except subprocess.CalledProcessError as e:
            return False
        
        return True
    
    def read_smiles_from_pdbqt(self, pdbqt_path):
        '''
        Read a smiles string from pdbqt file, only for pdbqt generated by meeko
    
        :param pdbqt_path: str, path to the pdbqt file
    
        :return: str, the smiles string
        '''
        with open(pdbqt_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK SMILES'):
                    return line.strip().split()[-1]
    
    def read_smiles_from_dlg(self, dlg_path):
        '''
        Read a smiles string from a ligand dlg file, only for docking with pdbqt generated by meeko
    
        :param dlg_path: str, path to the dlg file
    
        :return: str, the smiles string
        '''
        with open(dlg_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('INPUT-LIGAND-PDBQT: REMARK SMILES'):
                    return line.strip().split()[-1]
