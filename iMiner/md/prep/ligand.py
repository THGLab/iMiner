import os
from typing import Optional, Union, List
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem

from iMiner.utils import dist_mat
from iMiner.cmd import find_executable, run_command


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
            map_i = np.argwhere(np.logical_not(dmat[i])).flatten()
            if len(map_i) == 0:
                raise ValueError(f"No template atom mapped for atom {i}")
            elif len(map_i) > 1:
                raise ValueError(f"Multiple template atoms maaped for atom {i}")
            else:
                mapping['pdbqt_atom_id'].append(i)
                mapping['template_atom_id'].append(map_i[0])
        
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


def run_acpype(input: Union[str, Path, None] = None,
               basename: str = "MOL",
               charge_method: str = "bcc",
               atom_type: str = "gaff2",
               net_charge: Union[int, str] = "guess",
               args: Union[None, List[str]] = None):
    """
    Run acpype

    Parameters
    ----------
    input : str or Path or None
        input file name with extension that `acpype -i` support
    basename : str
        a basename for the project, `acpype -b` option
    charge method : str
        gas, bcc (default), user (user's charges in mol2 file)
    atom_type : str
        atom type, can be 'gaff', 'gaff2', 'amber' (AMBER14SB) or 'amber2' (AMBER14SB + GAFF2), default is gaff2
    net_charge : int or "guess"
        net molecular charge, default is 0. If "guess", acpype will guess a charge
    args : List[str] or None
        arguments used to run acpype. if `args` is not None, all other arguments are ignored
    """
    acpype = find_executable("acpype")
    if args is not None:
        cmd = [acpype] + args
    else:
        assert input is not None, "Input is None."
        if net_charge == "guess":
            cmd = [acpype, "-i", str(input), "-b", basename, "-c", charge_method, "-a", atom_type]
        else:
            cmd = [acpype, "-i", str(input), "-b", basename, "-c", charge_method, "-a", atom_type, "-n", str(net_charge)]
    
    return_code, out, err = run_command(cmd, raise_error=True)
    return 