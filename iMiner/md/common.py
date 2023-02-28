"""
Author: Eric Wang
Date: 01/14/2023

Common functions used in iMiner.md sub-package
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from iMiner.cmd import find_executable, run_command


def gmx_genidx(input_file: os.PathLike, output_file: Optional[os.PathLike] = None, input: str = ""):
    gmx = find_executable(["gmx_mpi", "gmx"])
    cmds = [gmx, 'make_ndx', '-f', input_file]
    if output_file:
        cmds += ['-o', output_file]
    run_command(cmds, input=input+'q\n')
    return 


def read_single_gro(fname: os.PathLike):
    """
    Read single-frame Gromacs GRO file 

    Parameters
    ----------
    fname: os.PathLike
        Path to GRO file

    Return
    ------
    coords: np.ndarray
        Coordinates, in unit of nm
    """
    coords = []
    with open(fname, "r") as f:
        f.readline()
        natoms = int(f.readline().strip())
        cnt = 0
        for line in f:
            coords.append([float(x) for x in line.split()[-3:]])
            cnt += 1
            if cnt == natoms:
                break
        
    coords = np.array(coords)
    return coords


def parse_index_file(fname: os.PathLike) -> Dict[str, List[int]]:
    """
    Parse GROMACS index file
    
    Parameters
    ----------
    fname: os.PathLike
        Path to GROMACS index file (.ndx)
    
    Return
    ------
    groupdict: Dict[str, List[int]]
        Dict with groups' names and corresponding atom indices
    """
    groupdict = {}
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("["):
                name = line[1:].strip(']\n').strip()
                groupdict[name] = list()
                continue
            
            for x in line.strip().split():
                groupdict[name].append(int(x))
    return groupdict


def mk_index_file(groupdict: Dict[str, List[int]], fname: os.PathLike, overwrite: bool = True):
    """
    Make a GROMACS index file (.ndx)
    
    Parameters
    ----------
    groupdict: Dict[str, List[int]]
        Dict with groups' names and corresponding atom indices
    fname: os.PathLike
        Path to GROMACS index file (.ndx)
    overwrite: bool
        Whether to overwrite and existing file
    """
    if os.path.isfile(fname) and (not overwrite):
        raise FileExistsError(f"{fname} exists")
    
    num_per_line = 15
    with open(fname, 'w') as f:
        for name, indices in groupdict.items():
            f.write(f'[ {name} ]\n')
            for i in range(0, len(indices), num_per_line):
                f.write(" ".join(f'{x:>4}' for x in indices[i: i + num_per_line]))
                f.write('\n')

                
def preprocess_index_file(
    f_ndx_ori: os.PathLike,
    f_ndx_new: os.PathLike,
    overwrite: bool = True,
    ligand_name: str = "MOL"
) -> Tuple[int, int]:
    """
    Pre-process index file
    
    Combine protein with ions except [Na+] and [Cl-] and get group id of 
    receptor and ligand
    
    Parameters
    ----------
    f_ndx_ori: os.PathLike
        Path to original index file
    f_ndx_new: os.PathLike
        Path to pre-processed index file
    overwrite: bool
        Whether to overwrite index file, default True
    ligand_name: str
        Group name of ligand, default "MOL"
    
    Return
    ------
    (r_grp_idx, l_grp_idx, c_grp_idx): Tuple[int, int]
        Group index of receptor, ligand and complex
    """
    receptor_name = 'Receptor'
    group_dict = parse_index_file(f_ndx_ori)
    group_dict[receptor_name] = group_dict['Protein'].copy()
    for ndx in group_dict.get('Ion', []):
        if ndx in group_dict.get('NA', []):
            continue
        if ndx in group_dict.get('CL', []):
            continue
        group_dict[receptor_name].append(ndx)
    complex_name = f"{ligand_name}_{receptor_name}"
    group_dict[complex_name] =  group_dict[ligand_name] + group_dict[receptor_name]
    
    mk_index_file(group_dict, f_ndx_new, overwrite)
    keys = list(group_dict.keys())
    r_grp_idx = keys.index(receptor_name)
    l_grp_idx = keys.index(ligand_name)
    c_grp_idx = keys.index(complex_name)
    return r_grp_idx, l_grp_idx, c_grp_idx