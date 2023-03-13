import os
from typing import Optional

import numpy as np
from rdkit import Chem
try:
    from rdkit.Chem.rdFMCS import FindMCS
except:
    from rdkit.Chem.MCS import FindMCS


def find_mcs(molA: Chem.Mol, molB: Chem.Mol, save_path: Optional[os.PathLike] = None) -> np.ndarray:
    """
    Find Maximum common structure between to molecules

    Parameters
    ----------
    molA: Chem.Mol
        Molecule A
    molB: Chem.Mol
        Molecule B
    save_path: os.PathLike, optional
        Path to save maaping information
    
    Returns
    -------
    mapping: np.ndarray
        Mapping info, the first column is the atom indices in molA, the second column is the atoms indices in 
        molB. Non-matching atoms are set to -1
    
    Raises
    ------
    RuntimeError:
        Raises if MCS failed to found.
    """
    mcs = FindMCS([molA, molB])
    # handle mcs failed
    if hasattr(mcs, 'canceled'):
        mcs_failed = mcs.canceled
    else:
        mcs_failed = False
    mcs_failed = (mcs_failed or mcs.numAtoms == 0)
    if mcs_failed:
        raise RuntimeError("MCS Failed")
    if hasattr(mcs, 'smarts'):
        mcsSmarts = mcs.smarts
    else:
        mcsSmarts = mcs.smartsString
    
    mcs_struct = Chem.MolFromSmarts(mcsSmarts)
    molA_match = molA.GetSubstructMatch(mcs_struct)
    molB_match = molB.GetSubstructMatch(mcs_struct)
    mapping_A_B = {}
    mapping_B_A = {}
    for i in range(len(molA_match)):
        mapping_A_B[molA_match[i]] = molB_match[i]
        mapping_B_A[molB_match[i]] = molA_match[i]
    mapping = []
    for i in range(molA.GetNumAtoms()):
        mapping.append([i, mapping_A_B.get(i, -1)])
    for i in range(molB.GetNumAtoms()):
        if i not in mapping_B_A:
            mapping.append([-1, i])
    mapping = np.array(mapping, dtype=int)
    # save
    if save_path:
        with open(save_path, 'w') as f:
            for m in mapping:
                f.write(f"{m[0]} {m[1]}\n")

    return mapping