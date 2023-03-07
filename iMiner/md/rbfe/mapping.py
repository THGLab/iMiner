from rdkit import Chem
import numpy as np


def perturb_rdmol(molA: Chem.Mol, molB: Chem.Mol, mapping: np.ndarray):
    """
    Add dummy atoms for a molecule based on another molecule and mapping info

    Parameters
    ----------
    molA: Chem.Mol
        Molecule to perturb
    molB: Chem.Mol
        Molecule to which molA is perturbed
    mapping: np.ndarray
        Mapping information, with dim Nx2
    
    Return
    ------
    mol: Chem.Mol
        Perturbed molecule
    """
    duIndices = []
    mapping_du = mapping.copy()
    rwmol = Chem.RWMol(molA)
    newpos = {}
    posA = molA.GetConformer(0).GetPositions()
    posB = molB.GetConformer(0).GetPositions()
    for i in range(mapping.shape[0]):
        atomIdx1 = mapping[i, 0]
        atomIdx2 = mapping[i, 1]
        if atomIdx1 == -1:
            du = Chem.Atom(0)
            duIdx = rwmol.AddAtom(du)
            mapping_du[i, 0] = duIdx
            duIndices.append((duIdx, atomIdx2))
            newpos[duIdx] = posB[atomIdx2]
        else:
            newpos[atomIdx1] = posA[atomIdx1]

    mapping_B_to_A = {m[1]: m[0] for m in mapping if m[1] != -1}
    for atomIdx1, atomIdx2 in duIndices:
        for nei in molB.GetAtomWithIdx(atomIdx2).GetNeighbors():
            neiIdx1 = mapping_B_to_A[nei.GetIdx()]
            rwmol.AddBond(atomIdx1, neiIdx1)
    
    mol = rwmol.GetMol()
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx in newpos:
        conf.SetAtomPosition(idx, newpos[idx])
    mol.AddConformer(conf, assignId=True)
    return mol
        