#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Overall layout of the code:
    1, We make RDKit Mol objects from the sdf files
    2, We go through the molecules and see which atoms have the same symbols and positions in all molecules
    3, We make dictionaries giving the correspondance between old and new coordinates
    4, We make new conformers where the first n atoms are the same for all molecules
    5, We write these out to sdf files
'''
from rdkit import Chem

def finish_dict(mol, mol_dict, count): 
    for atom in mol.GetAtoms():
        k = atom.GetIdx()
        if k not in mol_dict:
            mol_dict[k] = count
            count += 1
    return mol_dict, count

def make_new_mol(old_mol, mol_dict, count):
    assert count == len(old_mol.GetAtoms())
    new_mol = Chem.RWMol()
    rev_dict = {v:k for k,v in mol_dict.items()}
    for k in range(count):
        atom = old_mol.GetAtomWithIdx(rev_dict[k])
        new_mol.AddAtom(atom)
    for bond in old_mol.GetBonds():
        new_begin_idx = mol_dict[bond.GetBeginAtomIdx()]
        new_end_idx = mol_dict[bond.GetEndAtomIdx()]
        new_mol.AddBond(new_begin_idx, new_end_idx, bond.GetBondType())
    conf = old_mol.GetConformer()
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    for old_idx, new_idx in mol_dict.items():
        pos = conf.GetAtomPosition(old_idx)
        new_conf.SetAtomPosition(new_idx, pos)
    new_mol.AddConformer(new_conf)
    return new_mol

def similar_coords(conf1, idx1, conf2, idx2, tol=0.01): #test if coordinates are similar to within some tolerance
    c1, c2 = conf1.GetAtomPosition(idx1), conf2.GetAtomPosition(idx2)
    if ((c1.x - c2.x)**2 + (c1.y - c2.y)**2 + (c1.z - c2.z)**2)**0.5 <= tol:
        return True
    return False

def get_similar_atoms(molA, molB, molC):
    confA, confB, confC = molA.GetConformer(), molB.GetConformer(), molC.GetConformer()
    dictA, dictB, dictC = {}, {}, {}
    counter = 0
    for A in molA.GetAtoms():
        idxA = A.GetIdx()
        for B in molB.GetAtoms():
            idxB = B.GetIdx()
            if similar_coords(confA, idxA, confB, idxB) and A.GetSymbol() == B.GetSymbol():
                for C in molC.GetAtoms():
                    idxC = C.GetIdx()
                    if similar_coords(confA, idxA, confC, idxC) and A.GetSymbol() == C.GetSymbol():
                        dictA[idxA], dictB[idxB], dictC[idxC] = counter, counter, counter
                        counter += 1
    return dictA, dictB, dictC, counter

def renumber_indices(molA, molB, molC):
    dictA, dictB, dictC, counter = get_similar_atoms(molA, molB, molC)
    counterA, counterB, counterC = counter, counter, counter
    dictA, counterA = finish_dict(molA, dictA, counterA)
    dictB, counterB = finish_dict(molB, dictB, counterB)
    dictC, counterC = finish_dict(molC, dictC, counterC)
    newmolA = make_new_mol(molA, dictA, counterA)
    newmolB = make_new_mol(molB, dictB, counterB)
    newmolC = make_new_mol(molC, dictC, counterC)
    return newmolA, newmolB, newmolC

if __name__ == '__main__':
    sdf1 = 'zikv_hel/data/lig_2836_h.sdf'
    sdf2 = 'zikv_hel/data/lig_2841_h.sdf'
    sdf_mcs = 'zikv_hel/data/mcs.sdf'
    
    mol1 = Chem.SDMolSupplier(sdf1, removeHs=False)[0]
    mol2 = Chem.SDMolSupplier(sdf2, removeHs=False)[0]
    mol_mcs = Chem.SDMolSupplier(sdf_mcs, removeHs=False)[0]
    newmol_mcs, newmol1, newmol2 = renumber_indices(mol_mcs, mol1, mol2)
    
    Chem.SDWriter('zikv_hel/data/2836_test.sdf').write(newmol1)
    Chem.SDWriter('zikv_hel/data/2841_test.sdf').write(newmol2)