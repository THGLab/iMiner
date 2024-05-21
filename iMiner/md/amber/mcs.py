import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
try:
    from rdkit.Chem.rdFMCS import FindMCS
except:
    from rdkit.Chem.MCS import FindMCS



def find_mcs(molA, molB):
    molA_noh = Chem.RemoveHs(molA)
    molB_noh = Chem.RemoveHs(molB)
    
    mcs = FindMCS([molA_noh, molB_noh])
    # handle mcs failed
    mcs_failed = mcs.canceled if hasattr(mcs, 'canceled') else False
    mcs_failed = (mcs_failed or mcs.numAtoms == 0)
    if mcs_failed:
        raise RuntimeError("MCS Failed")
        
    mcsSmarts = mcs.smarts if hasattr(mcs, 'smarts') else mcs.smartsString
    
    mcs_struct = Chem.MolFromSmarts(mcsSmarts)
    return mcs_struct


def get_common_core(molA, molB, mcs, use_position_info=True):
    mcs = Chem.RemoveHs(mcs)
    ccA = molA.GetSubstructMatch(mcs)
    ccB = molB.GetSubstructMatch(mcs)
    cc = []
    
    if use_position_info:
        posA = molA.GetConformer().GetPositions()
        posB = molB.GetConformer().GetPositions()
        dist_mat = cdist(posA, posB)
    for indexA, indexB in zip(ccA, ccB):
        atomA = molA.GetAtomWithIdx(indexA)
        atomB = molB.GetAtomWithIdx(indexB)
        atomA_hs = [nei.GetIdx() for nei in atomA.GetNeighbors() if nei.GetSymbol() == 'H']
        atomB_hs = [nei.GetIdx() for nei in atomB.GetNeighbors() if nei.GetSymbol() == 'H']
        
        cc.append((indexA, indexB))

        if use_position_info:
            if len(atomA_hs) < len(atomB_hs):
                for hA in atomA_hs:
                    hB = np.argmin(dist_mat[hA])
                    if hB in atomB_hs:
                        cc.append((hA, hB))
            else:
                for hB in atomB_hs:
                    hA = np.argmin(dist_mat[:, hB])
                    if hA in atomA_hs:
                        cc.append((hA, hB))
        else:
            for hA, hB in zip(atomA_hs, atomB_hs):
                cc.append((hA, hB))
            
    cc.sort(key=lambda x: x[0])
    cc = np.array(cc, dtype=int)
    return cc


def get_common_core_from_soft_core(scA, scB, natomsA, natomsB):
    ccA = [i for i in range(natomsA) if i not in scA]
    ccB = [i for i in range(natomsB) if i not in scB]
    assert len(ccA) == len(ccB)
    cc = list(zip(ccA, ccB))
    cc.sort(key=lambda x: x[0])
    return np.array(cc, dtype=int)


def check_common_core(posA, posB, cc, pos_tol=1e-4, check_order=True, check_pos=True):
    cc = [(int(x[0]), int(x[1])) for x in cc]
    cc.sort(key=lambda x: x[0])
    last = -1
    for a, b in cc:
        if check_order: assert b > last
        if check_pos: assert np.allclose(posA[a], posB[b], atol=pos_tol)
        last = b

        
def generate_mask(natomsA, natomsB, ccA, ccB):
    scA = np.array([i for i in range(natomsA) if i not in ccA])
    scB = np.array([i for i in range(natomsB) if i not in ccB])
    res = {
        "noshakemask": f"'@1-{natomsA+natomsB}'",
        "timask1": f"'@1-{natomsA}'",
        "timask2": f"'@{natomsA+1}-{natomsA+natomsB}'",
        "scmask1": "'@{}'".format(','.join(str(i+1) for i in scA)),
        "scmask2": "'@{}'".format(','.join(str(i+1+natomsA) for i in scB))
    }
    return res