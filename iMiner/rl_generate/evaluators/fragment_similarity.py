import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.BRICS import BRICSDecompose

def remove_connect_from_smiles(smi):
    """
    remove connection annotations from smiles string
    """
    chunks = smi.split("*]")
    new = ""
    for i in range(len(chunks)):
        if i == len(chunks) - 1:
            new += chunks[i]
        else:
            new += "[".join(chunks[i].split("[")[:-1])
    # remove empty parenthesis
    nc = new.split("()")
    return "".join(nc)

def calc_cm(atoms, coords):
    coords_ = np.array(coords).T
    lib = {"H": 1, "C": 12, "N": 14, "O": 16, "Cl": 35.5, "F": 19, "S": 32,
           "Se": 79, "P": 31, "Br": 80, "I": 126, "B": 10.8
    }
    atom_weights = np.array([lib[a] for a in atoms])
    return (atom_weights*coords_).sum(axis=-1)/np.sum(atom_weights)
    
    
def calc_fragment_position(pose_path, fragment, frag_cm, threshold=1.8):
    """
    extract the coordinates of the most similar part of a molecule to a given fragment for each poses, 
    and return the pose whose identified fragment-like coordinates are in proximity to 
    the center of mass of the fragment within threshold
    """
    fragment_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(fragment), 2, nBits=1024)
    suppl = Chem.SDMolSupplier(pose_path)
    mol_idx = 0
    try:
        # rdkit fragments molecule
        pose_frags = list(BRICSDecompose(suppl[0], keepNonLeafNodes=True, returnMols=True))
        pose_fbit = [AllChem.GetMorganFingerprintAsBitVect(f, 2, nBits=1024) for f in pose_frags]
    except Exception:
        #print("error fragmenting molecule")
        return None
    scores = [DataStructs.DiceSimilarity(fragment_fp, fb) for fb in pose_fbit]
    if np.max(scores) < 0.35:
        #print("low similarity score")
        return None
    # choose highest scored fragment
    fbit_idx = scores.index(np.max(scores))
    fp_smi = remove_connect_from_smiles(Chem.MolToSmiles(pose_frags[fbit_idx]))
        
    for mol in suppl:
        try:
            sub = mol.GetSubstructMatch(Chem.MolFromSmiles(fp_smi))
        except Exception:
            #print(f"kekulization error {fp_smi}")
            return None
        atom_types = []
        fbit_coords = []
        conf = mol.GetConformer()
        #print(np.max(scores), fp_smi, sub)
        for s in sub:
           atom_types.append(mol.GetAtoms()[s].GetSymbol())
           fbit_coords.append(list(conf.GetAtomPosition(s)))
        if len(atom_types) == 0:
            #print("error finding substructure")
            continue
        docked_cm = calc_cm(atom_types, fbit_coords)
        if np.sqrt(((docked_cm - np.array(frag_cm))**2).sum()) < threshold:
            return mol_idx
        mol_idx += 1
    return None
    

class FragmentScorer():
    def __init__(self, fragments, weights=None, use_features=False, global_substructure_match=False, similarity_metric=DataStructs.DiceSimilarity) -> None:
        self.fragments = fragments
        self.use_features = use_features
        self.fragment_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(f), 2,
            nBits=1024, useFeatures=use_features) for f in fragments]
        if weights is None:
            weights = np.ones(len(fragments))
        self.weights = weights
        self.similarity = similarity_metric
        self.global_substructure_match = global_substructure_match


    def calc_score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fragments = [Chem.MolFromSmiles(fs) for fs in BRICSDecompose(mol, keepNonLeafNodes=self.global_substructure_match)]
        mol_fragment_fps = [AllChem.GetMorganFingerprintAsBitVect(frag, 2, nBits=1024, 
            useFeatures=self.use_features) for frag in fragments]
        fragment_scores = [max([self.similarity(frag_fp, mol_frag_fp) for mol_frag_fp in mol_fragment_fps])
                                     for frag_fp in self.fragment_fps]
        final_score = np.dot(fragment_scores, self.weights)
        return final_score
        
if __name__ == '__main__':
    import time
    start_time = time.time()
    path = [f"/global/scratch/users/ozhang/covid/rdkit_vina/outputs/{n}_out.sdf" for n in range(3)]
    for n in range(3):
        print(n, calc_fragment_position(path[n], "Cc2cc(N)c1cccc(Cl)c1n2", [-13.79580109,  15.63052044, -17.64702725]))
    print(time.time() - start_time)
   