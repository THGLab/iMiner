import numpy as np
import pandas as pd
import os.path

from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit import DataStructs
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

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


def extract_substructure(mol, match):
    # Create an editable molecule.
    emol = Chem.rdchem.EditableMol(Chem.Mol())

    # Add atoms to emol.
    for idx in match:
        emol.AddAtom(mol.GetAtomWithIdx(idx))

    # Add bonds to emol if they exist in the original molecule.
    for idx1 in range(len(match)):
        for idx2 in range(idx1+1, len(match)):
            bond = mol.GetBondBetweenAtoms(match[idx1], match[idx2])
            if bond is not None:
                emol.AddBond(idx1, idx2, bond.GetBondType())

    submol = emol.GetMol()

    conf = mol.GetConformer()
    subconf = Chem.Conformer(submol.GetNumAtoms())

    # Copy the atom coordinates.
    for idx in range(len(match)):
        subconf.SetAtomPosition(idx, conf.GetAtomPosition(match[idx]))

    # Add the conformer to the substructure molecule.
    submol.AddConformer(subconf)
    return submol
    
    
def calc_fragment_position(pose_smi, pose_path, fragment_path, 
            use_scaffold=False, similarity_threshold=0.25, distance_threshold=0.8):
    """
    extract the coordinates of the most similar part of a molecule to a given fragment for each poses, 
    and return the pose whose identified fragment-like coordinates are close in shape to the fragment within threshold
    
    Notes:
    This function takes an extra smiles input instead of converting from sdf for consistencies 
    with the FragmentScorer. Inconsistencies can be caused by
    obabel generated vina-gpu poses that only populates the polar hydrogens.
    
    Fragment and distance similarity threshold should be examined by your use case.
    TODO: consider implement a threshold scheduler 
    """
    query_frag = Chem.SDMolSupplier(fragment_path)[0]
    fragment_fp = AllChem.GetMorganFingerprintAsBitVect(query_frag, 2, nBits=1024)
    suppl = Chem.SDMolSupplier(pose_path)

    if use_scaffold:
        core = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(pose_smi))
        pose_fbit = AllChem.GetMorganFingerprintAsBitVect(core, 2, nBits=1024)
        score = DataStructs.DiceSimilarity(fragment_fp, pose_fbit)
        if score < similarity_threshold:
            return None
    else:
        try:
            pose_frags = list(BRICSDecompose(Chem.MolFromSmiles(pose_smi), keepNonLeafNodes=True, returnMols=True))
            pose_fbit = [AllChem.GetMorganFingerprintAsBitVect(f, 2, nBits=1024) for f in pose_frags]
        except Exception:
            return None
        scores = [DataStructs.DiceSimilarity(fragment_fp, fb) for fb in pose_fbit]
        # rejects molecules disimilar to the target fragment lower than a threshold
        if np.max(scores) < similarity_threshold:
            return None
        # choose highest scored fragment
        fbit_idx = scores.index(np.max(scores))
        fp_smi = remove_connect_from_smiles(Chem.MolToSmiles(pose_frags[fbit_idx]))

    for mol_idx, mol in enumerate(suppl):
        if use_scaffold:
            # align shape of the most similarity/core fragment
            sub_mol = MurckoScaffold.GetScaffoldForMol(mol)
        else:
            try:
                match = mol.GetSubstructMatch(Chem.MolFromSmiles(fp_smi))
                if len(match) == 0:
                    return None
                sub_mol = extract_substructure(mol, match)
            except Exception:
                #print(f"kekulization error {fp_smi}")
                return None
            #conf = mol.GetConformer()
            # Calculate the center of mass of the substructure.
            #coords = np.array([conf.GetAtomPosition(i) for i in match])
            #masses = np.array([mol.GetAtomWithIdx(i).GetMass() for i in match])
            #center_of_mass = np.sum(coords * masses[:, None], axis=0) / np.sum(masses)
        shape_dist = rdShapeHelpers.ShapeProtrudeDist(query_frag, sub_mol)
    
        del sub_mol
        # only accept if the distance disimilarity is below threshold
        if shape_dist < distance_threshold:
            return mol_idx
    return None
    

class FragmentScorer():
    def __init__(self, fragments, weights=None, use_features=False, global_substructure_match=True, 
            similarity_metric=DataStructs.DiceSimilarity) -> None:
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
        try:
            fragments = list(BRICSDecompose(mol, keepNonLeafNodes=self.global_substructure_match, returnMols=True))
            mol_fragment_fps = [AllChem.GetMorganFingerprintAsBitVect(frag, 2, nBits=1024, 
                useFeatures=self.use_features) for frag in fragments]
        except RuntimeError:
            return 0
        fragment_scores = [max([self.similarity(frag_fp, mol_frag_fp) for mol_frag_fp in mol_fragment_fps])
                                     for frag_fp in self.fragment_fps]

        final_score = np.dot(fragment_scores, self.weights)
        return final_score
        
        
class PharmacophoreScorer():
    def __init__(self, query, factory=None) -> None:
        if isinstance(query, str):
            assert os.path.exists(query), "Reads a path to file storing the molecules under column smiles, or a list of smiles"
            # assumes a path to molecules with header smiles
            df = pd.read_csv(query)
            query = df.smiles.values
        elif not isinstance(query, list):
            raise TypeError("query molecules should be a path to file storing the molecules, or a list of smiles")
        self.factory = factory
        if factory is None:
            self.factory = Gobbi_Pharm2D.factory
        self.query_pharm = [Generate.Gen2DFingerprint(Chem.MolFromSmiles(
                                    MurckoScaffold.MurckoScaffoldSmilesFromSmiles(q)
                                    ), self.factory) for q in query]
    
    def calc_score(self, smiles):
        p = Generate.Gen2DFingerprint(Chem.MolFromSmiles(smiles), self.factory)
        return max([DataStructs.TanimotoSimilarity(q, p) for q in self.query_pharm])
        

if __name__ == '__main__':
    import time
    start_time = time.time()
    path = [f"/global/scratch/users/ozhang/covid/rdkit_vina/outputs/{n}_out.sdf" for n in range(3)]
    for n in range(3):
        print(n, calc_fragment_position(path[n], "Cc2cc(N)c1cccc(Cl)c1n2", [-13.79580109,  15.63052044, -17.64702725]))
    print(time.time() - start_time)
   
