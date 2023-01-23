import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.BRICS import *

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