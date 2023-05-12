#####################
# Fixing random number seeds
#####################
SEED = 42
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import torch
torch.manual_seed(SEED)

from iMiner.rl_generate.core.model import Model
from iMiner.rl_generate.core.reward import convert_input_to_selfies
from group_selfies import GroupGrammar
from rdkit import Chem

n_samples = 20
#####################
# Load pretrained model
#####################                   
grammar = GroupGrammar.from_file("/global/scratch/users/ozhang/covid/rl_dataset/frag_grammar.txt")

prior_model = Model("/global/scratch/users/ozhang/covid/rl_dataset/dahai_frag_pretrained.pkl")
prior_model.set_as_prior()
#policy_model = Model("/global/scratch/users/ozhang/covid/rl_dataset/chembl_model.pth")
print(prior_model.tokens)

for n in range(n_samples):
    final_seq, _, _ = prior_model.sample_seq()
    sf = convert_input_to_selfies(final_seq.numpy(), prior_model.tokens)
    decoded = grammar.decoder(sf)
    print(n+1, Chem.MolToSmiles(decoded))
