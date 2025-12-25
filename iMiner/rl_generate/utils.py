#from fastai import *
from typing import List
#import torch.nn.functional as F
#import torch
#from torch.distributions import Categorical
import numpy as np
import subprocess
#from multiprocessing import Pool

import selfies as sf
from rdkit import Chem

defaults.text_spec_tok = [BOS, PAD]

def split_selfies(selfies: str):

    left_idx = selfies.find("[")
    while 0 <= left_idx < len(selfies):
       
        right_idx = selfies.find("]", left_idx + 1)
        if right_idx == -1:
            raise ValueError("malformed SELFIES string, hanging '[' bracket")
            
        next_symbol = selfies[left_idx: right_idx + 1]
        yield next_symbol
        left_idx = right_idx + 1
        if selfies[left_idx: left_idx + 1] == ".":
            yield "."
            left_idx += 1
            
class MolTokenizer(BaseTokenizer):
    def __init__(self, lang):
        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        pass
    def tokenizer(self, smiles):
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        tokens = list(temp_smiles)
        tokens = [BOS] + tokens 
        return tokens    
    
    def add_special_cases(self, toks):
        pass

class SELFIESTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        self.tokens = { 
            "#Branch1", "#Branch2", "#C", "#N", "#N+1", "-/Ring2", "/Br", "/C", "/C@", "/C@@", 
            "/C@@H1", "/C@H1", "/Cl", "/N", "/N+1", "/O", "/S", "\\S",
            "=Branch1",	"=Branch2", "=C", "=N",	"=N+1",	"=N-1",	"=O", "=P", "=Ring1",
            "=Ring2", "=S", "Br", "Branch1", "Branch2", "C", "C-1", "C@", "C@@",	"C@@H1", 
            "C@H1", "Cl", "F", "I", "N", "N+1",	"N-1", "NH1", "O", "O-1", "OH0", "\\O", "\\O-1",
            "P", "P+1",	"P@", "P@@", "PH1", "Ring1", "Ring2", "S", "S+1", 
            "\\C", "\\C@@H1", "\\C@H1", "\\Cl", "\\N", "\\N+1", "\\NH1", "pop"}
    
    def encoder(self, smi):
        try:
            encoded = sf.encoder(smi)
        except ValueError:
            print(smi)
            return ""
        return encoded
    
    def tokenizer(self, selfies: str) -> List[str]:
        selfies_tokens = selfies[1:-1].split("][")
        return [BOS] + selfies_tokens


class ModelSampler():
    '''
    language_model = a fast.ai model that implements the forward() function
    tokenizer = a list record of all the tokens according to their index
    '''
    def __init__(self, language_model, tokenizer):
        self.model = language_model
        self.tokenizer = tokenizer

    def get_batch_action(self, batch_input):
        self.model.reset()
        pred = self.model(batch_input)[0][:, -1:]
        probs = F.softmax(pred, dim=-1)
        action_dist = Categorical(probs)
        choices = action_dist.sample()
        return choices

    def sample_one_batch(self, batch_input, maximal_len=100):
        sampled_contents = []
        current_input = batch_input
        for _ in range(maximal_len):
            next_ = self.get_batch_action(current_input)
            # Find out finished sequences (output 0)
            finished = []
            for i in range(len(next_)):
                finished.append(next_[i][0] == 0)
            finished = torch.tensor(finished)
            finished_seq = current_input[finished]
            for seq in finished_seq:
                sampled_contents.append([self.tokenizer[n] for n in seq[1:]])
            not_finished = torch.logical_not(finished)
            current_input = torch.cat([current_input[not_finished], next_[not_finished]], dim=-1)
            if not_finished.sum() == 0:
                break
        if len(current_input) > 0:
            for seq in current_input:
                sampled_contents.append([self.tokenizer[n] for n in seq[1:]])
        return sampled_contents


    def sample(self, count, maximal_len=100, do_batch=False, batch_size=64):
        if not do_batch:
            sampled_contents = []
            for i in range(count):
                current_input = torch.tensor([[0]], device="cuda")
                while True:
                    next_ = self.get_batch_action(current_input)
                    if next_[0][0] != 0 and len(current_input) < maximal_len:
                        current_input = torch.cat([current_input, next_], dim=-1)
                    else:
                        break
                converted_content = [self.tokenizer[n] for n in current_input[0][1:]]
                sampled_contents.append(converted_content)
        else:
            num_batchs = int(count / batch_size) + 1
            sampled_contents = []
            for i in range(num_batchs - 1):
                batch_input = torch.zeros((batch_size, 1), dtype=torch.long, device="cuda")
                sampled_contents.extend(self.sample_one_batch(batch_input, maximal_len))
            if count % batch_size != 0:
                batch_input = torch.zeros((count % batch_size, 1), dtype=torch.long, device="cuda")
                sampled_contents.extend(self.sample_one_batch(batch_input, maximal_len))
        return sampled_contents

class SMILES_Sampler(ModelSampler):
    def __init__(self, language_model, tokenizer):
        super(SMILES_Sampler, self).__init__(language_model, tokenizer)

    def sample(self, count, maximal_len=100, do_batch=False, batch_size=64):
        sampled_contents = super(SMILES_Sampler, self).sample(count, maximal_len, do_batch, batch_size)
        sampled_contents = ["".join(content) for content in sampled_contents]
        return sampled_contents

class SELFIES_Sampler(ModelSampler):
    def __init__(self, language_model, tokenizer):
        super(SELFIES_Sampler, self).__init__(language_model, tokenizer)

    def convert_tokens_to_SELFIES(self, tokens):
        if len(tokens) == 0 or "^" in tokens[0]:
            return ""
        combined_tokens = []
        partial_token = ""
        for token in tokens:
            if partial_token == "":
                if token[-1] == "^":
                    partial_token = token[:-1]
                elif token[0] == "^":
                    return ""   # invalid token
                else:
                    combined_tokens.append(token)
            else:
                if token[-1] == "^":
                    return ""  # unexpected situation that two consecutive pre-tokens together
                else:
                    combined_tokens.append(partial_token + token)
                    partial_token = ""
        tokens = ["[%s]" % t for t in combined_tokens]
        return "".join(tokens)

    def sample(self, count, maximal_len=100, do_batch=False, batch_size=64):
        sampled_contents = super(SELFIES_Sampler, self).sample(count, maximal_len, do_batch, batch_size)
        sampled_contents = [self.convert_tokens_to_SELFIES(tok) for tok in sampled_contents]
        return sampled_contents


def get_gpu_count():
    names = subprocess.Popen(["nvidia-smi", "--query-gpu=name", "--format=csv"], stdout=subprocess.PIPE)
    n_lines = subprocess.check_output(["wc", "-l"], stdin=names.stdout)
    names.stdout.close()
    return int(n_lines.decode("utf-8")) - 1
