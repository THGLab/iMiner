from fastai import *
from fastai.text import *

import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import numpy as np
import selfies as sf

defaults.text_spec_tok = [BOS, PAD]

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
        self.tokens = ["#Branch1",	"#Branch2",	"#C",	"#N",	"#N+1",	"-/Ring2",	"/Br",	"/C",
        	"/C@",	"/C@@",	"/C@@H1",	"/C@H1",	"/Cl",	"/N",	"/N+1",	"/O",	"/S",
            	"=Branch1",	"=Branch2",	"=C",	"=N",	"=N+1",	"=N-1",	"=O",	"=P",	"=Ring1",
                	"=Ring2",	"=S",	"=Se",	"B",	"B-1",	"Br",	"Branch1",	"Branch2",	"C",
        	"C-1",	"C@",	"C@@",	"C@@H1",	"C@H1",	"Cl",	"F",	"I",	"N",	"N+1",	"N-1",
        	"NH1",	"O",	"O-1",	"OH0",	"P",	"P+1",	"P@",	"P@@",	"PH1",	"Ring1",	"Ring2",
        	"S",	"S+1",	"Se",	"Si",	"Te",	"\\C",	"\\C@@H1",	"\\C@H1",	"\\Cl",	"\\N",
            	"\\N+1",	"\\NH1",	"\\O",	"\\O-1",	"\\S"]

    def tokenizer(self, selfies: str) -> List[str]:
        selfies_tokens = selfies[1:-1].split("][")
        if np.any([tk not in self.tokens for tk in selfies_tokens]):
            return [BOS] # if any very rara token occurs in the SELFIES string, discard the sequence (should be rare)
        else:
            return [BOS] + selfies_tokens

class SELFIESCompressedTokenizer(BaseTokenizer):
    def __init__(self, lang):
        self.pre_modifier = ["#", "/", "\\", "=", "-/", "-\\"]
        self.tokens = ["Branch1", "Branch2", "As", "B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "Ring1", "Ring2", "S", "Se", "Si", "Te"]
        self.post_modifier = ["+1", "-1", "@", "@@", "@H1", "@@H1", "H1"]
        pass
    def tokenizer(self, selfies):
        orig_selfies_tokens = selfies[1:-1].split("][")
        new_tokens = []
        for tk in orig_selfies_tokens:
            pre_modifier = ""
            post_modifier = ""
            for pre_mod in self.pre_modifier:
                if tk.startswith(pre_mod):
                    pre_modifier = pre_mod
                    tk = tk[len(pre_mod):]
                    break
            for post_mod in self.post_modifier:
                if tk.endswith(post_mod):
                    post_modifier = post_mod
                    tk = tk[:-len(post_mod)]
                    break
            if tk in self.tokens:
                if pre_modifier != "":
                    new_tokens.append(pre_modifier + "^")
                new_tokens.append(tk)
                if post_modifier != "":
                    new_tokens.append("^" + post_modifier)
            else:
                # this is a token that cannot be broken (rara case), just ignore the whole sequence
                return [BOS]

        tokens = [BOS] + new_tokens
        return tokens    
    
    def add_special_cases(self, toks):
        pass

class Mol2SELFIESTokenizer(BaseTokenizer):
    def __init__(self, lang):
        self.tokens = ['#C', '#N', '#O', '#S', '=B', '=C', '=I', '=N', '=O', '=P', '=S', '=Se', '=Si', 'B', 'Br', 'Br+2', 'Branch1_1', 'Branch1_2', 'Branch1_3', 'Branch2_1', 'Branch2_2', 'Branch2_3', 'C', 'Cl', 'Cl+2', 'Cl+3', 'Expl=Ring1', 'Expl=Ring2', 'F', 'I', 'I+2', 'I+3', 'N', 'O', 'P', 'Ring1', 'Ring2', 'S', 'Se', 'Si']
        self.expl_tokens = ["H+expl", "H2+expl","H3+expl","+expl","Hexpl","H2expl","H-expl","H2-expl","H3-expl","-expl","expl"]
        pass
    def tokenizer(self, smiles):
        selfies = sf.encoder(smiles)
        if selfies is None:
            return [BOS]
        orig_selfies_tokens = selfies[1:-1].split("][")
        new_tokens = []
        for tk in orig_selfies_tokens:
            if not "expl" in tk or tk == "Hexpl":
                new_tokens.append(tk)
            else:
                for expl_tk in self.expl_tokens:
                    if expl_tk in tk:
                        new_tokens.append(tk.replace(expl_tk, ""))
                        new_tokens.append("^" + expl_tk)
                        break
        tokens = [BOS] + new_tokens
        return tokens    
    
    def add_special_cases(self, toks):
        pass
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