'''
Author: Jie Li
Date created: Oct 22, 2020
'''

from numpy.core.defchararray import array
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed
import numpy as np
from scipy.stats import gmean
#import selfies as sf
from group_selfies import GroupGrammar
import pandas as pd


def convert_input_to_selfies(input, tokens):
    '''
    A helper function to convert the prediction of the network into a SELFIES string
    '''
    token_list = [tokens[single_token_idx] for single_token_idx in input[1:-1]]
    if len(token_list) == 0 or "^" in token_list[0]:
        return ""
    # combined_tokens = []
    # for token in token_list:
    #     if token[0] != "^":
    #         combined_tokens.append(token)
    #     else:
    #         combined_tokens[-1] += token[1:]
    return_tokens = ["[%s]" % t for t in token_list]
    return "".join(return_tokens)

def safe_decode_selfies(selfies, grammar):
    '''
    A helper function to decode SELFIES string into SMILES, return empty string if decoding fails
    '''
    try:
        #sf.decoder(selfies)
        smiles = Chem.MolToSmiles(grammar.decoder(selfies))
    except:
        smiles = ""
    return smiles

class RewardAssigner():
    '''
    reward_types: list from ["qed", "drug_likeliness", "dock_score", "fragment_similarity"]
    reward_combination_method: one of {"sum", "arithmetic mean", "geometric mean"}
    '''
    def __init__(self, reward_combination_method="sum", tokens=None, logger=None, output_path=None, grammar_file=None) -> None:
        if grammar_file is None:
            raise RuntimeError("Need to define grammar to use group selfies")
        self.tokens = tokens
        self.grammar = GroupGrammar.from_file(grammar_file)
        self.reward_conversion_funcs = []
        self.property_calculators = {}
        self.reward_types = []
        if reward_combination_method == "sum":
            self.reward_combination = sum
        elif reward_combination_method == "arithmetic mean":
            self.reward_combination = np.mean
        elif reward_combination_method == "geometric mean":
            self.reward_combination = gmean
        self.logger = logger
        self.output_path = output_path
        self.iteration = 0

    def add_reward(self, reward_type, weight=1., extra_params=None):
        self.reward_types.append(reward_type)
        if reward_type == "qed":
            self.reward_conversion_funcs.append(lambda x: x)

        elif reward_type == "drug_likeliness":
            from iMiner.rl_generate.evaluators.drug_likeliness import DrugLikeliness
            self.reward_conversion_funcs.append(lambda x: max(x, 0)*weight)
            self.property_calculators[reward_type] = DrugLikeliness()
        
        elif reward_type == "solubility":
            from iMiner.rl_generate.evaluators.drug_likeliness import ESOLCalculator
            self.reward_conversion_funcs.append(lambda x: min(6+x, 3)*weight)
            self.property_calculators[reward_type] = ESOLCalculator()

        elif reward_type == "lead_likeness":
            from iMiner.rl_generate.evaluators.drug_likeness import LeadLikeliness
            self.reward_conversion_funcs.append(lambda x: max(x, 0))
            self.property_calculators[reward_type] = LeadLikeliness()

        elif reward_type == "docking":
            from iMiner.rl_generate.evaluators.docking_local import docking_score_assigner
            self.reward_conversion_funcs.append(lambda x: max(-x, 0)*weight)
            self.property_calculators[reward_type] = vina_score_assigner(path=self.output_path, **extra_params)

        elif reward_type == "fragment_similarity":
            from iMiner.rl_generate.evaluators.fragment_similarity import FragmentScorer
            self.reward_conversion_funcs.append(lambda x: x*weight)
            self.property_calculators[reward_type] = FragmentScorer(**extra_params)
            
        elif reward_type == "interaction":
            from iMiner.rl_generate.evaluators.interaction import InteractionScorer
            self.reward_conversion_funcs.append(lambda x: x*weight)
            self.property_calculators[reward_type] = InteractionScorer(**extra_params)
        else:
            raise RuntimeError("Unknown reward type: %s" % reward_type)
            
    def _reorder_reward_types(self):
        if "interaction" in self.reward_types:
            self.reward_names = [r for r in self.reward_types]
            self.reward_names.remove("interaction")
            vi = self.reward_names.index("vina_score")
            self.reward_names.insert(vi, "interaction")
        else:
            self.reward_names = self.reward_types

    def calc_reward_parallel(self, inputs):
        '''
        Calculate reward from the given list of inputs, do parallel assignment of scores, and return rewards together with whether each generated smiles string should contribute to training
        '''
        self.iteration += 1
        converted_selfies = [convert_input_to_selfies(item, self.tokens) for item in inputs]
        converted_smiles = [safe_decode_selfies(s, self.grammar) for s in converted_selfies]
        query_indices = [] # keep record of whether each element from the converted smiles should receive reward query. If not, then these are bad smiles and should receive a very low reward
        plot_mols = []
        for i in range(len(converted_smiles)):
            if converted_smiles[i] is None or converted_smiles[i] == "":
                query_indices.append(False)
                continue
            mol = Chem.MolFromSmiles(converted_smiles[i])
            if mol is None:
                query_indices.append(False)
                continue
            plot_mols.append(mol)
            query_indices.append(True)
        query_indices = np.array(query_indices)
        assert len(query_indices) == len(converted_smiles), "smiles: %s, query_indices: %s" % (str(converted_smiles), str(query_indices))

        valid_smiles = [smile for smile,validity in zip(converted_smiles,query_indices) if validity]

        metrics, validities, new_names = self.get_metric_values_parallel(valid_smiles)
        if np.sum(validities) == 0:
            return None
        # valid metrics are those received feedback (including NaN results) from the query within time limit
        valid_metrics = np.array(metrics)[:, validities].T.astype(float)

        non_nan_filter = ~np.isnan(np.sum(valid_metrics, axis=1))

        df = pd.DataFrame()
        non_nan_smiles = [s for s,v in zip(valid_smiles, validities) if v]
        df["smiles"] = non_nan_smiles
        for i in range(valid_metrics.shape[1]):
            df[self.reward_types[i]] = valid_metrics[:, i]
        df["names"] = [s for s,v in zip(new_names, validities) if v] 
        

        mean_valid_metrics = np.mean(valid_metrics[non_nan_filter], axis=0)
        converted_rewards = [[reward_conversion_func(num) for reward_conversion_func, num in zip(self.reward_conversion_funcs, metric_values)] for metric_values in valid_metrics]
        final_rewards = np.array([self.reward_combination(converted_reward) for converted_reward in converted_rewards])
        mean_valid_reward = np.mean(final_rewards[non_nan_filter])
        all_rewards = np.zeros(len(inputs))
        query_rewards = np.zeros(np.sum(query_indices))
        all_validities = np.ones(len(inputs)).astype(bool)

        query_rewards[validities] = final_rewards
        all_rewards[query_indices] = query_rewards
        all_rewards[~query_indices] = -10
        all_validities[query_indices] = validities
        all_rewards[np.isnan(all_rewards)] = -10
        return all_rewards, all_validities, [mean_valid_reward] + list(mean_valid_metrics), df

    def calc_reward(self, input):
        '''
        Calculate reward from given input, and return a single comprehensive reward score and individual metric values
        '''
        converted_selfies = convert_input_to_selfies(input, self.tokens)
        converted_smiles = Chem.MolToSmiles(self.grammar.decoder(converted_selfies)) #sf.decoder(converted_selfies)
        if converted_smiles is None or converted_smiles == "":
            return -10

        mol = Chem.MolFromSmiles(converted_smiles)


        if mol is None:
            return -10
        try:
            metric_values = self.get_metric_values(mol)
        except:
            return -10

        converted_rewards = [reward_conversion_func(num) for reward_conversion_func, num in zip(self.reward_conversion_funcs, metric_values)]
        final_reward = self.reward_combination(converted_rewards)
        return [final_reward, metric_values]
        
    def get_metric_values(self, mol):
        metrics = []
        for reward_item in self.reward_types:
            if reward_item == "qed":
                metrics.append(qed(mol))
            if reward_item in ["drug_likeliness", "lead_likeliness", "fragment_similarity", "solubility"]:
                metrics.append(self.property_calculators[reward_item].calc_score(mol))
        return metrics

    def get_metric_values_parallel(self, mols):
        metrics = []
        new_names = [str(self.iteration) + "_" + str(i) for i in range(len(mols))]
        for reward_item in self.reward_types:
            if reward_item in ["drug_likeliness", "lead_likeliness", "fragment_similarity", "solubility"]:
                metrics.append([self.property_calculators[reward_item].calc_score(mol) for mol in mols])
            if reward_item == "docking":
                dock_scores = self.property_calculators[reward_item].get_scores(mols, new_names, self.iteration)
                validities = ~np.isnan(dock_scores) 
                metrics.append(dock_scores)
                #print("iter %s finished docking calculation"%(self.iteration))
                if "interaction" in self.reward_types:
                    docking_df = self.property_calculators[reward_item].get_result_df(self.iteration)
                    # TODO: interaction results ordered by df; uses vina docking pose
                    interaction_result = self.property_calculators["interaction"].calc_score_parallel(docking_df["vina_path"])
                    # align & reorder scores
                    interaction_scores = self.property_calculators[reward_item].update_interaction(self.iteration, 
                        new_names, interaction_result)
                    metrics.append(interaction_scores)
                    #print("iter %s finished interaction calculation"%(self.iteration))
                    continue
            if reward_item == "fcd":
                chemnet_dist = self.property_calculators[reward_item].get_chemnet_dist(mols)
                metrics.append(chemnet_dist)
            if reward_item == "mfd":
                morgan_dist = self.property_calculators[reward_item].get_mfd(mols)
                metrics.append(morgan_dist)
        if "docking" not in self.reward_types:
            validities = [True] * len(metrics[0])
        #print("iter %s finished all reward evaluations"%(self.iteration))
        return metrics, validities, new_names
