'''
Author: Jie Li
Date created: Oct 29, 2020
'''

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

kldiv_loss = torch.nn.KLDivLoss()

def compute_losses(predicted_prob, target, advantage, old_prob, prior_probs, clipping_epsilon):
    target_pred_prob = torch.sum(predicted_prob * target, dim=-1)
    target_old_pred = torch.sum(old_prob * target, dim=-1)
    ratio = target_pred_prob / (target_old_pred + 1e-8)
    clipped_ratio = torch.clamp(ratio, 1 - clipping_epsilon, 1 + clipping_epsilon)
    ppo_target = torch.min(ratio * advantage, clipped_ratio * advantage)
    dist = Categorical(predicted_prob)
    entropy = dist.entropy()
    log_pred_prob = torch.log(predicted_prob + 1e-8)
    step_kl_div = kldiv_loss(log_pred_prob, old_prob)
    prior_kl_div = kldiv_loss(log_pred_prob, prior_probs)
    return {"ppo_target": torch.mean(ppo_target),
            "entropy": torch.mean(entropy),
            "step_kl_div": step_kl_div,
            "prior_kl_div": prior_kl_div}
            

class EntropyScheduler:
    def __init__(self, entropy_bound = (0.2, 1.5), init_coeff=0.1, learning_rate=0.1):
        """
        Initializes the Entropy Scheduler.
        
        Args:
        - entropy_bound (Tuple): The lower/upper bound of the desired entropy value.
        - initial_coefficient (float): Initial value for the entropy coefficient (a ratio of the entropy term to ppo target)
        """
        self.init_coeff = init_coeff
        self.coeff = 0
        assert len(entropy_bound) > 0, "target entropy should be a tuple with at least lower bound specified"
        self.bound = entropy_bound
        self.initiated = False
        self.ln = learning_rate
        
    def step(self, current_entropy, ppo_target):
        """
        Adjusts the entropy coefficient based on the difference between current and entropy bound.
        
        Args:
        - current_entropy (torch.Tensor): entropy of sampled distribution from the policy network.
        - ppo_target: set initial coeff such that the entropy term is in the given ratio (initial entropy coefficient) to the ppo term
        """
        if ppo_target < 0:
            return 0
        if (len(self.bound) > 1 and current_entropy > self.bound[1]):
            return 0
        if abs(ppo_target) / current_entropy < self.init_coeff / 5:
            return 0
            
        if not self.initiated:
            self.coeff = self.init_coeff * abs(ppo_target) / current_entropy
            self.initiated = True
            print("init entropy coeff", self.coeff)
            return self.coeff
            
        if current_entropy <= self.bound[0]:
            self.coeff *= (self.ln + 1)
            
        return self.coeff
            

