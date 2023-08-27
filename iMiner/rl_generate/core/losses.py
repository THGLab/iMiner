'''
Author: Jie Li
Date created: Oct 29, 2020
'''

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

kldiv_loss = torch.nn.KLDivLoss()

def compute_losses(predicted_prob, target, advantage, old_prob, clipping_epsilon):
    target_pred_prob = torch.sum(predicted_prob * target, dim=-1)
    target_old_pred = torch.sum(old_prob * target, dim=-1)
    ratio = target_pred_prob / (target_old_pred + 1e-8)
    clipped_ratio = torch.clamp(ratio, 1 - clipping_epsilon, 1 + clipping_epsilon)
    ppo_target = torch.min(ratio * advantage, clipped_ratio * advantage)
    dist = Categorical(predicted_prob)
    entropy = dist.entropy()
    kl_div = -kldiv_loss(old_prob, predicted_prob)
    return {"ppo_target": torch.mean(ppo_target),
            "entropy": torch.mean(entropy),
            "kl_div": kl_div}
            

class EntropyScheduler:
    def __init__(self, schedule="linear", target_entropy = (0.5, 1.5), init_coeff=0.1, learning_rate=0.02):
        """
        Initializes the Entropy Scheduler.
        
        Args:
        - schedule (str) : "constant" or "linear" adaptive based on target range
        - target_entropy (Tuple): The lower/upper bound of the desired entropy value.
        - initial_coefficient (float): Initial value for the entropy coefficient.
        - learning_rate (float): Learning rate for adjusting the entropy coefficient.
        """
        self.schedule_type = schedule
        self.coeff = init_coeff
        self.learning_rate = learning_rate
        if schedule == "linear":
            assert len(target_entropy) > 0, "target entropy should be a tuple with at least lower bound specified"
            self.target_entropy = target_entropy
        

    def step(self, current_entropy):
        """
        Adjusts the entropy coefficient based on the difference between current and target entropy.
        
        Args:
        - current_entropy (torch.Tensor): entropy of sampled distribution from the policy network.
        """
        if self.schedule_type == "constant":
            return self.coeff
            
        if current_entropy <= self.target_entropy[0]:
            self.coeff += self.learning_rate * (self.target_entropy[0] - current_entropy)
        elif len(self.target_entropy) > 1 and current_entropy > self.target_entropy[1]:
            self.coeff += self.learning_rate * (self.target_entropy[1] - current_entropy)
            
        return self.coeff

