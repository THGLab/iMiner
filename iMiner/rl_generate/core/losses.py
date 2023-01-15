'''
Author: Jie Li
Date created: Oct 29, 2020
'''

import torch
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