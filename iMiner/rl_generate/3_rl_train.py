'''
Author: Jie Li
Date created: Oct 29, 2020
'''

#####################
# Parameters for training control
#####################
import yaml
with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

#####################
# Fixing random number seeds
#####################
SEED = config["run"]["random_seed"]
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import torch
torch.manual_seed(SEED)

#####################
# Other imports
#####################
from iMiner.rl_generate.core.trainer import Trainer
from iMiner.rl_generate.core.model import Model
from iMiner.rl_generate.core.reward import RewardAssigner
import os
import argparse

import warnings
warnings.filterwarnings("ignore", "reduction: 'mean' divides the total loss by both the batch size and the support size.")

from iMiner.rl_generate.rl_utils import Logger, make_optimizer



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--start_from", default=None)
parser.add_argument("--start_iter", default=0, type=int)
parser.add_argument("--output_dir", default="outputs", type=str)
args = parser.parse_args()
rl_dataset_path = config["training_specs"]["dataset_path"]
if rl_dataset_path[-1] != "/":
    rl_dataset_path += "/"


#####################
# Prepare output dirs and logger
#####################
output_directory = config["run"]["output_dir"]
if output_directory[-1] == "/":
    output_directory = output_directory[:-1] # remove trailing slash
if os.path.exists(output_directory):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_directory += "_" + timestamp

os.makedirs(output_directory)
os.makedirs(output_directory + "/docking")
os.makedirs(output_directory + "/details")
print("All results logged in", output_directory)
logger = Logger(output_directory)
config["run"]["output_dir"] = output_directory


#####################
# Load pretrained model
#####################                   

prior_model = Model(config["training_specs"]["prior_model"])
prior_model.set_as_prior()
policy_model = Model(config["training_specs"]["starting_policy_model"])
if "model_weights" in config["training_specs"]:
    policy_model.load_model_chk(config["training_specs"]["model_weights"])


#####################
# Prepare rewards
#####################
rewards = RewardAssigner(reward_combination_method="sum", tokens=prior_model.tokens,
     logger=logger, output_path=output_directory + "/docking")
for item in config["rewards"]:
    if type(item) is str:
        rewards.add_reward(item)
    elif type(item) is dict:
        rewards.add_reward(item["type"], item["params"])

#####################
# Set up optimizer, trainer and start training
#####################
optimizer = make_optimizer(config["optimizer_specs"], policy_model.get_trainable_parameters())
trainer = Trainer(policy_model, prior_model, rewards, config, optimizer, logger)
trainer.training_loop(config["training_specs"]["n_iters"], save_each_iteration=True)


#####################
# Save trained model
#####################
policy_model.save_model(output_directory, "final_model")
