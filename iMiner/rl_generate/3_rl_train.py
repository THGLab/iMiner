'''
Author: Jie Li
Date created: Oct 29, 2020
'''

#####################
# Parsing arguments
#####################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--start_from", default=None)
parser.add_argument("--start_iter", default=0, type=int)
parser.add_argument("--output_dir", default="outputs", type=str)
parser.add_argument("--config", default="config.yml", type=str)
args = parser.parse_args()


#####################
# Parameters for training control
#####################
import yaml
with open(args.config, "r") as f:
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

import warnings
warnings.filterwarnings("ignore", "reduction: 'mean' divides the total loss by both the batch size and the support size.")

from iMiner.rl_generate.rl_utils import Logger, make_optimizer



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
#os.makedirs(output_directory + "/models")
print("All results logged in", output_directory)
logger = Logger(output_directory)
config["run"]["output_dir"] = output_directory
start_iter = 0
if "start_iter" in config["training_specs"]:
    start_iter = config["training_specs"]["start_iter"]

#####################
# Load pretrained model
#####################                   

prior_model = Model(config["training_specs"]["prior_model"])
prior_model.set_as_prior()
policy_model = Model(config["training_specs"]["starting_policy_model"])
if "model_weights" in config["training_specs"]:
    prior_weights = config["training_specs"]["model_weights"]
    if prior_weights.endswith(".pth"):
        prior_weights = prior_weights[:-4]
    policy_model.load_model_chk(prior_weights)


#####################
# Prepare rewards
#####################
rewards = RewardAssigner(reward_combination_method="sum", tokens=prior_model.tokens,
     grammar_file=config["training_specs"]["grammar"],
     logger=logger, output_path=output_directory + "/docking")
for item in config["rewards"]:
    if type(item) is str:
        rewards.add_reward(item)
    elif type(item) is dict:
        if "weight" in item and "params" in item:
            rewards.add_reward(item["type"], weight=item["weight"], extra_params=item["params"])
        elif "weight" in item:
            rewards.add_reward(item["type"], extra_params=item["weight"])
        elif "params" in item:
            rewards.add_reward(item["type"], extra_params=item["params"])
        else:
            rewards.add_reward(item["type"])

#####################
# Set up optimizer, trainer and start training
#####################
optimizer = make_optimizer(config["optimizer_specs"], policy_model.get_trainable_parameters())
trainer = Trainer(policy_model, prior_model, rewards, config, optimizer, logger)
trainer.training_loop(config["training_specs"]["n_iters"], save_each_iteration=True, start_iteration=start_iter)


#####################
# Save trained model
#####################
policy_model.save_model(output_directory, "final_model")
