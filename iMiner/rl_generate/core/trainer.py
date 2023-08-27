'''
Author: Jie Li
Date created: Oct 22, 2020
'''

from numpy.random import RandomState
import numpy as np
import torch
from torch.nn.utils.rnn import *
from iMiner.rl_generate.rl_utils import create_chunks, to_numpy
from iMiner.rl_generate.core.losses import compute_losses, EntropyScheduler
import time


class Trainer():
    def __init__(self, policy_model, prior_model, rewards, config, optimizer, logger, show_progress=True, output_freq=1) -> None:
        '''
        policy_model = a model that acts as the agent for generating sequences that maximizes the reward while follows the prior distribution
        prior_model = a model that provides the prior distribution
        rewards = the reward value types for considering in rl
        prior_diff_coef = coefficient that is multiplied to the prior probability difference and subtracted from the total reward. When this is zero, then there will not be prior difference considered during training (float)
        rl_params = a dictionary specifying the parameters for running reinforcement learning training, including the algorithm to use, and hyperparameters for the algorithms (dict of {"algorithm": x, "epsilon": x, "entropy_coef": x})
        n_seq_per_iteration = the number of sequences to collect and calculate gradient for in each model iteration (int)
        batch_size = batch size used for training the model at each step (int)
        optimizer_specs = a dictionary describing the specifications for the optimizer used for training the model (dict)
        logger = a logger object that logs metrics (rl_utils.Logger)
        random_seed = the random seed used for fixing random behavior, which ensures the reproducibility of the program (float)
        show_progress = whether or not show a progress bar to indicate the process of iterations that take long time (bool)
        output_freq = decides the frequency for outputting metrics and losses to the screen. Log file will always be updated at each iteration (int)
        '''
        self.policy_model = policy_model
        self.prior_model = prior_model
        self.algorithm = config["rl_params"]["algorithm"]
        self.reward_assiner = rewards
        self.reward_names = ["mean_reward"] + rewards.reward_types + ["prior_prob_diff"]
        self.prior_diff_coef = config["training_specs"]["prior_diff_coef"]
        self.rl_params = config["rl_params"]
        self.n_seq_per_iteration = config["training_specs"]["n_seq_per_iter"]
        self.batch_size = config["training_specs"]["batch_size"]
        self.optimizer = optimizer
        self.logger = logger
        self.output_dir = config["run"]["output_dir"]
        self.random_state = RandomState(config["run"]["random_seed"])
        self.show_progress = show_progress
        self.output_freq = output_freq
        self.entropy_scheduler = EntropyScheduler(**config["rl_params"]["entropy"])

        # Initialize logger title
        self.logger.initialize(["iteration", "loss", "ppo_target", "entropy"] + self.reward_names + ["time_elapsed"])


    def training_loop(self, n_iters, save_each_iteration, start_iteration=0):
        '''
        Run training of RL for n iterations

        args:
            n_iters = total number of iterations for training
            save_each_iteration = whether or not save the policy model at each iteration
        '''
        time_start = time.time()
        for n in range(start_iteration, n_iters):
            print("=" * 20 + "  Iteration %d  " % (n + 1) + "=" * 20)
            iter_result = self.run_single_iteration(time_start, n + 1)
            iter_log_contents = {"loss": iter_result["mean_loss"],
                                 "ppo_target": iter_result["mean_ppo_target"],
                                 "entropy": iter_result["mean_entropy"],
                                 "time_elapsed": iter_result["time_elapsed"]}
            for metric, metric_value in zip(self.reward_names, iter_result["individual_metrics"]):
                iter_log_contents[metric] = metric_value
            self.logger.log(n + 1, iter_log_contents)
            if save_each_iteration:
                self.policy_model.save_model(self.output_dir, f"model_{n + 1}")

    def run_single_iteration(self, time_start, current_iter):
        print("Collecting trajectories...")
        trajs, individual_metrics, df = self.collect_trajs(self.n_seq_per_iteration)
        # save generated molecule vina scores and drug likelihood scores
        df.to_csv(f"{self.output_dir}/details/{current_iter}.csv", index=None)
        print("Training model with collected data...")
        rewards = trajs["rewards"]
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        standardized_advantage = (rewards - rewards_mean) / rewards_std

        if np.isnan(standardized_advantage).any():
            print("NaN occurred in standardized advantage!")
            import pickle
            with open(f"{self.output_dir}/trajs.pkl", "wb") as f:
                pickle.dump(trajs, f)

        total_timesteps_this_iter = len(trajs["observations"])
        batch_indices = list(range(total_timesteps_this_iter))
        for step in range(self.rl_params["n_epochs_before_resample"]):
            self.random_state.shuffle(batch_indices)
            idx_chunks = create_chunks(batch_indices, self.batch_size)
            kl_div_record = []
            loss_record = []
            ppo_target_record = []
            entropy_record = []
            for chunk in idx_chunks:
                # batch_lens = torch.tensor([len(trajs["observations"][idx]) for idx in chunk])
                batch_observation = [trajs["observations"][idx] for idx in chunk]
                batch_prediction = self.policy_model.get_prediction(batch_observation)
                batch_action = torch.nn.functional.one_hot(torch.tensor(trajs["actions"][chunk], device=batch_prediction.device), num_classes=batch_prediction.shape[1])
                advantage = torch.tensor(standardized_advantage[chunk], device=batch_prediction.device)
                old_probs = torch.tensor(trajs["probabilities"][chunk], device=batch_prediction.device)
                all_loss_terms = compute_losses(batch_prediction, batch_action, advantage, old_probs, self.rl_params["epsilon"])
                entropy_coeff = self.entropy_scheduler.step(all_loss_terms["entropy"].item())
                loss = -(all_loss_terms["ppo_target"] * 10 + entropy_coeff * all_loss_terms["entropy"])
                # For debug: record old parameters
                state_dict = self.policy_model.model.state_dict().copy()
                # Do gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # For debug: check new parameters
                new_state_dict = self.policy_model.model.state_dict()
                new_params = torch.cat([item.flatten() for item in new_state_dict.values()])
                if torch.any(torch.isnan(new_params)).item():
                    # NaN found!
                    torch.save(state_dict, f"{self.output_dir}/last_good_model.sav")
                    torch.save({
                        "batch_observation": batch_observation,
                        "batch_prediction": batch_prediction,
                        "batch_action": batch_action,
                        "advantage": advantage,
                        "old_probs": old_probs,
                        "all_loss_terms": all_loss_terms
                    }, f"{self.output_dir}/data.sav")
                    exit()
                kl_div_record.append(all_loss_terms["kl_div"].item())
                loss_record.append(loss.item())
                ppo_target_record.append(all_loss_terms["ppo_target"].item())
                entropy_record.append(all_loss_terms["entropy"].item())
            mean_kl_div = np.mean(kl_div_record)
            mean_loss = np.mean(loss_record)
            mean_ppo_target = np.mean(ppo_target_record)
            mean_entropy = np.mean(entropy_record)
            if mean_kl_div > self.rl_params["kl_threshold"]:
                print(f"Early stopping at step {step} due to KL-divergence exceeding threshold")
                break
        iteration_result = {"individual_metrics": individual_metrics,
                            "mean_loss": mean_loss,
                            "mean_ppo_target": mean_ppo_target,
                            "mean_entropy": mean_entropy,
                            "time_elapsed": (time.time() - time_start) / 3600}
        return iteration_result



    def collect_trajs(self, n_rollouts, maximum_retry=5):
        '''
        Returns a dictionary recording all the data needed for training the algorithm, and the mean of the individual metrics from the current trajectory collection
        '''
        retry_count = 0
        while True:
            retry_count += 1
            observations = []
            actions = []
            rewards = []
            length_records = []
            sequences = []
            individual_metrics = []
            probabilities = []
            prob_diffs = []
            for _ in range(n_rollouts):
                final_seq, selected_action_probs, all_probs = self.policy_model.sample_seq()
                for i in range(len(final_seq) - 1):
                    observations.append(final_seq[:i + 1])
                    actions.append(final_seq[i + 1].item())
                probabilities.extend(all_probs)
                selected_prob = selected_action_probs.numpy()
                prior_prob = np.array(self.prior_model.get_seq_log_prob(final_seq, return_sum=False))
                prob_diff = np.abs(selected_prob-prior_prob).mean() # This probability difference might be > 0 due to random dropout resulting non-determinate selected_prob
                prob_diffs.append(prob_diff)
                final_seq = to_numpy(final_seq)

                length_records.append(len(final_seq))
                sequences.append(final_seq)
            print("Querying reward...")
            collected_results = self.reward_assiner.calc_reward_parallel(sequences)
            if collected_results is None:
                if retry_count <= maximum_retry:
                    print("Failed collecting valid data on trial %d. Retrying..." % retry_count)
                    continue
                else:
                    raise RuntimeError("Cannot collect valid data within %d retries" % maximum_retry)
            else:
                all_rewards, all_validities, mean_valid_rewards, df = collected_results
                break
        print("Reward collection finished")
        rewards = []
        for reward_value, length in zip(all_rewards[all_validities], np.array(length_records)[all_validities]):
            rewards += [reward_value] * (length - 1)
        index_selector = np.concatenate([[validity] * (length - 1) for validity, length in zip(all_validities, length_records)])
        individual_metrics = mean_valid_rewards + [np.mean(prob_diffs)]
        return {"observations": [item for item, selected in zip(observations, index_selector) if selected],
                # "observations": np.array(observations, dtype=object)[index_selector],
                "actions": np.array(actions)[index_selector],
                "rewards": np.array(rewards),
                "probabilities": to_numpy(torch.cat(probabilities))[index_selector]}, individual_metrics, df
                
