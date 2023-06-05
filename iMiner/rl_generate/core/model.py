'''
Author: Jie Li
Date created: Oct 22, 2020
'''

# from fastai import *
# from fastai.text import *
import os
from pathlib import Path
from fastai.text import load_learner
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import *


class Model():
    def __init__(self, model_pkl) -> None:
        model_path = os.path.dirname(model_pkl)
        model_filename = os.path.basename(model_pkl)
        self._language_model = load_learner(model_path, model_filename)
        self.model = self._language_model.model.train()
        self.tokens = self._language_model.data.train_ds.x.vocab.itos
        self.device = next(self.model.parameters()).device

    def load_model_chk(self, chk_name):
        chk_path = os.path.dirname(chk_name)
        chk_filename = os.path.basename(chk_name)
        if ".pth" in chk_filename:
            chk_filename = chk_filename.replace(".pth", "")
        self._language_model.path = Path(chk_path)
        self._language_model.model_dir = Path("./")
        self._language_model.load(chk_filename)

    def save_model(self, output_path, filename):
        self._language_model.path = Path(output_path)
        self._language_model.save(filename)

    def set_as_prior(self):
        '''
        Set the current model as prior model, which means it will be executed under evaluation mode, and no gradients will be back-propogated into the model
        '''
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_action(self, input):
        '''
        Sample an action using the current policy and return the sampled action and the predicted probability distribution
        '''
        with torch.no_grad():
            self.model.reset()
            if len(input.shape) != 2:
                input = input[None]
            pred = self.model(input)[0][:, -1]
            probs = F.softmax(pred, dim=-1)
            dist = Categorical(probs)
            sampled_action = dist.sample()
        return sampled_action, probs

    def get_action_log_prob(self, input_seq, action_taken):
        '''
        Obtain the log probability for the sampled action given specific sequence input.

        The return will be the log probability of the action, together with the entropy of the probability distribution
        '''
        if len(input_seq.shape) != 2:
            input_seq = input_seq[None]
        probs = self.get_prediction(input_seq)
        dist = Categorical(probs)
        entropy = dist.entropy()
        log_prob = dist.log_prob(action_taken)
        return log_prob, entropy

    def get_prediction(self, input):
        '''
        Feed the input into the network and obtain the direct output prediction. This version fully supports variant-length sequences, and only outputs the last hidden state

        args:
            input = a list of variant_length tensors
        '''
        self.model.reset()
        input_lens = [len(item) for item in input]
        awd_lstm_encoder = self.model[0]
        linear_decoder = self.model[1]
        if len(input_lens) > 1:
            input_padded = pad_sequence(input, batch_first=True)
            embedded = awd_lstm_encoder.encoder_dp(input_padded)
            raw_output = awd_lstm_encoder.input_dp(embedded)
            raw_outputs,outputs = [],[]
            for l, (rnn, hid_dp) in enumerate(zip(awd_lstm_encoder.rnns, awd_lstm_encoder.hidden_dps)):
                packed_seqs = pack_padded_sequence(raw_output, torch.tensor(input_lens), batch_first=True, enforce_sorted=False)
                packed_output, new_h = rnn(packed_seqs)
                raw_output, lengths = pad_packed_sequence(packed_output, batch_first=True)
                raw_outputs.append(raw_output)
                if  l != awd_lstm_encoder.n_layers - 1: raw_output = hid_dp(raw_output)
                outputs.append(raw_output)
            encoder_return = raw_outputs, outputs
            decoder_return = linear_decoder(encoder_return)
            pred = decoder_return[0][torch.arange(len(input_lens)), torch.tensor(input_lens) - 1]
        else:
            pred = self.model(input)[0][:, -1]
        probs = F.softmax(pred, dim=-1)
        return probs

    def sample_seq(self, maximal_length=500):
        '''
        Sample a single sequence using the model. Return the sampled sequence (as the actions taken), total log probability of the sequence, and individual timestep probabilities
        '''
        current_seq = torch.tensor([[0]], device=self.device)
        all_probs = []
        selected_action_prob = []
        for _ in range(maximal_length):
            next_action, prob = self.get_action(current_seq)
            current_seq = torch.cat([current_seq, next_action[None]], dim=-1)
            all_probs.append(prob)
            selected_action_prob.append(torch.log(prob[0][next_action[0]]).item())
            if next_action[0] == 0:
                break
        return current_seq.squeeze(), torch.tensor(selected_action_prob), all_probs

    def get_seq_log_prob(self, seq_onehot, return_sum=True):
        '''
        Calculate the log probability of a complete sequence (represented as onehot vector) under the current model

        args:
            return_sum = whether the return is the sum of log probability at each position, or return a list of probabilities
        '''
        log_probs = []
        for i in range(len(seq_onehot) - 1):
            prob, _ = self.get_action_log_prob(seq_onehot[: i + 1], seq_onehot[i + 1])
            log_probs.append(prob.item())
        if return_sum:
            return sum(log_probs)
        else:
            return log_probs


    def get_trainable_parameters(self):
        return self.model.parameters()

