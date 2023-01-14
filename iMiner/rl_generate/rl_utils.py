'''
Author: Jie Li
Date created: Oct 29, 2020
'''

import math
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter



def create_chunks(complete_list, chunk_size):
    '''
    Cut a list into multiple chunks, each having chunk_size (the last chunk might be less than chunk_size)
    '''
    chunks = []
    for i in range(math.ceil(len(complete_list) / chunk_size)):
        chunks.append(complete_list[i * chunk_size: (i + 1) * chunk_size])
    return chunks

def to_numpy(torch_tensor):
    '''
    Helper function for converting a torch tensor to numpy array
    '''
    if torch_tensor.device.type == "cuda":
        torch_tensor = torch_tensor.cpu()
    return torch_tensor.numpy()

def convert_hours_to_time_string(num_hours):
    left_days, days = math.modf(num_hours / 24)
    left_hours, hours = math.modf(left_days * 24)
    left_minutes, minutes = math.modf(left_hours * 60)
    return "%dd %02dh %02dm" % (days, hours, minutes)

def make_optimizer(optimizer_specs, params):
    if optimizer_specs["optimizer"] == "adam":
        from torch.optim import Adam
        opt = Adam(params, lr=optimizer_specs["learning_rate"])
    elif optimizer_specs["optimizer"] == "sgd":
        from torch.optim import SGD
        opt = SGD(params, lr=optimizer_specs["learning_rate"])
    return opt

class Logger():
    def __init__(self, log_path, print_freq=1) -> None:
        self.logfile = os.path.join(log_path, "logs.csv")
        self.tb_writer = SummaryWriter(log_path, flush_secs=1, max_queue=1)
        self.print_freq = print_freq
        self.headers = None

    def initialize(self, headers):
        self.headers = headers
        if not os.path.exists(self.logfile):
            with open(self.logfile, "w") as f:
                f.write(",".join(headers) + "\n")

    def _tb_log_scalar(self, value, name, iteration):
        self.tb_writer.add_scalar(str(name), value, iteration)

    def log(self, n_iter, logged_values):
        assert self.headers is not None, "The logger has not been initialized"
        # tensorboad logging
        for category in logged_values:
            self._tb_log_scalar(logged_values[category], category, n_iter)

        # csv logging
        if "iteration" not in logged_values:
            logged_values["iteration"] = n_iter
        logged_contents = [logged_values[item] for item in self.headers]
        with open(self.logfile, "a") as f:
            f.write(",".join([str(item) for item in logged_contents]) + "\n")

        # print logging
        if n_iter % self.print_freq == 0:
            self.print_log(logged_values)

    def log_dist(self, n_iter, values, name):
        values = np.array(values)
        self.tb_writer.add_histogram(name, values, n_iter)

    def print_log(self, logged_values):
        for category in logged_values:
            if category == "time_elapsed":
                print(category + ":", convert_hours_to_time_string(logged_values[category]))
            elif category != "iteration":
                print(category + ":", "%.4f" % (logged_values[category]))

class AzureLogger(Logger):
    def __init__(self, azure_run_log, logfile, print_freq=1) -> None:
        super(AzureLogger, self).__init__(logfile, print_freq)
        self.azure_run_log = azure_run_log

    def log(self, n_iter, logged_values):
        super(AzureLogger, self).log(n_iter, logged_values)
        for category in logged_values:
            self.azure_run_log.log(category, float(logged_values[category]))

    def log_image(self, name, image_addr):
        self.azure_run_log.log_image(name=name, path=image_addr)

    def log_distribution(self, name, number_list, bins=20):
        plt.figure()
        plt.hist(number_list, bins=bins)
        plt.xlabel("vina score")
        plt.ylabel("count")
        self.azure_run_log.log_image(name, plot=plt)
