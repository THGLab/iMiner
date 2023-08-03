import os
from fastai import *
from fastai.text import *
from utils import *
import pandas as pd

#GPU configurations
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
torch.cuda.set_device(0)
print("GPU availability:", cuda_available)
print("Total GPU count:", num_gpus)
print("Current device:", torch.cuda.get_device_name(torch.cuda.current_device()))


# parse inputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epochs', type = int, default=20)
parser.add_argument('--dropout', type = float, default=0.2)
parser.add_argument('--bs', type = int, default=128)
parser.add_argument('--representation', type=str, default='SELFIES')


args = parser.parse_args()
wd = args.wd
lr = args.lr
n_epochs = args.n_epochs
bs = args.bs
drops = args.dropout
rep = args.representation
datapath = "/global/scratch/users/ozhang/covid/MPro/"
assert rep in ["SMILES", "SELFIES"]


# read prepared data bunch
data = load_data(datapath, 'WJ_moles.pkl', bs=bs, bptt=70)
vocab = data.train_ds.x.vocab
print(vocab.stoi)
print('number of training items:', len(data.train_ds.items), len(data.train_dl))
print('number of valid items:', len(data.valid_ds.items), len(data.valid_dl))


#create learner
learner = language_model_learner(data, AWD_LSTM, drop_mult=drops, wd=wd, pretrained=False)
learner = load_learner(Path("/global/scratch/users/ozhang/covid/rl_dataset/"), 'final_model.pkl')
learner.data = data
#learner.load_pretrained(datapath + "models/chembl_checkpoint.pth", datapath + "chembl_vocab.pkl")
learner.unfreeze()
learner.path = Path(datapath)

calls = []
calls.append(callbacks.SaveModelCallback(learner, every='epoch', name="checkpoint"))

#time to fit
print('----fitting stats-------')
print('wd:', learner.wd)
print('lr:', lr)
print(learner.callbacks)
print(learner.callback_fns)
print(' ')
print(' ')
learner.fit_one_cycle(n_epochs, lr, moms=(0.8, 0.7), callbacks=calls)
learner.export(datapath + "mpro_WJ_pretrained.pkl")

