import os
from fastai import *
from fastai.text import *
from utils import *
# from rdkit import RDLogger

try:
    from azureml.core.run import Run
    from AzureMetricLogger import MetricLogger
    azure_training = True
    run = Run.get_context()
except:
    azure_training = False

# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)


#==================================================================================
#GPU configurations
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
torch.cuda.set_device(0)
print("GPU availability:", cuda_available)
print("Total GPU count:", num_gpus)
print("Current device:", torch.cuda.get_device_name( torch.cuda.current_device() ) )


#=================================================================================
# parse inputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--wd', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_epochs', type = int, default=30)
parser.add_argument('--dropout', type = float, default=0.2)
parser.add_argument('--bs', type = int, default=512)
parser.add_argument('--representation', type=str, default='SELFIES')


args = parser.parse_args()
dp = args.data_path
wd = args.wd
lr = args.lr
n_epochs = args.n_epochs
bs = args.bs
drops = args.dropout
rep = args.representation
assert rep in ["SMILES", "SELFIES"]
print("Representation format:", rep)
print("Mounted data path:", dp)


running_path = os.getcwd()
print("Running script path:", running_path)

'''
#for deubugging:
wd, lr, n_epochs, bs, drops = 1e-2, 8e-3, 20, 512, 0.2
'''

if azure_training:
    run.log('weight_decay', wd)
    run.log('learning_rate', lr)
    run.log('n_epochs', n_epochs)
    run.log('batch_size', bs)
    run.log('dropout', drops)

#=================================================================================
# read prepared data bunch
os.makedirs("./outputs/", exist_ok=True)
data = load_data('./data/', f'databunch-production-{rep}.pkl', bs=bs, bptt=70)
print('loading prod data bunch')

vocab = data.train_ds.x.vocab
torch.save(vocab, './outputs/vocab_init.pkl')
print( vocab.stoi)
print('model output path:', data.path)
print('number of training items:', len( data.train_ds.items ), len(data.train_dl) )
print('number of valid items:', len( data.valid_ds.items), len(data.valid_dl) )
xx, yy = data.one_batch()
print('batch example:', xx.size() )
print( xx )
print( yy )
print('train item:', data.train_ds.x[0] )
print('valid item:', data.valid_ds.x[0] )




#=================================================================================
#create learner
learner = language_model_learner(data, AWD_LSTM, drop_mult=drops, wd=wd, pretrained=False)
learner.unfreeze()
learner.path = Path("./outputs/")
os.makedirs("./outputs/models/")
print(learner.model )
print('learner path:', learner.path)
print('learner wd:', learner.wd)


#=================================================================================
#callbacks
calls = []
calls.append( callbacks.SaveModelCallback(learner,every='epoch',name="checkpoint") )
if azure_training:
    calls.append( MetricLogger(learner, run))


#=================================================================================
#time to fit
print('----fitting stats-------')
print('wd:', learner.wd)
print('lr:', lr)
print('bs:', learner.data.bs)
print('n_epochs:', n_epochs)
print('dropout:', drops)
print( learner.callbacks)
print( learner.callback_fns)
print(' ')
print(' ')
print(' ')
learner.fit_one_cycle(n_epochs, lr, moms=(0.8,0.7), callbacks=calls )

#==============================================================================
#export
# learner.path = Path(outE)
learner.export("final_model.pkl")


