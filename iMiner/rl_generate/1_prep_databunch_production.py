from fastai.text import Tokenizer, TextLMDataBunch, load_data
from utils import SELFIESTokenizer, MolTokenizer
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np

def create_data(data, train_size, test_size, datafile=None, batch_size=64, format="SMILES"):
    assert format in ["SMILES", "SELFIES"]
    train, test = train_test_split(np.arange(len(data)), train_size=train_size, test_size=test_size, random_state=42)
    if format == "SMILES":
        tok = Tokenizer(partial(MolTokenizer), pre_rules=[], post_rules=[])
    elif format == "SELFIES":
        tok = Tokenizer(partial(SELFIESTokenizer), pre_rules=[], post_rules=[])

    data = TextLMDataBunch.from_df("/".join(datafile.split("/")[:-1]), data.loc[train], data.loc[test], 
        bs=batch_size, tokenizer=tok, text_cols=format.lower(), min_freq=5, include_bos=False, include_eos=False)
    if datafile is not None:
        data.save(f'{datafile}.pkl')
    return data


if __name__ == "__main__":
    import pandas as pd
    import pickle
    
    # create data from txt file
    datapath = "/global/scratch/users/ozhang/covid/rl_dataset/"
    #df1 = pd.read_csv(datapath + "chembl_cleaned.txt")
    #df2 = pd.read_csv(datapath + "frag_smi.csv")
    #df = pd.concat([df1, df2], ignore_index=True)
    
    # add fragment token
    #tok.add_frag_to_tokens(['Cc1cc(N*1)c2cccc(Cl)c2n1'])
    data = load_data(datapath, 'chembl_frag_gselfies.pkl', bs=1024, bptt=70)
    #data = create_data(df2, train_size=0.85, test_size=0.15, batch_size=128, datafile=datapath + "frag_gselfies", format="SELFIES")
    
    vocab = data.train_ds.x.vocab
    with open(datapath + 'chembl_vocab.pkl', 'wb') as f:
        pickle.dump(vocab.stoi, f)
    print('number of training items:', len(data.train_ds.items), len(data.train_dl))
    print('number of valid items:', len(data.valid_ds.items), len(data.valid_dl))
    xx, yy = data.one_batch()
    print('batch example:', xx.size())
    #print(xx)
    #print(yy)
    print('train item:', data.train_ds.x[0])
    print('valid item:', data.valid_ds.x[0])

