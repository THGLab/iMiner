from fastai.text import Tokenizer, TextLMDataBunch, load_data
from utils import SELFIESTokenizer, MolTokenizer
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np

def create_data(data, train_size, test_size, datafile=None, vocab=None, batch_size=64, format="SMILES"):
    assert format in ["SMILES", "SELFIES"]
    train, test = train_test_split(np.arange(len(data)), train_size=train_size, test_size=test_size, random_state=42)
    if format == "SMILES":
        tok = Tokenizer(partial(MolTokenizer), pre_rules=[], post_rules=[])
    elif format == "SELFIES":
        tok = Tokenizer(partial(SELFIESTokenizer), pre_rules=[], post_rules=[])

    data = TextLMDataBunch.from_df("/".join(datafile.split("/")[:-1]), data.loc[train], data.loc[test], 
            vocab=vocab, bs=batch_size, tokenizer=tok, 
            text_cols=format.lower(), min_freq=1, include_bos=False, include_eos=False)
    if datafile is not None:
        data.save(f'{datafile}.pkl')
    return data


if __name__ == "__main__":
    import pandas as pd
    import pickle
    import selfies as sf
    
    # create data from txt file
    datapath = "/global/scratch/users/ozhang/covid/MPro/"
    df = pd.read_csv(datapath + "WJ_frags.csv")
    df["selfies"] = df.smiles.apply(sf.encoder)
    
    
    #data = load_data(datapath, 'chembl_frag_gselfies.pkl', bs=1024, bptt=70)
    with open("/global/scratch/users/ozhang/covid/rl_dataset/chembl_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    data = create_data(df, train_size=0.8, test_size=0.2, batch_size=128, vocab=vocab, datafile=datapath + "WJ_frags", format="SELFIES")
    print(data.train_ds.x.vocab.itos)    
    print('number of training items:', len(data.train_ds.items), len(data.train_dl))
    print('number of valid items:', len(data.valid_ds.items), len(data.valid_dl))
    xx, yy = data.one_batch()
    print('batch example:', xx.size())
    #print(xx)
    #print(yy)
    print('train item:', data.train_ds.x[0])
    print('valid item:', data.valid_ds.x[0])

