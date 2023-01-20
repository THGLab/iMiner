from fastai import *
from fastai.text import *
from utils import *


def create_data(dataset_folder, batch_size=64, format="SMILES"):
    assert format in ["SMILES", "SELFIES"]
    corpus_train = pd.read_csv(dataset_folder + 'chembl_train.csv',index_col=None,)
    corpus_valid = pd.read_csv(dataset_folder + 'chembl_val.csv',index_col=None)
    if format == "SMILES":
        tok = Tokenizer(partial(MolTokenizer), pre_rules=[], post_rules=[])
    elif format == "SELFIES":
        tok = Tokenizer(partial(SELFIESTokenizer), pre_rules=[], post_rules=[])
    data_path = './data/'

    data = TextLMDataBunch.from_df(data_path, corpus_train, corpus_valid, bs=batch_size, tokenizer=tok,
         text_cols=format.lower(), min_freq=1, include_bos=False, include_eos=False)
    data.save(f'databunch-production-{format}.pkl')
    return data

if __name__ == "__main__":
    data = create_data("./data/", format="SELFIES")
    print( data.train_ds.x[0].text )
    print( data.valid_ds.x[0].text )
    print( data.train_ds.x.vocab.stoi)
    print( data.train_ds.x.vocab.itos)


