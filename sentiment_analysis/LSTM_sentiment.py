import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.clip_grad import clip_grad_norm_

class SentimentDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_ids, y = self.data[idx]
        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
        
        
    
class SentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw_train = pd.read_csv("/sentiment_analysis/data/sentiment_lstm_medium.csv")
        raw_val = pd.read_csv("/sentiment_analysis/data/sentiment_lstm_val.csv")
        
        tokenizer = get_tokenizer("basic_english")
        label2id = self.label_conv_dict()
        tokenized_train = [(tokenizer(row[0]), label2id[row[1]]) for row in raw_train.values]
        tokenized_val = [(tokenizer(row[0]), label2id[row[1]]) for row in raw_val.values]
        
        special_tok = ["UNK"]
        
        flattened_tok_train = [tok for sent, _ in tokenized_train for tok in sent]
        vocab = build_vocab_from_iterator(
            flattened_tok_train,
            specials=special_tok,
            special_first=True,
            min_freq=1
        )
        
        vocab.set_default_index(vocab["UNK"])
        train_ids = [(vocab(tok), class_) for tok, class_ in tokenized_train]
        val_ids = [(vocab(tok), class_) for tok, class_ in tokenized_val]
        
    def label_conv_dict(self):
        return {"negative": 0,
                 "neutral": 1, 
                 "positive": 2}