import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class Model(nn.Module):
    def __init__(self, batch_size, shuffle=True, drop_last=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        raw_train = pd.read_csv("translation/data/train.csv", delimiter=',')
        raw_val = pd.read_csv("translation/data/val.csv", delimiter=',', quotechar='"')
        raw_test = pd.read_csv("translation/data/test.csv", delimiter=',', quotechar='"')
        tokenizer = get_tokenizer("basic_english")
        tokenized_train = [(tokenizer(row[0]), tokenizer(row[1])) for row in raw_train.values]
        tokenized_val = [(tokenizer(row[0]), tokenizer(row[1])) for row in raw_val.values]
        tokenized_test = [(tokenizer(row[0]), tokenizer(row[1])) for row in raw_test.values]
        
        PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unknown>"
        vocab_eng = build_vocab_from_iterator(
            (src for src, _ in tokenized_train + tokenized_val),
            specials=["<pad>", "<sos>", "<eos>", "<unknown>"],
            special_first=True,
            min_freq=1
        )
        vocab_eng.set_default_index(vocab_eng[UNK])

        vocab_latin = build_vocab_from_iterator(
            (tgt for _, tgt in tokenized_train + tokenized_val),
            specials=["<pad>", "<sos>", "<eos>", "<unknown>"],
            special_first=True,
            min_freq=1
        )
        vocab_latin.set_default_index(vocab_eng[UNK])
        
        
        self.vocab_eng = vocab_eng
        self.vocab_latin = vocab_latin
        self.pad_src = vocab_eng[PAD]
        self.pad_tgt = vocab_latin[PAD]
        self.sos = vocab_latin[SOS]
        self.eos = vocab_latin[EOS]
        
        train_ids = [
            (vocab_eng(x), [self.sos] + vocab_latin(y) + [self.eos])
            for x, y in tokenized_train
        ]

        val_ids = [
            (vocab_eng(x), [self.sos] + vocab_latin(y) + [self.eos])
            for x, y in tokenized_val
        ]

        test_ids = [
            (vocab_eng(x), [self.sos] + vocab_latin(y) + [self.eos])
            for x, y in tokenized_test
        ]
        
        train_dataset = TranslationDataset(train_ids)
        val_dataset   = TranslationDataset(val_ids)
        test_dataset  = TranslationDataset(test_ids)
        
        self.loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        src_seqs, tgt_seqs = zip(*batch)

        def pad_to_tensor(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            return out

        src = pad_to_tensor(src_seqs, self.pad_src)
        tgt = pad_to_tensor(tgt_seqs, self.pad_tgt)
        return src, tgt
        
        

        
        
class TranslationDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x_ids, y_ids = self.dataset[idx]
        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)
        return x, y
        
class STSLstm(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.model = nn.LSTM()
        

