import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def data_prep(dir):
    data_set = []
    sentence_tokens, sentence_tags = [], []
    with open(dir, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence_tokens:
                    data_set.append([sentence_tokens, sentence_tags])
                    sentence_tokens, sentence_tags = [], []
                continue
            sample = line.split()
            if len(sample) >= 2:
                sentence_tokens.append(sample[0])
                sentence_tags.append(sample[-1])
    if sentence_tokens:
        data_set.append([sentence_tokens, sentence_tags])
    return data_set
                    
                    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dir = "entity_recognition/dataset/ner_train.conll"
        val_dir = "entity_recognition/dataset/ner_dev.conll"
        raw_train = data_prep(train_dir)
        raw_val = data_prep(val_dir)
        tokenized_train = [[token.lower() for token in sentence[0]] for sentence in raw_train if sentence[0]]
        target_train = [sentence[1] for sentence in raw_train if sentence[1]]
        tokenized_val = [[token.lower() for token in sentence[0]] for sentence in raw_val if sentence[0]]
        target_val = [sentence[1] for sentence in raw_val if sentence[1]]
        PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unknown>"
        vocab_eng = build_vocab_from_iterator(
            (src for src in tokenized_train),
            specials=[PAD, SOS, EOS, UNK],
            special_first=True,
            min_freq=1
        )
        vocab_eng.set_default_index(vocab_eng[UNK])
        self.vocab_eng = vocab_eng
        train_ids = [(vocab_eng(x)) for x in tokenized_train]
        train_dataset = [train_ids, target_train]
        val_ids = [(vocab_eng(x)) for x in tokenized_val]
        val_dataset = [val_ids, target_val]

class EntityDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset[:,0])
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx][0], self.dataset[idx][1]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
model = Model()
