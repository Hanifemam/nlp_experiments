import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=128, num_classes=3, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size,
                                  padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.embed(x)
        self.out, (h, c) = self.lstm(x)
        h_last = h[-1]
        logits = self.linear(h_last)
        return logits
        
        self.lstm = nn.LSTM()
class Model(nn.Module):
    def __init__(self, batch_size, shuffle=True, drop_last=True, emb_size = 128, epochs = 10, lr=1e-3):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw_train = pd.read_csv("/sentiment_analysis/data/sentiment_lstm_medium.csv")
        raw_val = pd.read_csv("/sentiment_analysis/data/sentiment_lstm_val.csv")
        
        tokenizer = get_tokenizer("basic_english")
        label2id = self.label_conv_dict()
        tokenized_train = [(tokenizer(row[0]), label2id[row[1]]) for row in raw_train.values]
        tokenized_val = [(tokenizer(row[0]), label2id[row[1]]) for row in raw_val.values]
        
        special_tok = ["UNK"]
        
        PAD, UNK = "<pad>", "<unk>"
        vocab = build_vocab_from_iterator(
            (tokens for tokens, _ in tokenized_train),
            specials=[PAD, UNK],
            special_first=True,
            min_freq=1
        )
        vocab.set_default_index(vocab[UNK])
        pad_idx = vocab[PAD]
        
        def to_ids(tokens): 
            return vocab.lookup_indices(tokens)

        train_ids = [(to_ids(tokens), y) for tokens, y in tokenized_train]
        val_ids   = [(to_ids(tokens), y) for tokens, y in tokenized_val]
        def collate_batch(batch):
            xs, ys = zip(*batch)
            lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
            xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
            ys = torch.stack(ys)
            return xs_padded, ys, lengths
        
        self.loader = DataLoader(
            dataset=SentimentDataset(train_ids),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_batch
        )
        self.val_loader = DataLoader(
            dataset=SentimentDataset(val_ids),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_batch
        )
        
        vocab_size = len(vocab)
        self.model = SentLSTM(vocab_size=vocab_size, embedding_dim=emb_size, hidden_size=128, num_classes=3, pad_idx=0)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
    def label_conv_dict(self):
        return {"negative": 0,
                 "neutral": 1, 
                 "positive": 2}