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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size,
                                  padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):                        
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False) 
        _, (h, c) = self.lstm(packed)
        h_last = h[-1]
        logits = self.linear(h_last)
        return logits
        
class Model(nn.Module):
    def __init__(self, batch_size, shuffle=True, drop_last=True, emb_size = 128, epochs = 10, lr=1e-3):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.model = SentLSTM(vocab_size=vocab_size, embedding_dim=emb_size, hidden_size=128, num_classes=3, pad_idx=pad_idx)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
    def train_one_epoch(self, epoch:int):
        self.model.train()
        total_loss = 0.0
        for src, tgt, lengths  in tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}",
                            unit="batch", dynamic_ncols=True, leave=False):
            src, tgt, lengths = src.to(self.device), tgt.to(self.device), lengths.to(self.device)

            # Forward pass
            logits = self.model(src, lengths)  
            logits = self.model(src, tgt)
            loss = self.criterion(logits, tgt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            tqdm.write(f"batch_loss={loss.item():.4f}") if False else None 
            
        avg_loss = total_loss / len(self.loader)
        return avg_loss
    
    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        for src, tgt, lengths in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            logits = self.model(src, lengths)
            loss = self.criterion(
                logits,
                tgt
            )
            total_loss += loss.item()
        return total_loss / len(self.val_loader)
        
    def label_conv_dict(self):
        return {"negative": 0,
                 "neutral": 1, 
                 "positive": 2}