import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.clip_grad import clip_grad_norm_

class Model(nn.Module):
    def __init__(self, batch_size, shuffle=True, drop_last=True, emb_size = 128, epochs = 10, lr=1e-3):
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
            specials=[PAD, SOS, EOS, UNK],
            special_first=True,
            min_freq=1
        )
        vocab_eng.set_default_index(vocab_eng[UNK])

        vocab_latin = build_vocab_from_iterator(
            (tgt for _, tgt in tokenized_train + tokenized_val),
            specials=[PAD, SOS, EOS, UNK],
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

        self.model = STSLstm(
            src_vocab_size=len(self.vocab_eng),
            tgt_vocab_size=len(self.vocab_latin),
            emb_size=emb_size,            
            hidden_size=256,
            num_layers=1,
            pad_idx_src=self.pad_src,
            pad_idx_tgt=self.pad_tgt
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_tgt)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        self.epochs = epochs
        self.batch_size = batch_size
    
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
    
    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        for src, tgt in tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}",
                            unit="batch", dynamic_ncols=True, leave=False):
            src = src.to(self.device)
            tgt = tgt.to(self.device) 
            
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            logits = self.model(src, tgt_in)    # (B, T-1, V)
            B, Tm1, V = logits.shape
            
            loss = self.criterion(
            logits.reshape(B*Tm1, V),
            tgt_out.reshape(B*Tm1)
            )
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            tqdm.write(f"batch_loss={loss.item():.4f}") if False else None 
            
        avg_loss = total_loss / len(self.loader)
        return avg_loss
    
    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0.0
        for src, tgt in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:,  1:]

            logits = self.model(src, tgt_in)
            B, Tm1, V = logits.shape
            loss = self.criterion(
                logits.reshape(B*Tm1, V),
                tgt_out.reshape(B*Tm1)
            )
            total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def fit(self):
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch() if len(self.val_loader) > 0 else float("nan")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}/{self.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        return history
    
    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        self.model.eval()
        src = src.to(self.device)
        emb_src = self.model.src_emb(src)
        _, (h, c) = self.model.encoder(emb_src)

        B = src.size(0)
        cur = torch.full((B, 1), self.sos, dtype=torch.long, device=self.device)
        outputs = []

        for _ in range(max_len):
            dec_out, (h, c) = self.model.decoder(self.model.tgt_emb(cur), (h, c))
            step_logits = self.model.proj(dec_out[:, -1:, :])   # (B,1,V)
            next_tok = step_logits.argmax(dim=-1)               # (B,1)
            outputs.append(next_tok)
            cur = torch.cat([cur, next_tok], dim=1)
            if (next_tok == self.eos).all():
                break

        return torch.cat(outputs, dim=1)  # (B, T_gen)


        
        

        
        
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
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=128, hidden_size=256, num_layers=1,
                 pad_idx_src=0, pad_idx_tgt=0):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx_src)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx_tgt)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, tgt_vocab_size)
        
    def forward(self, src, tgt_in):
        enc_out, (h, c) = self.encoder(self.src_emb(src))
        dec_out, _ = self.decoder(self.tgt_emb(tgt_in), (h, c))
        logits = self.proj(dec_out)
        return logits
    
        

m = Model(batch_size=32, epochs=100, lr=1e-3)
history = m.fit()

# Try decoding on a small test batch
src_batch, _ = next(iter(m.test_loader))
pred_ids = m.greedy_decode(src_batch[:4], max_len=30)
