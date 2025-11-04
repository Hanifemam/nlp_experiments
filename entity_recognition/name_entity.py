import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm

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
    def __init__(self, batch_size=32, shuffle=True, drop_last=True, epochs = 10, lr=1e-3):
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
        PAD_TAG = "<pad_tag>"
        vocab_tags = build_vocab_from_iterator(
            (tags for tags in target_train + target_val),
            specials=[PAD_TAG],
            special_first=True,
            min_freq=1
        )
        vocab_tags.set_default_index(vocab_tags[PAD_TAG])
        self.vocab_tags = vocab_tags 
        self.pad_word_idx = vocab_eng[PAD] 
        self.pad_tag_idx = vocab_tags[PAD_TAG]
        self.vocab_eng = vocab_eng
        train_ids = [vocab_eng.lookup_indices(x) for x in tokenized_train]
        train_tag_ids = [vocab_tags.lookup_indices(tags) for tags in target_train] 
        val_ids = [vocab_eng.lookup_indices(x) for x in tokenized_val]
        val_tag_ids = [vocab_tags.lookup_indices(tags) for tags in target_val]
        train_dataset = EntityDataset(list(zip(train_ids, train_tag_ids))) 
        val_dataset   = EntityDataset(list(zip(val_ids, val_tag_ids)))
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
        self.model = EntityLSTM(
            len(vocab_eng), len(vocab_tags),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_tag_idx)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        self.epochs = epochs
        self.batch_size = batch_size
    
    def collate_fn(self, batch):
        tokens, tags = zip(*batch)
        max_len = max(len(seq) for seq in tokens)
        batch_size = len(tokens)
        token_tensor = torch.full((batch_size, max_len), self.pad_word_idx, dtype=torch.long)
        tag_tensor = torch.full((batch_size, max_len), self.pad_tag_idx, dtype=torch.long)
        for i, (token_seq, tag_seq) in enumerate(zip(tokens, tags)):
            length = len(token_seq)
            token_tensor[i, :length] = torch.tensor(token_seq, dtype=torch.long)
            tag_tensor[i, :length] = torch.tensor(tag_seq, dtype=torch.long)
        return token_tensor, tag_tensor
    
    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        for src, tgt in tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}",
                            unit="batch", dynamic_ncols=True, leave=False):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            logits = self.model(src)
            B, T, C = logits.shape  
            loss = self.criterion(
                logits.view(B * T, C),  
                tgt.view(B * T)         
            )
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
        for src, tgt in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            logits = self.model(src)
            B, T, C = logits.shape  
            loss = self.criterion(
                logits.view(B * T, C),  
                tgt.view(B * T)         
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

class EntityDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)  
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
class EntityLSTM(nn.Module):
    def __init__(self, src_vocab_size, category_size, emb_size=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, category_size)
        
    def forward(self, src):
        enc_out, (h, c) = self.encoder(self.src_emb(src))
        logits = self.proj(enc_out)
        return logits
model = Model()
