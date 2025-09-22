import torch
from torch import nn
from tqdm.auto import tqdm
from cbow_dataloader import PrecomputedCBOWDataset, CBOWDataLoader

class Model():
    def __init__(self, file_dir="word2Vec/dataset.txt", context_words=4, batch_size=8, shuffle=True, embeding_size=128, epochs=10):
        self.epochs = epochs
        ds = PrecomputedCBOWDataset(file_dir=file_dir, context_words=context_words)
        self.loader = CBOWDataLoader(train_dataset=ds.get_context_target_words(), batch_size=batch_size, shuffle=shuffle)
        vocab_size = len(ds.vocab_stoi)
        pad_id = ds.pad_id
        self.model = CBOW(input_size=vocab_size, output_size=vocab_size, embeding_size=embeding_size, pad_id=pad_id)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def __call__(self):
        self.model.train()
        for i in range(self.epochs):
            total_loss, batches = 0.0, 0
            for b, (contexts, lengths, targets) in enumerate(tqdm(self.loader, desc=f"Epoch {i+1}/{self.epochs} - Training", unit="batch", dynamic_ncols=True, leave=False),
                start=1
            ):
                contexts = contexts.to(self.device)
                lengths  = lengths.to(self.device)
                targets  = targets.to(self.device)
                logits = self.model(contexts, lengths)   
                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1
            avg = total_loss / max(1, batches)
            print(f"epoch {i+1}/{self.epochs} - loss: {avg:.4f}")
            
                
        
class CBOW(nn.Module):
    def __init__(self, input_size, output_size, embeding_size, pad_id=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embeding_size = embeding_size
        self.pad_id = pad_id
        
        dev = 1.0 / (self.embeding_size ** (0.5))
        self.W_in = nn.Parameter(torch.empty(self.input_size, self.embeding_size).uniform_(-dev, dev))
        self.W_out = nn.Parameter(torch.zeros(self.embeding_size, self.output_size))
        
    def forward(self, contexts: torch.Tensor, lengths: torch.Tensor):
        cxt_emb = self.W_in[contexts]
        mask = (contexts != self.pad_id).unsqueeze(-1)
        cxt_emb = cxt_emb * mask
        
        sum_vec = cxt_emb.sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).to(sum_vec.dtype)
        h = sum_vec / denom     
        logits = h @ self.W_out                               
        return logits
    
    def get_embeddings(self, kind: str = "in", normalize: bool = False):
        E = self.W_in
        return E.detach()
    
if __name__ == "__main__":
    def main():
        import os
        import torch

        torch.manual_seed(0)

        data_path = "word2Vec/dataset.txt"
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Expected corpus at: {data_path}")

        m = Model(
            file_dir=data_path,
            context_words=4,
            batch_size=64,
            shuffle=True,
            embeding_size=128,
            epochs=2,   
        )

        first_batch = next(iter(m.loader))
        contexts, lengths, targets = first_batch
        print(
            f"First batch shapes — contexts: {tuple(contexts.shape)}, "
            f"lengths: {tuple(lengths.shape)}, targets: {tuple(targets.shape)}"
        )

        m()
        
        print("✓ Training run completed.")

    main()