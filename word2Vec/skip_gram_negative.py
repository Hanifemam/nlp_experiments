import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import skip_gram_negative_dataloader
# from skipgram_negative_data_prepration import dictionary  # ❌ not needed for sizing

class Model():
    def __init__(self, file_dir="word2Vec/dataset.txt", context_words=4, batch_size=8,
                 shuffle=True, embeding_size=128, epochs=10, lr=1e-3):
        # must yield (centers, pos, neg)
        self.loader = skip_gram_negative_dataloader.loader

        # ✅ Size the model from loader IDs (single source of truth)
        max_id = -1
        for centers, pos, neg in self.loader:
            bmax = max(
                centers.max().item() if centers.numel() else -1,
                pos.max().item()     if pos.numel()     else -1,
                neg.max().item()     if neg.numel()     else -1,
            )
            if bmax > max_id:
                max_id = bmax
        vocab_size = max_id + 1
        # (DataLoader can be iterated again later; this pass doesn't "consume" it permanently.)

        self.model = SkipGramNegative(input_size=vocab_size, output_size=vocab_size, embeding_size=embeding_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def __call__(self):
        self.model.train()
        for i in range(self.epochs):
            total_loss, batches = 0.0, 0
            for centers, pos, neg in tqdm(self.loader, desc=f"Epoch {i+1}/{self.epochs} - Training",
                                          unit="batch", dynamic_ncols=True, leave=False):
                centers = centers.to(self.device)   # [B]
                pos     = pos.to(self.device)       # [B]
                neg     = neg.to(self.device)       # [B, K]

                # Optional sanity checks (helpful while debugging)
                V = self.model.input_size
                if centers.numel(): assert centers.max().item() < V, f"center_id >= vocab_size: {centers.max().item()} vs {V}"
                if pos.numel():     assert pos.max().item()     < V, f"pos_id >= vocab_size: {pos.max().item()} vs {V}"
                if neg.numel():     assert neg.max().item()     < V, f"neg_id >= vocab_size: {neg.max().item()} vs {V}"

                s_pos, s_neg = self.model(centers, pos, neg)  # [B], [B, K]

                loss_pos = F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos), reduction="mean")
                loss_neg = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg), reduction="mean")
                loss = loss_pos + loss_neg

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg = total_loss / max(1, batches)
            print(f"epoch {i+1}/{self.epochs} - loss: {avg:.4f}")


class SkipGramNegative(nn.Module):
    def __init__(self, input_size, output_size, embeding_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embeding_size = embeding_size
        dev = 1.0 / (self.embeding_size ** 0.5)
        self.W_in  = nn.Parameter(torch.empty(input_size, embeding_size).uniform_(-dev, dev))  # [V, D]
        self.W_out = nn.Parameter(torch.zeros(embeding_size, output_size))                      # [D, V]

    def forward(self, centers: torch.Tensor, pos: torch.Tensor, negs: torch.Tensor):
        h     = self.W_in[centers]       # [B, D]
        v_pos = self.W_out.t()[pos]      # [B, D]
        v_neg = self.W_out.t()[negs]     # [B, K, D]
        s_pos = (h * v_pos).sum(dim=-1)                  # [B]
        s_neg = (v_neg * h.unsqueeze(1)).sum(dim=-1)     # [B, K]
        return s_pos, s_neg

    def get_embeddings(self):
        return self.W_in.detach()


if __name__ == "__main__":
    def main():
        import os
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

        centers, pos, neg = next(iter(skip_gram_negative_dataloader.loader))
        print(f"First batch shapes — centers: {tuple(centers.shape)}, pos: {tuple(pos.shape)}, neg: {tuple(neg.shape)}")

        m()
        print("✓ Training run completed.")
    main()
