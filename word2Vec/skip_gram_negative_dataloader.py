import torch
from torch.utils.data import TensorDataset, DataLoader

from skipgram_dataloader import PrecomputedSkipGramDataset
from skipgram_negative_data_prepration import dictionary, negative_samples

def pos_neg_samples():
    pairs = PrecomputedSkipGramDataset().get_context_target_words()
    
    centers_tensor = torch.tensor([c for c, _ in pairs], dtype=torch.long)
    contexts_tensor = torch.tensor([o for _, o in pairs], dtype=torch.long)
    
    dataset = TensorDataset(centers_tensor, contexts_tensor)
    
    return dataset

def make_collate_fn(probs: torch.Tensor, K: int = 5):
    def collate(batch):
        centers, pos = zip(*batch)
        centers = torch.tensor(centers, dtype=torch.long)  
        pos     = torch.tensor(pos,     dtype=torch.long)   

        B = centers.size(0)

        neg = torch.multinomial(probs, num_samples=B * K, replacement=True).view(B, K)

        eq = neg.eq(pos.unsqueeze(1))
        while eq.any():
            neg[eq] = torch.multinomial(probs, num_samples=eq.sum().item(), replacement=True)
            eq = neg.eq(pos.unsqueeze(1))

        return centers, pos, neg 
    return collate

# loader = DataLoader(
#     pos_neg_samples(),
#     batch_size=256,
#     shuffle=True,
#     num_workers=0,                 
#     pin_memory=torch.cuda.is_available(),
#     drop_last=True,               
#     collate_fn=make_collate_fn(negative_samples(), K=5),
# )

if __name__ == "__main__":
    def main():
        loader = DataLoader(
            pos_neg_samples(),
            batch_size=256,
            shuffle=True,
            num_workers=0,                 
            pin_memory=torch.cuda.is_available(),
            drop_last=True,               
            collate_fn=make_collate_fn(negative_samples(), K=5),
        )


        # 4) Peek at one batch
        centers, pos, neg = next(iter(loader))
        print("centers:", centers.shape, "pos:", pos.shape, "neg:", neg.shape)
        print("centers[:5]:", centers[:5].tolist())
        print("pos[:5]:", pos[:5].tolist())
        print("neg[0]:", neg[0].tolist())  # first example's 5 negatives

        print("✓ DataLoader with negative sampling looks good.")

    main()
    
    