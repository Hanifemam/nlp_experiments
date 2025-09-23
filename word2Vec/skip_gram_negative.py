import torch
from torch import nn
from tqdm.auto import tqdm


class Model():
    def __init__(self, file_dir="word2Vec/dataset.txt", context_words=4, batch_size=8,
                 shuffle=True, embeding_size=128, epochs=10):
        pass

    def __call__(self):
        self.model.train()
        for i in range(self.epochs):
           pass


class SkipGram(nn.Module):
    def __init__(self, input_size, output_size, embeding_size, pad_id=0):
        super().__init__()
        
    def forward(self, centers: torch.Tensor):
        """
        centers: LongTensor [B] of center word ids
        returns: logits [B, V] over context vocabulary
        """
        pass

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

        # sanity check first batch
        centers, contexts = next(iter(m.loader))
        print(f"First batch shapes — centers: {tuple(centers.shape)}, contexts: {tuple(contexts.shape)}")

        m()
        print("✓ Training run completed.")
    main()
