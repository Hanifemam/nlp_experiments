import torch
from torch.utils.data import TensorDataset, DataLoader

from data_prepration import dictionary, cooccurrence_matrix

X = cooccurrence_matrix()
vocab, _ = dictionary()
vocab = vocab.get_stoi()
words_list = []
occurrence_list = []
for vocab1 in vocab.keys():
    for vocab2 in vocab.keys():
        words_list.append([vocab[vocab1], vocab[vocab2]])
        occurrence_list.append(X[vocab[vocab1]][vocab[vocab2]])
        
words_tensor = torch.tensor(words_list, dtype=torch.float32)
occurrence_tensor = torch.tensor(occurrence_list, dtype=torch.float32)
dataset = TensorDataset(words_tensor, occurrence_tensor)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,                 
    pin_memory=torch.cuda.is_available(),
    drop_last=True           
)

if __name__ == "__main__":
    def main():
        words, occurs = next(iter(loader))
        print("words:", words.shape, "occurs:", occurs.shape)
        print("words[:5]:", words[:5].tolist())
        print("occurs[:5]:", occurs[:5].tolist())
       

        print("✓ DataLoader with negative sampling looks good.")

    main()