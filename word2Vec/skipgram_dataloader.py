import torch
import random

from data_prepration import CorpusBuilder, Vocabulary

class PrecomputedSkipGramDataset():
    def __init__(self, file_dir="word2Vec/dataset.txt", context_words=4):
        self.tokenized_content = CorpusBuilder(file_dir).get_tokenized_list()
        self.vocab_stoi, self.vocab_itos = Vocabulary(self.tokenized_content).get_vocabulary()
        self.context_words = context_words
        self.unknown_id = self.vocab_stoi['<unknown>']
        self.half_window = self.context_words // 2

    def get_context_target_words(self):
        target_context_list = []
        for sentence in self.tokenized_content:
            for i, tok in enumerate(sentence):
                # safe fallback for center
                center_id = self.vocab_stoi.get(tok, self.unknown_id)
                if center_id == self.unknown_id:
                    continue  # skip bad centers

                # collect neighbors up to half_window (exclude center)
                for k in range(1, self.half_window + 1):
                    li = i - k
                    if li >= 0:
                        context_id = self.vocab_stoi.get(sentence[li], self.unknown_id)
                        if context_id != self.unknown_id:  # skip unknown contexts
                            target_context_list.append([center_id, context_id])

                    ri = i + k
                    if ri < len(sentence):
                        context_id = self.vocab_stoi.get(sentence[ri], self.unknown_id)
                        if context_id != self.unknown_id:
                            target_context_list.append([center_id, context_id])

        return target_context_list


class SkipGramDataLoader():
    def __init__(self, train_dataset=None, batch_size=8, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.precomputed_skipgram_dataset = PrecomputedSkipGramDataset()
        if train_dataset is None:
            self.center_context_list = self.precomputed_skipgram_dataset.get_context_target_words()
        else:
            self.center_context_list = train_dataset

        self.limit = len(self.center_context_list)
        self.current = 0
        self.data_index_list = []

    def __iter__(self):
        self.data_index_list = list(range(self.limit))
        if self.shuffle:
            random.shuffle(self.data_index_list)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration

        end = min(self.current + self.batch_size, self.limit)
        batch_indices = self.data_index_list[self.current:end]
        self.current = end

        # gather pairs correctly: centers first, contexts second
        batch_centers = []
        batch_contexts = []
        for idx in batch_indices:
            center_id, context_id = self.center_context_list[idx]
            batch_centers.append(center_id)
            batch_contexts.append(context_id)

        centers = torch.tensor(batch_centers, dtype=torch.long)
        contexts = torch.tensor(batch_contexts, dtype=torch.long)
        return centers, contexts

    def __len__(self):
        return (self.limit + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    def main():
        ds = PrecomputedSkipGramDataset(file_dir="word2Vec/dataset.txt", context_words=4)
        loader = SkipGramDataLoader(train_dataset=ds.get_context_target_words(), batch_size=8, shuffle=True)
        for b, (centers, contexts) in enumerate(loader):
            print(f"batch {b}: centers {centers.shape}, contexts {contexts.shape}")
            break
    main()
