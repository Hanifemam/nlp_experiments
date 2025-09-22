import torch
import random

from data_prepration import CorpusBuilder, Vocabulary

class PrecomputedCBOWDataset():
    def __init__(self, file_dir="word2Vec/dataset.txt", context_words=4):
        self.tokenized_content = CorpusBuilder(file_dir).get_tokenized_list()
        self.vocab_stoi, self.vocab_itos = Vocabulary(self.tokenized_content).get_vocabulary()
        self.context_words = context_words
        self.pad_id = self.vocab_stoi['<pad>']
        self.unknown_id = self.vocab_stoi['<unknown>']
        self.half_window = self.context_words // 2

    def get_context_target_words(self):
        target_context_list = []
        for sentence in self.tokenized_content:
            for i, tok in enumerate(sentence):
                target_id = self.vocab_stoi.get(tok, self.unknown_id)
                # skip bad targets
                if target_id in (self.unknown_id, self.pad_id):
                    continue

                target_context_words = []

                # collect neighbors up to half_window (exclude center)
                for k in range(1, self.half_window + 1):
                    li = i - k
                    if li >= 0:
                        left_id = self.vocab_stoi.get(sentence[li], self.unknown_id)
                        if left_id != self.pad_id:
                            target_context_words.append(left_id)

                    ri = i + k
                    if ri < len(sentence):
                        right_id = self.vocab_stoi.get(sentence[ri], self.unknown_id)
                        if right_id != self.pad_id:
                            target_context_words.append(right_id)

                if target_context_words:
                    # (context_ids_list, target_id)
                    target_context_list.append([target_context_words, target_id])

        return target_context_list


class CBOWDataLoader():
    def __init__(self, train_dataset=None, batch_size=8, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build dataset internally if none provided
        self.precomputed_CBOW_dataset = PrecomputedCBOWDataset()
        if train_dataset is None:
            self.context_target_list = self.precomputed_CBOW_dataset.get_context_target_words()
        else:
            self.context_target_list = train_dataset

        self.limit = len(self.context_target_list)
        self.current = 0
        self.data_index_list = []

    def __iter__(self):
        # prepare (and optionally shuffle) indices each epoch
        self.data_index_list = list(range(self.limit))
        if self.shuffle:
            random.shuffle(self.data_index_list)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration

        end = self.current + self.batch_size
        if end > self.limit:
            end = self.limit

        batch_indices = self.data_index_list[self.current:end]
        self.current = end

        # gather raw contexts/targets for this batch
        batch_contexts = []
        batch_targets = []
        for idx in batch_indices:
            context_ids, target_id = self.context_target_list[idx]
            batch_contexts.append(list(context_ids))  # ensure a fresh list
            batch_targets.append(target_id)

        # pad to the batch's max context length
        Lmax = max(len(c) for c in batch_contexts) if batch_contexts else 0
        padded_contexts = []
        lengths = []
        pad_id = self.precomputed_CBOW_dataset.pad_id

        for c in batch_contexts:
            lengths.append(len(c))
            if len(c) < Lmax:
                c = c + [pad_id] * (Lmax - len(c))
            padded_contexts.append(c)

        contexts = torch.tensor(padded_contexts, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)
        targets = torch.tensor(batch_targets, dtype=torch.long)

        return contexts, lengths, targets


if __name__ == "__main__":
    def main():
        ds = PrecomputedCBOWDataset(file_dir="word2Vec/dataset.txt", context_words=4)
        loader = CBOWDataLoader(train_dataset=ds.get_context_target_words(), batch_size=8, shuffle=True)
        for b, (contexts, lengths, targets) in enumerate(loader):
            print(f"batch {b}: contexts {contexts.shape}, lengths {lengths.shape}, targets {targets.shape}")
            break  # show only first batch
    main()
