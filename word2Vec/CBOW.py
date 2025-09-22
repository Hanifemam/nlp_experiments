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
                if target_id in (self.unknown_id, self.pad_id):
                    continue  

                target_context_words = []

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
                    target_context_list.append([target_context_words, target_id])

        return target_context_list

if __name__ in '__main__':
    def main():
        print(PrecomputedCBOWDataset().get_context_target_words())
    main()