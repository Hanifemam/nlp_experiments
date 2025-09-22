
from data_prepration import CorpusBuilder, Vocabulary

class PrecomputedSkipGramataset():
    def __init__(self, file_dir="word2Vec/dataset.txt", batch_size=8, context_words=4, shuffle=True):
        self.tokenized_content = CorpusBuilder("word2Vec/dataset.txt").get_tokenized_list()
        self.vocab_stoi, self.vocab_itos = Vocabulary(self.tokenized_content).get_vocabulary()
        self.batch_size = batch_size
        self.context_words = context_words
        self.shuffle = shuffle
        self.pad = '<pad>'
        self.unknown = '<unknown>'
        
    def get_context_target_words(self):
        target_context_list = []
        for sentence in self.tokenized_content:
            target_context_words = []
            for ind, token in enumerate(sentence):
                for window_ind in range(self.context_words):
                    if ind - window_ind >= 0:
                        target_context_words.append(self.vocab_stoi[sentence[ind - 1]])
                    if ind + window_ind < len(sentence):
                        target_context_words.append(self.vocab_stoi[sentence[ind + 2]])
                if len(target_context_words) > 0:
                    target_context_list.append([target_context_words, self.vocab_stoi[token]])
        return target_context_list
        