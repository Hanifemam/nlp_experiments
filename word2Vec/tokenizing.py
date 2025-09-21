import torch
import torch.nn as nn
import re

class CorpusBuilder():
    
    def __init__(self):
        self.dataset_dir = "word2Vec/dataset.txt"
        
    def read_file(self):
        with open(self.dataset_dir, 'r') as f:
            content = f.read()
        return content
    
    def sentence_cleaning(self, content:str):
        content = content.lower().strip().replace(',', '').replace(';', '').replace(':', '').replace('!', '.').replace('?', '.')
        return content
    
    def sentence_to_list(self, content:str):
        content_list = re.split(r'[.\n\t]', content)
        content_list = [sentence for sentence in content_list if len(sentence.strip()) > 0]
        content_list = ['<START> ' + sentence + ' <END>' for sentence in content_list]
        return content_list
    
    def tokenizer(self, content_list:list):
        return [sentence.split(' ') for sentence in content_list]
    
    def get_tokenized_list(self):
        content = self.read_file()
        cleaned_content = self.sentence_cleaning(content)
        content_list = self.sentence_to_list(cleaned_content)
        return self.tokenizer(content_list)

class Vocabulary(): 
    def __init__(self, tokenized_list):
        self.stoi = dict()
        self.itos = dict()
        self.tokenized_list = tokenized_list
        
    def vocabulary_generation(self, tokenized_list):
        id_number = 0
        min_frequency = 3
        word_counter = self.word_frequencies(tokenized_list)
        self.stoi['<pad>'] = id_number
        self.itos[id_number] = '<pad>'
        id_number += 1
        self.stoi['<unknown>'] = id_number
        self.itos[id_number] = '<unknown>'
        id_number += 1
        for tokenized_sentence in tokenized_list:
            for token in tokenized_sentence:
                if word_counter[token] >= min_frequency and token not in self.stoi.keys():
                    self.stoi[token] = id_number
                    self.itos[id_number] = token
                    id_number += 1
        return self.stoi, self.itos
    
    def word_frequencies(self, tokenized_list):
        word_counter = dict()
        for tokenized_sentence in tokenized_list:
            for token in tokenized_sentence:
                if token not in word_counter.keys():
                    word_counter[token] = 1
                else:
                    word_counter[token] = word_counter[token] + 1
        return word_counter
    
    def get_vocabulary(self):
        return self.vocabulary_generation(self.tokenized_list)
        
            
if __name__ == "__main__":
    def main():
        print(Vocabulary(CorpusBuilder().get_tokenized_list()).get_vocabulary())
        
    main()
        