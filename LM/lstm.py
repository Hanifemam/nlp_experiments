
import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from data_preparation import tokenizer, vocab

class LSTMDataset(Dataset):
    def __init__(self):
        super().__init__()
        
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokens = tokenizer()
        vocab_obj = vocab()
        flattened = [tok for sent in tokens for tok in sent]
        tok_id = [vocab_obj[tok] for tok in flattened]
        
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()