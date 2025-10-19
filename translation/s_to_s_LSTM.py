import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_train = pd.read_csv("translation/data/wmt14_translate_fr-en_train.csv", delimiter=',', quotechar='"')
        self.raw_valid = pd.read_csv("translation/data/wmt14_translate_fr-en_validation.csv", delimiter=',', quotechar='"')
        
class TranslationDataset(Dataset):
    def __init__(self):
        super().__init__()
        
class STSLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.LSTM()
        

model = Model()