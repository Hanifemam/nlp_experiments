import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
class TranslationDataset(Dataset):
    def __init__(self):
        super().__init__()
        
class STSLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.LSTM()