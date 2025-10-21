import torch
from torch import nn
import torch.nn.init as init
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.clip_grad import clip_grad_norm_

class SentimentDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
class SentLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()