import os
import numpy as np
import torch
import torch.nn as nn
import random

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)