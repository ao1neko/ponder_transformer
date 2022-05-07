import argparse
from cmath import log
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict,Tuple

def sample_datasets2strlist()-> Tuple[List[str],List[str]]:
    inputs = ["A = 1, B = 12, C = A + B, C = ?", "1", "21"]
    labels = ["A = 1, B = 12, C = A + B, C = ?", "1", "21"]
    return (inputs,labels)
    
