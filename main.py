# Before we get started we will load all the packages we will need

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os.path
import time
import math
import random
import matplotlib.pyplot as plt
import string

# Use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('hello')