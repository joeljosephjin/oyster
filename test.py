import torch
import numpy as np
import random
from test1 import f


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# def f():
	# return random.random()

print(f())
print(torch.rand(1).item())
print(np.random.rand())
