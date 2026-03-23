import random
import math

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


class TwoNormalsDataset(IterableDataset):
    def __init__(
            self
            , mu1=1.0
            , mu2=-1.0
            , sigma1=0.1
            , sigma2=0.1
    ):
        super().__init__()

        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __iter__(self):
        while True:
            if random.random() < 0.5:
                yield (torch.randn(2) * self.sigma1 + self.mu1) / (1 + self.sigma1 ** 2) ** 0.5 # 2
            else:
                yield (torch.randn(2) * self.sigma2 + self.mu2) / (1 + self.sigma2 ** 2) ** 0.5 # 2

class UniformDataset(IterableDataset):
    def __init__(
            self
    ):
        super().__init__()

    def __iter__(self):
        while True:
                yield torch.rand(1)

def collate_fn_one_dimension(samples):
    x0 = torch.cat(samples, dim=0) # B

    return x0 # B

def collate_fn_two_dimensions(samples):
    x0 = torch.stack(samples, dim=0) # B, 2

    return x0 # B, 2

class TimeDataset(IterableDataset):
    def __init__(
            self
            , start
            , end
    ):
        super().__init__()

        self.start = start
        self.end = end

    def __iter__(self):
        while True:
            yield self.start + torch.rand(1) * (self.end - self.start)