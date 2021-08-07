#https://wikidocs.net/52460
import torch
import numpy as np
t = torch.FloatTensor([[1.,2.,3.],[4.,5.,6.]])
print(f'{t.dim()}, {t.shape}, {t.size()}')
m1 = torch.FloatTensor([[1,2]])
m2 = torch.FloatTensor([[3],[4]])
print(m1+m2)
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print(m1.matmul(m2))
t = torch.FloatTensor([[10,4],[20,6]])
print(t.mean(dim=0), t.mean(dim=1))
print(t.sum(dim=0),t.sum(dim=1))
print(t.max(dim=0)) # Returns two values: max and argmax(location)