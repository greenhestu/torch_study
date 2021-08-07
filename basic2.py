#https://wikidocs.net/52846
import torch
import numpy as np

t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
print(ft.view([-1,3]).shape)
print(ft.view([-1,1,3]).shape)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.squeeze())
print(ft.squeeze().unsqueeze(0)) #첫번째 차원에 추가

bt = torch.ByteTensor([True, False, False, True])
print(bt.long())
print(bt.float())

x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
print(torch.cat([x,y], dim=0)) #2*2 -> 4*2
print(torch.cat([x,y], dim=1)) #2*2 -> 2*4
print(torch.stack([x, x, y]).shape) #2*2 -> 3*2*2
print(torch.stack([x, x, y], dim=1).shape) #2*2 -> 2*3*2
print(torch.stack([x, x, y], dim=2).shape) #2*2 -> 2*2*3

ones = torch.ones_like(x)
zeros = torch.zeros_like(torch.cat([x,y]))
print(ones)
print(ones)
print(zeros)
