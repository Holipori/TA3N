import numpy as np
import torch
import torch.nn.functional as F
a = torch.Tensor([[[6,2,4],[2,3,4]],[[6,2,4],[2,3,4]]])
b = torch.Tensor([[2,3,4],[1,2,2]])
c = torch.Tensor([[1],[2]])
d = torch.Tensor([1,2,3])
# print(F.softmax(a))
# print(d-a)
# print(torch.exp(a))

import matplotlib.pyplot as plt

def normal(mu, sigma):
    ''' Gaussian PDF using keras' backend abstraction '''
    def f(y):
        pdf = y - mu
        pdf = pdf / sigma
        pdf = - torch.square(pdf) / 2.
        return torch.exp(pdf) / sigma
    return f

y = torch.tensor([3,6,9.1])
mu = torch.tensor([3,6,9])
sigma = torch.tensor([1,2,3])
a = normal(mu, sigma)(y)
# print(a)

feat1 = np.load('feat1.npy')
plt.imshow(feat1)
plt.show()