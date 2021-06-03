import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from torch.nn.init import *
# a = torch.Tensor([[[6,2,4],[2,3,4]],[[6,2,4],[2,3,4]]])
# b = torch.Tensor([[2,3,4],[1,2,2]])
# c = torch.Tensor([[1],[2]])
# d = torch.Tensor([1,2,3])
# # print(F.softmax(a))
# # print(d-a)
# # print(torch.exp(a))
#
import matplotlib.pyplot as plt
#
# def normal(mu, sigma):
#     ''' Gaussian PDF using keras' backend abstraction '''
#     def f(y):
#         pdf = y - mu
#         pdf = pdf / sigma
#         pdf = - torch.square(pdf) / 2.
#         return torch.exp(pdf) / sigma
#     return f
#
# y = torch.tensor([3,6,9.1])
# mu = torch.tensor([3,6,9])
# sigma = torch.tensor([1,2,3])
# a = normal(mu, sigma)(y)
# # print(a)

# # for extracted feature num
# dir = '/home/xinyue/dataset/ucf101/RGB-feature-i3d'
# sets = os.listdir(dir)
# print(len(sets))
#
# # for raw video num
# total = '/home/xinyue/dataset/hmdb51/RGB'
# dirs = os.listdir(total)
# num = 0
# for dir in dirs:
#     path = total + '/' + dir
#     # print(path)
#     sets = os.listdir(path)
#     num += len(sets)
#     print(num)
#
# # for TA3N original parition
# # file = '/home/xinyue/dataset/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'
# file = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf-feature.txt'
# num = 0
# with open(file) as f:
#     for line in f.readlines():
#         num +=1
#     print(num)

# feat1 = np.load('W.npy')
# feat1 = np.transpose(feat1, (0,2,1))
# feat = feat1[:,:,1]
# print(feat.shape)
# plt.imshow(feat)
# plt.show()


# a = torch.randn(128,4096).cuda(0)
# net = nn.Linear(4096,4096).cuda(0)
# normal_(net.weight, 0, 1)
# constant_(net.bias, 0)
# b= net(a)
# print(b)


print(os.environ['HOME'])
