import os
import re

path = '/home/ubuntu/dataset/hmdb51/RGB-feature'
dirs = os.listdir(path)
for dir in dirs:
    if '&' in dir:
        old = dir
        new = dir.replace('&','')
        print('old:', old)
        print(new)
        old_p = os.path.join(path,old)
        new_p = os.path.join(path,new)
        os.rename(old_p, new_p)
for dir in dirs:
    if '(' in dir:
        old = dir
        dir = dir.replace('(','')
        new = dir.replace(')','')
        print('old:', old)
        print(new)
        old_p = os.path.join(path,old)
        new_p = os.path.join(path,new)
        os.rename(old_p, new_p)