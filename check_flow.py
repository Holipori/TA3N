import cv2
import os
import torch
import numpy as np

# dir = '/home/xinyue/flow/tvl1_flow/v/_Kill_Bill__Uma_Thruman_sword_fight_with_Lucy_liu_sword_u_cm_np2_fr_med_2'
# # dir = '/home/xinyue/dataset/hmdb51/RGB/climb/The_Fugitive_5_climb_f_cm_np1_ri_med_8'
# sets = os.listdir(dir)
# for item in sets:
#     path = os.path.join(dir, item)
#     x = cv2.imread(path)

# dir = '/home/xinyue/dataset/ucf101/RGB-feature/'
# classes = os.listdir(dir)
# for items in classes:
#     path = os.path.join(dir, items)
#     item = os.listdir(path)
#     for it in item:
#         p = os.path.join(path, it)
#         print(p)
#         x = torch.load(p)
#         if x.shape[0] != 100352:
#             print('a')

dir = '/home/xinyue/dataset/ucf101/RGB-feature/v_BoxingSpeedBag_g24_c01/img_00055.t7'
x = torch.load(dir)
print('a')