import cv2
import os
import numpy as np

dir = '/home/xinyue/flow/tvl1_flow/v/_Kill_Bill__Uma_Thruman_sword_fight_with_Lucy_liu_sword_u_cm_np2_fr_med_2'
# dir = '/home/xinyue/dataset/hmdb51/RGB/climb/The_Fugitive_5_climb_f_cm_np1_ri_med_8'
sets = os.listdir(dir)
for item in sets:
    path = os.path.join(dir, item)
    x = cv2.imread(path)

# dir = '/home/xinyue//dataset/hmdb51/flow'
# sets = os.listdir(dir)
# for item in sets:
#     path = os.path.join(dir, item)
#     x = np.load(path)