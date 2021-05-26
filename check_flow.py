import cv2
import os

dir = '/home/xinyue/dataset/hmdb51/RGB/climb/The_Fugitive_5_climb_f_cm_np1_ri_med_8'
sets = os.listdir(dir)
for item in sets:
    path = os.path.join(dir, item)
    x = cv2.imread(path)

