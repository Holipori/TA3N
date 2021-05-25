import cv2
import os

dir = '/home/xinyue/dataset/ucf101/RGB/Archery/v_Archery_g06_c05'
sets = os.listdir(dir)
for item in sets:
    path = os.path.join(dir, item)
    x = cv2.imread(path)