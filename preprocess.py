import os
import sys
import cv2
import numpy as np
import time

DATA_DIR = '/home/xinyue/dataset/ucf101/RGB'
SAVE_DIR = '/home/xinyue/dataset/ucf101/flow'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224
_CLASS_NAMES = 'data/label_map.txt'


def get_video_length(video_path):
  _, ext = os.path.splitext(video_path)
  if not ext in _EXT:
    raise ValueError('Extension "%s" not supported' % ext)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened(): 
    raise ValueError("Could not open the file.\n{}".format(video_path))
  if cv2.__version__ >= '3.0.0':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
  else:
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
  length = int(cap.get(CAP_PROP_FRAME_COUNT))
  cap.release()
  return length

def compute_rgb(video_path):
    """Compute RGB"""
    rgb = []
    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    while success:
        frame = cv2.resize(frame, (342,256)) 
        frame = (frame/255.)*2 - 1
        frame = frame[16:240, 59:283]    
        rgb.append(frame)        
        success,frame = vidcap.read()
    vidcap.release()
    rgb = rgb[:-1]
    rgb = np.asarray([np.array(rgb)])
    print('save rgb with shape ',rgb.shape)
    np.save(SAVE_DIR+'/rgb.npy', rgb)
    return rgb
        

def compute_TVL1(video_path):
  """Compute the TV-L1 optical flow."""
  print(video_path)
  flow = []
  TVL1 = cv2.DualTVL1OpticalFlow_create()
  vidcap = cv2.VideoCapture(video_path)
  success,frame1 = vidcap.read()
  bins = np.linspace(-20, 20, num=256)
  prev = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
  vid_len = get_video_length(video_path)
  for _ in range(0,vid_len-2):
      success, frame2 = vidcap.read()
      try:
        curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
      except:
        print('no')
        break
      curr_flow = TVL1.calc(prev, curr, None)
      assert(curr_flow.dtype == np.float32)
      
      #Truncate large motions
      curr_flow[curr_flow >= 20] = 20
      curr_flow[curr_flow <= -20] = -20
     
      #digitize and scale to [-1;1]
      curr_flow = np.digitize(curr_flow, bins)
      curr_flow = (curr_flow/255.)*2 - 1

      new_width = int(curr_flow.shape[1]/curr_flow.shape[0] * 224)
      curr_flow = cv2.resize(curr_flow, (new_width, 224))
      #cropping the center
      # curr_flow = curr_flow[8:232, 48:272]
      curr_flow = curr_flow[:, 37:261]
      flow.append(curr_flow)
      prev = curr
  vidcap.release()
  flow = np.asarray([np.array(flow)])
  print('Save flow with shape ', flow.shape)
  name = video_path.split('/')[-1][:-4]
  print(name)
  np.save(SAVE_DIR+'/'+name + '.npy', flow)
  return flow

def main():
  start_time = time.time()
  print('Extract Flow...')
  dirs = os.listdir(DATA_DIR)
  for dir in dirs:
      path = os.path.join(DATA_DIR, dir)
      vids = os.listdir(path)
      for vid in vids:
          newname = vid[:-4] + '.npy'
          newpath = os.path.join(SAVE_DIR, newname)
          if os.path.exists(newpath):
              continue
          compute_TVL1(path + '/' + vid)
  # compute_TVL1(DATA_DIR+'/v_CricketShot_g04_c01.avi')
  print('Compute flow in sec: ', time.time() - start_time)
  # start_time = time.time()
  # print('Extract RGB...')
  # compute_rgb(DATA_DIR+'/v_CricketShot_g04_c01.avi')
  # print('Compute rgb in sec: ', time.time() - start_time)
  
if __name__ == '__main__':
  print(cv2.__version__)
  main()
