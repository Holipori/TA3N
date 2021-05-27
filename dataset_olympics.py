import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import torchvision.transforms as transforms
from PIL import Image

'''
    this file is for I3D to load jpg frames from 'out' dir.
    also, this file is for extract jpg from videos in 'RGB' dir.
'''

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='olympic', split='', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = self.output_dir

        self.crop_size = 224
        self.label2index = {}
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # if (not self.check_preprocess()) or preprocess:


        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            if label == '.DS_Store':
                continue
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # # Prepare a mapping between the label names (strings) and indices (ints)
        # self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # # Convert the list of label names into an array of label indices
        # self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            with open('/home/xinyue/TA3N/dataloaders/ucf_labels.txt', 'r') as f:
                for line in f.readlines():
                    temp = line.split()
                    self.label2index[temp[1]] = int(temp[0])
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        elif dataset == 'hmdb51':
            with open('/home/xinyue/TA3N/dataloaders/hmdb_labels.txt', 'r') as f:
                for line in f.readlines():
                    temp = line.split()
                    self.label2index[temp[1]] = int(temp[0])
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        elif dataset == 'olympic':
            with open('/home/xinyue/TA3N/data/olympic_splits/class_list_ucf_olympic.txt', 'r') as f:
                for line in f.readlines():
                    temp = line.split()
                    self.label2index[temp[1]] = int(temp[0])
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer, name = self.load_frames(self.fnames[index])
        # buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        # if self.split == 'test':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)
        # buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        # buffer = self.transform(buffer)
        return buffer, torch.from_numpy(labels), name

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def to_tensor(self, buffer):
        return buffer.permute(3, 0, 1, 2)

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        frame = np.array(cv2.imread(frames[0])).astype(np.float64)
        # buffer = np.empty((frame_count, frame.shape[0], frame.shape[1], 3), np.dtype('float32'))
        buffer = torch.zeros(frame_count, self.crop_size, self.crop_size,3)
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name))
            im = Image.fromarray(frame)
            t_im = self.transform(im)
            buffer[i] = t_im.permute(1,2,0)
        name = file_dir.split('/')[-1]
        return buffer, name

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(max(1, buffer.shape[0] - clip_len))
        h_start = int((buffer.shape[1] - self.crop_size)/2)
        w_start = int((buffer.shape[2] - self.crop_size)/2)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        # sample_fequency = 2
        # buffer2 = buffer[time_index:time_index + clip_len:sample_fequency,
        #           height_index:height_index + crop_size,
        #           width_index:width_index + crop_size, :].copy()
        buffer2 = buffer[:,h_start: h_start+ self.crop_size, w_start: w_start+ self.crop_size,:]
        # if buffer2.shape[0] < clip_len:
        #     buffer2 = np.append(buffer2, buffer[time_index:time_index + (
        #                 clip_len - buffer2.shape[0]) * sample_fequency: sample_fequency,
        #                                  height_index:height_index + crop_size,
        #                                  width_index:width_index + crop_size, :].copy(), axis=0)

        return buffer2





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # train_data = VideoDataset(dataset='ucf101', split='train', clip_len=8, preprocess=False)
    train_data = VideoDataset(dataset='olympic')
    # train_data = VideoDataset(dataset='ucf101', split='val', clip_len=8, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 100:
            break