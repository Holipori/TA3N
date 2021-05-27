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

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        if dataset == 'olympic':
            folder = self.output_dir
        else:
            folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 240
        self.resize_width = 320
        # if dataset == 'ucf101':
        #     self.resize_width = 320
        # else:
        #     self.resize_width = 352
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
        if preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

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

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            if video_class == '.DS_Store':
                continue
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):

                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            if file == '.DS_Store':
                continue
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        print(save_dir)
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 1
        # if frame_count // EXTRACT_FREQUENCY <= 16:
        #     EXTRACT_FREQUENCY -= 1
        #     if frame_count // EXTRACT_FREQUENCY <= 16:
        #         EXTRACT_FREQUENCY -= 1
        #         if frame_count // EXTRACT_FREQUENCY <= 16:
        #             EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                # if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                #     frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                frame_name = str(i).zfill(5)
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '{}.jpg'.format(frame_name)), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

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
    # train_data = VideoDataset(dataset='hmdb51', split='test', clip_len=8, preprocess=False)
    # train_data = VideoDataset(dataset='ucf101', split='val', clip_len=8, preprocess=True)

    train_data = VideoDataset(dataset='olympic')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break