import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode',  default= 'rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', default= '2', type= str)
parser.add_argument('--source', type=str, default= 'ucf101')
# parser.add_argument('--target', type=str, default= 'hmdb51')
parser.add_argument('--num_segments', type=int, default=100)
parser.add_argument('--val_segments', type=int, default=100)
parser.add_argument('--save_dir', type = str, default= '/home/xinyue/dataset/')

parser.add_argument('-b', '--batch_size', default=[1,1,1], type=int, nargs="+", # 128 74 128 #64,74,128
                    metavar='N', help='mini-batch size ([source, target, testing])')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d
from dataset_ori import VideoDataset
from torch.utils.data import DataLoader


from charades_dataset_full import Charades as Dataset

print(torch.cuda.device_count())
print(torch.cuda.is_available())
total_feat_out = []
total_feat_in = []
device = 'cuda:1'
def hook_fn_forward(module, input, output):
    total_feat_out.append(output)  # 然后分别存入全局 list 中
    total_feat_in.append(input)


def run(dataset_name = args.source, split = 'train', mode='rgb', root='/ssd2/charades/Charades_v1_rgb', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #
    # val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataset = VideoDataset(dataset=dataset_name, split= split, clip_len=args.num_segments)
    dataloader = DataLoader( dataset, batch_size=args.batch_size[0], shuffle=False, num_workers=1)

    # val_dataset = VideoDataset(dataset=args.target, split='test', clip_len=args.val_segments)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size[1], num_workers=4)

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    # modules = i3d.named_children()
    # for name, module in modules:
    #     module.register_forward_hook(hook_fn_forward)

    i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load(load_model))
    i3d = i3d.to(device)
    # i3d = torch.nn.DataParallel(i3d, 2).cuda()

    # path = '/home/xinyue/dataset/ucf/out/'
    # classes = os.listdir(path)
    # for cls in classes:
    #     cls_dir = os.path.join(path, cls)
    #     for video in cls:

    i3d.train(False)  # Set model to evaluate mode

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, labels, name = data
        if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
            continue

        b,c,t,h,w = inputs.shape
        feature_dir = save_dir + dataset_name + '/RGB-feature-i3d/'
        if t > 1600: # PASS
            features = []
            for start in range(1, t-56, 1600):
                end = min(t-1, start+1600+56)
                start = max(1, start-48)
                ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
        else:
            # wrap them in Variable
            with torch.no_grad():
                inputs = inputs.to(device)
                out_dir = feature_dir + name[0]
                print(out_dir)
                # if os.path.exists(out_dir):
                #     continue
                try:
                    features = i3d.extract_features(inputs)[0]
                except:
                    print(inputs.shape)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                features = features.squeeze(0).permute(1,0,2,3).data.detach().cpu()
                print(features.shape)
                for i in range(features.shape[0]):
                    id_frame = i + 1
                    id_frame_name = str(id_frame).zfill(5)
                    filename = out_dir +'/img_' +id_frame_name +'.t7'
                    torch.save(features[i].clone(), filename)
                torch.cuda.empty_cache()
                # np.save(filename, features)
            # np.save(os.path.join(out_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'hmdb51',split= 'train')
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'hmdb51',split= 'test')
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'hmdb51',split= 'val')
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'ucf101',split= 'train')
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'ucf101',split= 'test')
    # run(mode='rgb', save_dir=args.save_dir, dataset_name = 'ucf101',split= 'val')
    run(mode='rgb', save_dir=args.save_dir, dataset_name = 'olympic')
