import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from opts import parser
import torch
from models import VideoModel
from dataset import TSNDataSet
import torch.nn as nn
import seaborn
# from torch.nn import Linear, LayerNorm, TransformerEncoder, MultiheadAttention
from myTransformer import mytrans
import os


def modify_argument(args):
    args.batch_size = [16,16,16]
    args.num_segments = 20
    args.num_segments = 20

    return args

transformer = mytrans(d_model=16, nhead=4, dim_feedforward=4096,
                                          num_encoder_layers=2, num_decoder_layers=0)


a = torch.randn(49,15,16) # 15 is batch or length, means 15 images; 16 is d
b = transformer.encoder(a)
def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax)



# sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
# fig, axs = plt.subplots(figsize=(10, 10))
# # for h in range(4):
# #     draw(transformer.encoder.layers[1].attn[0, h].data,
# #         sent, sent if h ==0 else [], ax=axs[h])
# draw(transformer.encoder.layers[1].attn[0].data,
#         [], [], ax=axs)
# plt.show()

args = parser.parse_args()
args = modify_argument(args)

args.batch = [1,1,1]
args.source = 'hmdb51'
args.target = 'ucf101'
args.use_i3d = False
args.use_attention = True
args.method= 'path_gen'
args.use_cdan = False



num_class = 12
model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments,
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params, if_trm = args.if_trm, trm_bottleneck=args.trm_bottleneck)

print(model.transformer)
model = torch.nn.DataParallel(model, args.gpus).cuda(2)

checkpoint = torch.load('exp-tempRGB/temp/hmdb51-ucf101-full-res-att-path-dann/model_best.pth.tar') #
# checkpoint = torch.load('exp-tempRGB/u-h_dann82.2/model_best.pth.tar') #
model.load_state_dict(checkpoint['state_dict'])





def extract(dataet, videoname):
    # feat_path = os.environ['HOME'] + '/dataset/ucf101/RGB-feature/v_Punch_g02_c03'
    # feat_path = os.environ['HOME'] + '/dataset/hmdb51/RGB-feature/50_FIRST_DATES_punch_f_nm_np1_ri_med_16'


    feat_path = os.environ['HOME'] + '/dataset/'+ dataset +'/RGB-feature/' + videoname

    # feat_path = os.environ['HOME'] + '/dataset/ucf101/RGB-feature/v_Punch_g01_c01'
    feat = []
    imgs = os.listdir(feat_path)
    imgs.sort()
    num = len(imgs)
    for img in imgs:
        img_path = os.path.join(feat_path, img)
        feat = torch.load(img_path)
        feat = feat.view(1, 2048, 49).permute(2, 0, 1).cuda(2)
        with torch.no_grad():
            out = model.module.transformer.encoder(feat)
            # _, _, _, _, _, _, _, _, _, _, _,_, _, _,_,attn, _, _= model(feat, [0],feat)
            attn = model.module.transformer.encoder.layers[0].attn[0].cpu()
            region = attn.sum(0).view(7, 7)

            fig, axs = plt.subplots(figsize=(10, 10))
            plt.imshow(region.numpy())
            name = img_path.split('/')[-1][:-3]
            path = './attention_vid/'+ videoname
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path,name))
            # plt.show()
            # fig, axs = plt.subplots(figsize=(10, 10))
            # draw(attn,
            #      [], [], ax=axs)
            # plt.show()
            print(os.path.join(path,name))

if __name__ == '__main__':

    dataset = 'ucf101'
    videoname = 'v_Biking_g01_c01'
    extract(dataset, videoname)









