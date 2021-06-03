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

def modify_argument(args):
    args.batch_size = [16,16,16]
    args.num_segments = 20
    args.num_segments = 20

    return args

transformer = mytrans(d_model=16, nhead=4, dim_feedforward=4096,
                                          num_encoder_layers=2, num_decoder_layers=0)


a = torch.randn(49,15,16)
b = transformer.encoder(a)
def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax)



sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
fig, axs = plt.subplots(figsize=(10, 10))
# for h in range(4):
#     draw(transformer.encoder.layers[1].attn[0, h].data,
#         sent, sent if h ==0 else [], ax=axs[h])
draw(transformer.encoder.layers[1].attn[0].data,
        [], [], ax=axs)
plt.show()

args = parser.parse_args()
args = modify_argument(args)

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
model = torch.nn.DataParallel(model, args.gpus).cuda()
checkpoint = torch.load('exp-tempRGB/hmdb51-ucf101-full-flow-att-path-cdan/model_best.pth.tar') #
# checkpoint = torch.load('exp-tempRGB/u-h_dann82.2/model_best.pth.tar') #
model.load_state_dict(checkpoint['state_dict'])




with torch.no_grad():
    _, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val, feat_base_source0, feat_base_target0, feat_base_source, feat_base_target, _, attention, cdan_loss, latent_features = model(
        source_data, source_label, target_data, [0] * len(args.beta), 0, is_train=True, reverse=False,
        batchsize=int(batch_source_ori))

