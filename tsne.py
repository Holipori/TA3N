import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from opts import parser
import torch
from models import VideoModel
from dataset import TSNDataSet

args = parser.parse_args()

def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
num_class = len(class_names)
model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments,
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params, if_trm = args.if_trm, trm_bottleneck=args.trm_bottleneck)

model = torch.nn.DataParallel(model, args.gpus).cuda()
# checkpoint = torch.load('exp-tempRGB/temp/2021-05-27 23:48:42/model_best.pth.tar') #
checkpoint = torch.load('exp-tempRGB/u-h_no_path_cdan/model_best.pth.tar') #
model.load_state_dict(checkpoint['state_dict'])


num_source = sum(1 for i in open(args.train_source_list))
num_target = sum(1 for i in open(args.train_target_list))

source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source, num_segments=args.num_segments,
									new_length=1, modality=args.modality,
									image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
									random_shift=True,
									test_mode=True,
									)

source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target, num_segments=args.num_segments,
                        new_length=1, modality=args.modality,
                        image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                        random_shift=True,
                        test_mode=True,
                        )

target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)
data_loader = enumerate(zip(source_loader, target_loader))
## val
source = []
target = []
for i, ((source_data, source_label),(target_data, target_label)) in data_loader:
	source_size_ori = source_data.size()  # original shape
	target_size_ori = target_data.size()  # original shape
	batch_source_ori = source_size_ori[0]
	batch_target_ori = target_size_ori[0]
	if batch_source_ori < args.batch_size[0]:
		dummy = True
		source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1],
										source_size_ori[2], source_size_ori[3], source_size_ori[4])
		source_data = torch.cat((source_data, source_data_dummy))
	if batch_target_ori < args.batch_size[1]:
		target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1],
										target_size_ori[2], target_size_ori[3], target_size_ori[4])
		target_data = torch.cat((target_data, target_data_dummy))

	with torch.no_grad():
		_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val, feat_base_source0, feat_base_target0, feat_base_source, feat_base_target, _, avg_loss, cdan_loss, latent_features = model(source_data, source_label, target_data, [0]*len(args.beta), 0, is_train=True, reverse=False, batchsize = int(batch_source_ori))
		latent_features[0], latent_features[1], out_val_2, pred_domain_val, feat_val = removeDummy(latent_features[0], latent_features[1], out_val_2,
																			  pred_domain_val, feat_val, batch_source_ori)
	if source == []:
		source = latent_features[0].view(-1, latent_features[0].shape[-1]).cpu()
	else:
		source = torch.cat([source,latent_features[0].view(-1, latent_features[0].shape[-1]).cpu()])
	if target == []:
		target = latent_features[1].view(-1, latent_features[1].shape[-1]).cpu()
	else:
		target = torch.cat([target,latent_features[1].view(-1, latent_features[1].shape[-1]).cpu()])

source_num = len(source)
target_num = len(target)
X = torch.cat([source,target])
'''X是特征，不包含target;X_tsne是已经降维之后的特征'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(5, 5))

half = int(X_norm.shape[0]/2)

for i in range(target_num):
	plt.scatter(X_norm[source_num+ i, 0], X_norm[source_num + i, 1], c='r', s = 5, alpha = 0.5)
for i in range(source_num):
	plt.scatter(X_norm[i, 0], X_norm[i, 1], c ='b', s = 4, alpha= 0.5)

# X = target
# '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
# X_tsne = tsne.fit_transform(X)
# print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#
# for i in range(X_norm.shape[0]):
# 	plt.scatter(X_norm[i, 0], X_norm[i, 1], c='r', s = 1, alpha = 0.5)
# for i in range(X_norm.shape[0]):
# 	plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
# 			 fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()

