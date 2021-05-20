from torch import nn

from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import TRNmodule
import math
import numpy as np

from colorama import init
from colorama import Fore, Back, Style

from soft_dtw_cuda import SoftDTW

from opts import parser
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

args = parser.parse_args()

# definition of Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None

# definition of Gradient Scaling Layer
class GradScale(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output * ctx.beta
		return grad_input, None

class AdversarialNetwork(nn.Module):
	def __init__(self, in_feature, hidden_size):
		super(AdversarialNetwork, self).__init__()
		self.ad_layer1 = nn.Linear(in_feature, hidden_size)
		self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
		self.ad_layer3 = nn.Linear(hidden_size, 2)
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()
		# self.apply(init_weights)
		# self.iter_num = 0
		# self.alpha = 10
		# self.low = 0.0
		# self.high = 1.0
		# self.max_iter = 10000.0

	def forward(self, feat, alpha=1):
		# x = x * 1.0
		x = GradReverse.apply(feat, 1)
		# x.register_hook(grl_hook(alpha))
		x = self.ad_layer1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.ad_layer2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		y = self.ad_layer3(x)
		y = self.sigmoid(y)
		return y

class AdversarialNetwork_v(nn.Module):
	def __init__(self, in_feature, hidden_size):
		super(AdversarialNetwork_v, self).__init__()
		self.ad_layer1 = nn.Linear(in_feature, hidden_size)
		self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
		self.ad_layer3 = nn.Linear(hidden_size, 2)
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()
		# self.apply(init_weights)
		# self.iter_num = 0
		# self.alpha = 10
		# self.low = 0.0
		# self.high = 1.0
		# self.max_iter = 10000.0

	def forward(self, feat, alpha=1):
		# x = x * 1.0
		# x = GradReverse.apply(feat, 1)
		x = feat
		# x.register_hook(grl_hook(alpha))
		x = self.ad_layer1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.ad_layer2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		y = self.ad_layer3(x)
		y = self.sigmoid(y)
		return y


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

# definition of Temporal-ConvNet Layer
class TCL(nn.Module):
	def __init__(self, conv_size, dim):
		super(TCL, self).__init__()

		self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size,1), padding=(conv_size//2,0))

		# initialization
		kaiming_normal_(self.conv2d.weight)

	def	forward(self, x):
		x = self.conv2d(x)

		return x



class fc_l(nn.Module):
    def __init__(self, out, inp=256, methods="weight"):
        super(fc_l, self).__init__()
        self.fc = nn.Linear(inp, out)
        # self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


def Entropy(input_):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	return entropy

class VideoModel(nn.Module):
	def __init__(self, num_class, baseline_type, frame_aggregation, modality,
				train_segments=5, val_segments=25,
				base_model='resnet101', path_pretrained='', new_length=None,
				before_softmax=True,
				dropout_i=0.5, dropout_v=0.5, use_bn='none', ens_DA='none',
				crop_num=1, partial_bn=True, verbose=True, add_fc=1, fc_dim=1024,
				n_rnn=1, rnn_cell='LSTM', n_directions=1, n_ts=5,
				use_attn='TransAttn', n_attn=1, use_attn_frame='none',
				share_params='Y', if_trm = False, trm_bottleneck = 256):
		super(VideoModel, self).__init__()
		self.num_class = num_class
		self.modality = modality
		self.train_segments = train_segments
		self.val_segments = val_segments
		self.baseline_type = baseline_type
		self.frame_aggregation = frame_aggregation
		self.reshape = True
		self.before_softmax = before_softmax
		self.dropout_rate_i = dropout_i
		self.dropout_rate_v = dropout_v
		self.use_bn = use_bn
		self.ens_DA = ens_DA
		self.crop_num = crop_num
		self.add_fc = add_fc
		self.fc_dim = fc_dim
		self.share_params = share_params
		self.if_trm = if_trm
		self.trm_bottleneck = trm_bottleneck

		#cdan


		# transformer, attention
		self.d_model = 2048
		# self.fc_tr = nn.Linear(2048, self.d_model).cuda()
		self.enc_pos_encoding = torch.autograd.Variable(torch.empty((49, self.d_model)).cuda().uniform_(-0.1, 0.1),
														requires_grad=True)
		self.transformer = nn.Transformer(d_model=self.d_model, nhead=8, dim_feedforward=4096,
										  num_encoder_layers=2, num_decoder_layers=0).cuda() # [seq_len, batch, d]
		self.pool = nn.AdaptiveAvgPool2d((1, 1))

		# temporal attention
		self.enc_pos_encoding2 = torch.autograd.Variable(torch.empty((args.num_segments, self.d_model)).cuda().uniform_(-0.1, 0.1),
														requires_grad=True)
		self.transformer2 = nn.Transformer(d_model=self.d_model, nhead=8, dim_feedforward=4096,
										  num_encoder_layers=2, num_decoder_layers=0).cuda()


		# feature size
		self.feature_size = 2048
		# gtw
		self.basis_num = 6
		self.hidden_size2 = 2048
		self.layer_num = 2

		# video classifier
		self.video_cls = torch.nn.LSTM(self.feature_size, self.feature_size, 2 )
		self.video_cls_fc = nn.Linear(self.feature_size, 256)
		self.video_cls_fc2 = nn.Linear(256, num_class)
		# domain classifier
		self.domain_cls = torch.nn.LSTM(self.feature_size, self.feature_size, 2 )
		self.domain_cls_fc = nn.Linear(self.feature_size, 2)
		self.domain_cls_fc2 = nn.Linear(256, 2)

		if args.method == 'path_gen':
			self.path_gen = torch.nn.LSTM(self.feature_size, self.train_segments, self.layer_num)
		elif args.method == 'QB':
			# Q and B method
			self.gamma = 60
			self.Q_gen = torch.nn.LSTM(self.feature_size, self.gamma, self.layer_num )
			# self.H_gen = nn.Linear(self.feature_size, self.gamma)
			self.H_gen = torch.nn.LSTM(self.feature_size, self.gamma, self.layer_num, batch_first=True)
			# self.H_gen_fc = nn.Linear(256, self.gamma)
			self.basis = torch.nn.Parameter(torch.randn(self.gamma, self.train_segments))

			self.H_gen_fc = nn.Sequential(
					nn.Linear(self.feature_size, self.feature_size),
					# nn.BatchNorm1d(self.feature_size, affine=True),
					nn.ReLU(),
					nn.Linear(self.feature_size, self.gamma)
				)



		# self.Q_gen = nn.Sequential(
		# 	# nn.Linear(self.feature_size, self.hidden_size2),
		# 	# nn.BatchNorm1d(self.hidden_size2, affine=True),
		# 	# nn.ReLU(),
		# 	nn.Linear(self.hidden_size2, self.gamma)
		# )


		#lstm
		# input: seq, batch, input_size
		# hidden: layer, batch, hidden_size
		self.hidden_size = 256
		self.output_size = self.feature_size
		self.group_number = 100
		# self.lstm = torch.nn.LSTM(2048,self.hidden_size) #input output
		# self.fc_mu = nn.Linear(self.hidden_size, self.output_size*self.group_number)
		# normal_(self.fc_mu.weight, 0, 1)
		# constant_(self.fc_mu.bias, 0)
		# self.fc_sigma = nn.Linear(self.hidden_size, self.output_size* self.group_number)
		# normal_(self.fc_sigma.weight, 0, 0.001)
		# constant_(self.fc_sigma.bias, 0)
		# self.fc_pi = nn.Linear(self.hidden_size, self.group_number)
		# normal_(self.fc_pi.weight, 0, 0.001)
		# constant_(self.fc_pi.bias, 0)




		# # rmdn correction
		# self.fc_feature_class0 = nn.Linear(self.output_size,256)
		# self.fc_feature_class1 = nn.Linear(256,num_class)
		# self.fc_feature_class2 = nn.Linear(256,num_class)
		# self.criterion =  torch.nn.CrossEntropyLoss(weight=torch.ones(num_class).cuda()).cuda()


		# average sequence
		# self.sdtw = SoftDTW(True, gamma=1.0, normalize=False)
		# self.avg = torch.nn.Parameter(torch.rand(num_class,train_segments,self.output_size)) # class, frame, feature

		# self.trm_bottleneck = self.output_size
		if args.use_cdan == True:
			self.netD = AdversarialNetwork2(2048* self.num_class, 1024).cuda()
		else:
			self.netD = AdversarialNetwork(2048, 1024).cuda()
		self.netD2 = AdversarialNetwork_v(2048, 1024).cuda()
		# fc test
		# self.fc_layer = fc_l(256,int(10240 * ((1/4)*add_fc + (1-add_fc))))
		# self.fc_layer2 = fc_l(256,int(10240 * ((1/4)*add_fc + (1-add_fc))))

		# RNN
		self.n_layers = n_rnn
		self.rnn_cell = rnn_cell
		self.n_directions = n_directions
		self.n_ts = n_ts # temporal segment

		# Attention
		self.use_attn = use_attn
		self.n_attn = n_attn
		self.use_attn_frame = use_attn_frame

		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		if verbose:
			print(("""
				Initializing TSN with base model: {}.
				TSN Configurations:
				input_modality:     {}
				num_segments:       {}
				new_length:         {}
				""".format(base_model, self.modality, self.train_segments, self.new_length)))

		self._prepare_DA(num_class, base_model)

		if not self.before_softmax:
			self.softmax = nn.Softmax()

		self._enable_pbn = partial_bn
		if partial_bn:
			self.partialBN(True)

	# def feature_class(self, feature):
	# 	output = self.fc_feature_class0(feature)
	# 	output = self.relu(output)
	# 	output = self.fc_feature_class1(output)
	# 	return output

	def _prepare_DA(self, num_class, base_model): # convert the model to DA framework
		# if base_model == 'c3d': # C3D mode: in construction...
		# 	from C3D_model import C3D
		# 	model_test = C3D()
		# 	self.feature_dim = model_test.fc7.in_features
		# else:
		# 	model_test = getattr(torchvision.models, base_model)(True) # model_test is only used for getting the dim #
		# 	self.feature_dim = model_test.fc.in_features
		self.feature_dim = 2048

		std = 0.001
		feat_shared_dim = min(self.fc_dim, self.feature_dim) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
		## my code
		feat_shared_dim = self.output_size

		feat_frame_dim = feat_shared_dim

		self.relu = nn.ReLU(inplace=True)
		self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
		self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

		#------ frame-level layers (shared layers + source layers + domain layers) ------#
		# if self.add_fc < 1:
		# 	raise ValueError(Back.RED + 'add at least one fc layer')

		# 1. shared feature layers
		# self.fc_feature_shared_source = nn.Linear(self.feature_dim, 2048)

		self.fc_feature_shared_source = nn.Sequential(
			nn.Linear(self.feature_dim, self.output_size),
			nn.BatchNorm1d(self.output_size, affine=True),
			nn.ReLU(),
			nn.Linear(self.output_size, self.output_size),
		)
		# normal_(self.fc_feature_shared_source.weight, 0, std)
		# constant_(self.fc_feature_shared_source.bias, 0)

		if self.add_fc > 1:
			self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_2_source.weight, 0, std)
			constant_(self.fc_feature_shared_2_source.bias, 0)

		if self.add_fc > 2:
			self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_3_source.weight, 0, std)
			constant_(self.fc_feature_shared_3_source.bias, 0)

		# 2. frame-level feature layers
		# self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
		# normal_(self.fc_feature_source.weight, 0, std)
		# constant_(self.fc_feature_source.bias, 0)

		# 3. domain feature layers (frame-level)
		# self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
		# normal_(self.fc_feature_domain.weight, 0, std)
		# constant_(self.fc_feature_domain.bias, 0)

		# 4. classifiers (frame-level)
		self.fc_classifier_source = nn.Linear(feat_frame_dim * self.train_segments, num_class)
		normal_(self.fc_classifier_source.weight, 0, std)
		constant_(self.fc_classifier_source.bias, 0)

		# self.fc_classifier_source2 = nn.Linear(2048, num_class)
		# normal_(self.fc_classifier_source.weight, 0, std)
		# constant_(self.fc_classifier_source.bias, 0)

		self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
		normal_(self.fc_classifier_domain.weight, 0, std)
		constant_(self.fc_classifier_domain.bias, 0)

		if args.share_mapping == False:
			self.fc_feature_shared_target = nn.Sequential(
			   nn.Linear(self.feature_dim, self.output_size),
			   nn.BatchNorm1d(self.output_size, affine=True),
			   nn.ReLU(),
			   nn.Linear(self.output_size, self.output_size),
			)

		if self.share_params == 'N':
			self.fc_feature_shared_target = nn.Sequential(
			   nn.Linear(self.feature_dim, self.output_size),
			   nn.BatchNorm1d(self.output_size, affine=True),
			   nn.ReLU(),
			   nn.Linear(self.output_size, self.output_size),
			)
			if self.add_fc > 1:
				self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_2_target.weight, 0, std)
				constant_(self.fc_feature_shared_2_target.bias, 0)
			if self.add_fc > 2:
				self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_3_target.weight, 0, std)
				constant_(self.fc_feature_shared_3_target.bias, 0)

			self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
			normal_(self.fc_feature_target.weight, 0, std)
			constant_(self.fc_feature_target.bias, 0)
			self.fc_classifier_target = nn.Linear(feat_frame_dim*self.train_segments, num_class)
			normal_(self.fc_classifier_target.weight, 0, std)
			constant_(self.fc_classifier_target.bias, 0)


		#------ aggregate frame-based features (frame feature --> video feature) ------#
		if self.frame_aggregation == 'rnn': # 2. rnn
			self.hidden_dim = feat_frame_dim
			if self.rnn_cell == 'LSTM':
				self.rnn = nn.LSTM(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))
			elif self.rnn_cell == 'GRU':
				self.rnn = nn.GRU(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))

			# initialization
			for p in range(self.n_layers):
				kaiming_normal_(self.rnn.all_weights[p][0])
				kaiming_normal_(self.rnn.all_weights[p][1])

			self.bn_before_rnn = nn.BatchNorm2d(1)
			self.bn_after_rnn = nn.BatchNorm2d(1)

		elif self.frame_aggregation == 'trn': # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 512
			# self.TRN = TRNmodule.RelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		elif self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = self.trm_bottleneck ## testing
			# self.TRN = TRNmodule.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

		elif self.frame_aggregation == 'temconv': # 3. temconv

			self.tcl_3_1 = TCL(3, 1)
			self.tcl_5_1 = TCL(5, 1)
			self.bn_1_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_1_T = nn.BatchNorm1d(feat_frame_dim)

			self.tcl_3_2 = TCL(3, 1)
			self.tcl_5_2 = TCL(5, 2)
			self.bn_2_S = nn.BatchNorm1d(feat_frame_dim)
			self.bn_2_T = nn.BatchNorm1d(feat_frame_dim)

			self.conv_fusion = nn.Sequential(
				nn.Conv2d(2, 1, kernel_size=(1, 1), padding=(0, 0)),
				nn.ReLU(inplace=True),
			)

		# ------ video-level layers (source layers + domain layers) ------#
		if self.frame_aggregation == 'avgpool': # 1. avgpool
			feat_aggregated_dim = feat_shared_dim
		if 'trn' in self.frame_aggregation : # 4. trn
			feat_aggregated_dim = self.num_bottleneck
		elif self.frame_aggregation == 'rnn': # 2. rnn
			feat_aggregated_dim = self.hidden_dim
		elif self.frame_aggregation == 'temconv': # 3. temconv
			feat_aggregated_dim = feat_shared_dim

		feat_video_dim = feat_aggregated_dim

		# 1. source feature layers (video-level)
		self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_video_source.weight, 0, std)
		constant_(self.fc_feature_video_source.bias, 0)

		self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
		normal_(self.fc_feature_video_source_2.weight, 0, std)
		constant_(self.fc_feature_video_source_2.bias, 0)

		# 2. domain feature layers (video-level)
		# self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
		# normal_(self.fc_feature_domain_video.weight, 0, std)
		# constant_(self.fc_feature_domain_video.bias, 0)

		# 3. classifiers (video-level)
		self.fc_classifier_video_source = nn.Linear(self.output_size, num_class)
		normal_(self.fc_classifier_video_source.weight, 0, std)
		constant_(self.fc_classifier_video_source.bias, 0)

		if self.ens_DA == 'MCD':
			self.fc_classifier_video_source_2 = nn.Linear(self.output_size, num_class) # second classifier for self-ensembling
			normal_(self.fc_classifier_video_source_2.weight, 0, std)
			constant_(self.fc_classifier_video_source_2.bias, 0)

		self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
		normal_(self.fc_classifier_domain_video.weight, 0, std)
		constant_(self.fc_classifier_domain_video.bias, 0)

		# domain classifier for TRN-M
		if self.frame_aggregation == 'trn-m':
			self.relation_domain_classifier_all = nn.ModuleList()
			for i in range(self.train_segments-1):
				relation_domain_classifier = nn.Sequential(
					nn.Linear(feat_aggregated_dim, feat_video_dim),
					nn.ReLU(),
					nn.Linear(feat_video_dim, 2)
				)
				self.relation_domain_classifier_all += [relation_domain_classifier]

		if self.share_params == 'N':
			self.fc_feature_video_target = nn.Linear(feat_aggregated_dim, feat_video_dim)
			normal_(self.fc_feature_video_target.weight, 0, std)
			constant_(self.fc_feature_video_target.bias, 0)
			self.fc_feature_video_target_2 = nn.Linear(feat_video_dim, feat_video_dim)
			normal_(self.fc_feature_video_target_2.weight, 0, std)
			constant_(self.fc_feature_video_target_2.bias, 0)
			self.fc_classifier_video_target = nn.Linear(feat_video_dim, num_class)
			normal_(self.fc_classifier_video_target.weight, 0, std)
			constant_(self.fc_classifier_video_target.bias, 0)

		# BN for the above layers
		if self.use_bn != 'none':  # S & T: use AdaBN (ICLRW 2017) approach
			self.bn_source_video_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_T = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_S = nn.BatchNorm1d(feat_video_dim)
			self.bn_source_video_2_T = nn.BatchNorm1d(feat_video_dim)

		self.alpha = torch.ones(1)
		if self.use_bn == 'AutoDIAL':
			self.alpha = nn.Parameter(self.alpha)

		# ------ attention mechanism ------#
		# conventional attention
		if self.use_attn == 'general':
			self.attn_layer = nn.Sequential(
				nn.Linear(feat_aggregated_dim, feat_aggregated_dim),
				nn.Tanh(),
				nn.Linear(feat_aggregated_dim, 1)
				)


	def train(self, mode=True):
		# not necessary in our setting
		"""
		Override the default train() to freeze the BN parameters
		:return:
		"""
		super(VideoModel, self).train(mode)
		count = 0
		if self._enable_pbn:
			print("Freezing BatchNorm2D except the first one.")
			for m in self.base_model.modules():
				if isinstance(m, nn.BatchNorm2d):
					count += 1
					if count >= (2 if self._enable_pbn else 1):
						m.eval()

						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False

	def partialBN(self, enable):
		self._enable_pbn = enable

	def get_trans_attn(self, pred_domain):
		softmax = nn.Softmax(dim=1)
		logsoftmax = nn.LogSoftmax(dim=1)
		entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
		weights = 1 - entropy

		return weights

	def get_general_attn(self, feat):
		num_segments = feat.size()[1]
		feat = feat.view(-1, feat.size()[-1]) # reshape features: 128x4x256 --> (128x4)x256
		weights = self.attn_layer(feat) # e.g. (128x4)x1
		weights = weights.view(-1, num_segments, weights.size()[-1]) # reshape attention weights: (128x4)x1 --> 128x4x1
		weights = F.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

		return weights

	def get_attn_feat_frame(self, feat_fc, pred_domain): # not used for now
		if 1:
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, 1).repeat(1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 512)
		feat_fc = feat_fc.view(-1, feat_fc.shape[-1])
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn

	def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn, weights_attn[:,:,0]

	def aggregate_frames(self, feat_fc, num_segments, pred_domain):
		feat_fc_video = None
		if self.frame_aggregation == 'rnn':
			# 2. RNN
			feat_fc_video = feat_fc.view((-1, num_segments) + feat_fc.size()[-1:])  # reshape for RNN

			# temporal segments and pooling
			len_ts = round(num_segments/self.n_ts)
			num_extra_f = len_ts*self.n_ts-num_segments
			if num_extra_f < 0: # can remove last frame-level features
				feat_fc_video = feat_fc_video[:, :len_ts * self.n_ts, :]  # make the temporal length can be divided by n_ts (16 x 25 x 512 --> 16 x 24 x 512)
			elif num_extra_f > 0: # need to repeat last frame-level features
				feat_fc_video = torch.cat((feat_fc_video, feat_fc_video[:,-1:,:].repeat(1,num_extra_f,1)), 1) # make the temporal length can be divided by n_ts (16 x 5 x 512 --> 16 x 6 x 512)

			feat_fc_video = feat_fc_video.view(
				(-1, self.n_ts, len_ts) + feat_fc_video.size()[2:])  # 16 x 6 x 512 --> 16 x 3 x 2 x 512
			feat_fc_video = nn.MaxPool2d(kernel_size=(len_ts, 1))(
				feat_fc_video)  # 16 x 3 x 2 x 512 --> 16 x 3 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(2)  # 16 x 3 x 1 x 512 --> 16 x 3 x 512

			hidden_temp = torch.zeros(self.n_layers * self.n_directions, feat_fc_video.size(0),
									  self.hidden_dim // self.n_directions).cuda()

			if self.rnn_cell == 'LSTM':
				hidden_init = (hidden_temp, hidden_temp)
			elif self.rnn_cell == 'GRU':
				hidden_init = hidden_temp

			self.rnn.flatten_parameters()
			feat_fc_video, hidden_final = self.rnn(feat_fc_video, hidden_init)  # e.g. 16 x 25 x 512

			# get the last feature vector
			feat_fc_video = feat_fc_video[:, -1, :]

		else:
			# 1. averaging
			feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
			if self.use_attn == 'TransAttn': # get the attention weighting
				weights_attn = self.get_trans_attn(pred_domain)
				weights_attn = weights_attn.view(-1, 1, num_segments,1).repeat(1,1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
				feat_fc_video = (weights_attn+1) * feat_fc_video

			feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

		return feat_fc_video

	def final_output(self, pred, pred_video, num_segments, batch):
		if self.baseline_type == 'video': # YES
			base_out = pred_video
		else:
			base_out = pred

		if not self.before_softmax: # true, SKIP
			base_out = self.softmax(base_out)


		# output = (pred + pred_video)/2
		output = pred_video
		# output = pred

		if self.baseline_type == 'tsn': # no
			if self.reshape:
				base_out = base_out.view((-1, num_segments) + base_out.size()[1:]) # e.g. 16 x 3 x 12 (3 segments)

			output = base_out.mean(1) # e.g. 16 x 12

		return output

	def domain_classifier_frame(self, feat, beta):
		# feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
		# feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
		# feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
		# pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

		pred_fc_domain_frame = self.netD(feat, beta[2])

		# print('shape:',pred_fc_domain_frame.shape)
		return pred_fc_domain_frame

	def domain_classifier_video2(self, feat_video, beta = 1): # not good
		feat_video = feat_video.transpose(0,1)
		feat_video = GradReverse.apply(feat_video, 1)
		out, (h, c) = self.domain_cls(feat_video)
		out = out.transpose(0, 1)[:, -1, :]
		out = self.domain_cls_fc(out)
		out = F.relu(out)
		out = F.dropout(out, p= 0.5)
		out = self.domain_cls_fc2(out)
		# print('shape:',pred_fc_domain_video.shape)

		# feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
		# feat_fc_domain_video = self.relu(feat_fc_domain_video)
		# pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

		return out
	def domain_classifier_video_pre(self, feat_video, beta = 1):
		feat_video = feat_video.transpose(0, 1)
		feat_video = GradReverse.apply(feat_video, 1)
		out, (h, c) = self.domain_cls(feat_video)
		out = out.transpose(0, 1)[:, -1, :]
		return out
	def domain_classifier_video(self, feat_video, beta = 1):
		feat_video = GradReverse.apply(feat_video, 1)
		out = self.domain_classifier_video_pre(feat_video)
		# pred_fc_domain_video = self.domain_cls_fc(out)
		pred_fc_domain_video = self.netD2(out)
		# print('shape:',pred_fc_domain_video.shape)

		# feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
		# feat_fc_domain_video = self.relu(feat_fc_domain_video)
		# pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

		return pred_fc_domain_video

	def domain_classifier_relation(self, feat_relation, beta):
		# 128x4x256 --> (128x4)x2
		pred_fc_domain_relation_video = None
		for i in range(len(self.relation_domain_classifier_all)):
			feat_relation_single = feat_relation[:,i,:].squeeze(1) # 128x1x256 --> 128x256
			feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[0]) # the same beta for all relations (for now)

			pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
	
			if pred_fc_domain_relation_video is None:
				pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
			else:
				pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
		
		pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)

		return pred_fc_domain_relation_video
	#
	# def domainAlign(self, input_S, input_T, is_train, name_layer, alpha, num_segments, dim):
	# 	input_S = input_S.view((-1, dim, num_segments) + input_S.size()[-1:])  # reshape based on the segments (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
	# 	input_T = input_T.view((-1, dim, num_segments) + input_T.size()[-1:])  # reshape based on the segments
	#
	# 	# clamp alpha
	# 	alpha = max(alpha,0.5)
	#
	# 	# rearange source and target data
	# 	num_S_1 = int(round(input_S.size(0) * alpha))
	# 	num_S_2 = input_S.size(0) - num_S_1
	# 	num_T_1 = int(round(input_T.size(0) * alpha))
	# 	num_T_2 = input_T.size(0) - num_T_1
	#
	# 	if is_train and num_S_2 > 0 and num_T_2 > 0:
	# 		input_source = torch.cat((input_S[:num_S_1], input_T[-num_T_2:]), 0)
	# 		input_target = torch.cat((input_T[:num_T_1], input_S[-num_S_2:]), 0)
	# 	else:
	# 		input_source = input_S
	# 		input_target = input_T
	#
	# 	# adaptive BN
	# 	input_source = input_source.view((-1, ) + input_source.size()[-1:]) # reshape to feed BN (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
	# 	input_target = input_target.view((-1, ) + input_target.size()[-1:])
	#
	# 	if name_layer == 'shared':
	# 		input_source_bn = self.bn_shared_S(input_source)
	# 		input_target_bn = self.bn_shared_T(input_target)
	# 	elif 'trn' in name_layer:
	# 		input_source_bn = self.bn_trn_S(input_source)
	# 		input_target_bn = self.bn_trn_T(input_target)
	# 	elif name_layer == 'temconv_1':
	# 		input_source_bn = self.bn_1_S(input_source)
	# 		input_target_bn = self.bn_1_T(input_target)
	# 	elif name_layer == 'temconv_2':
	# 		input_source_bn = self.bn_2_S(input_source)
	# 		input_target_bn = self.bn_2_T(input_target)
	#
	# 	input_source_bn = input_source_bn.view((-1, dim, num_segments) + input_source_bn.size()[-1:])  # reshape back (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
	# 	input_target_bn = input_target_bn.view((-1, dim, num_segments) + input_target_bn.size()[-1:])  #
	#
	# 	# rearange back to the original order of source and target data (since target may be unlabeled)
	# 	if is_train and num_S_2 > 0 and num_T_2 > 0:
	# 		input_source_bn = torch.cat((input_source_bn[:num_S_1], input_target_bn[-num_S_2:]), 0)
	# 		input_target_bn = torch.cat((input_target_bn[:num_T_1], input_source_bn[-num_T_2:]), 0)
	#
	# 	# reshape for frame-level features
	# 	if name_layer == 'shared' or name_layer == 'trn_sum':
	# 		input_source_bn = input_source_bn.view((-1,) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
	# 		input_target_bn = input_target_bn.view((-1,) + input_target_bn.size()[-1:])
	# 	elif name_layer == 'trn':
	# 		input_source_bn = input_source_bn.view((-1, num_segments) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
	# 		input_target_bn = input_target_bn.view((-1, num_segments) + input_target_bn.size()[-1:])
	#
	# 	return input_source_bn, input_target_bn

	def normal(self, mu, sigma):
		''' Gaussian PDF using keras' backend abstraction '''

		def f(y):
			pdf = y - mu
			pdf = pdf / sigma
			pdf = - torch.square(pdf) / 2.
			return torch.exp(pdf) / sigma

		return f
	def means_adjust(self,means):
		means = means
		# means = means + 1000
		return means
	# def sigma_adjust(self, sigma):

	def compare_feat_avg(self, feat_stdw, batch, segments):
		feat_stdw = feat_stdw.view(batch, segments, -1).unsqueeze(1)
		# print(feat_target_sdtw.size())
		feat_stdw = feat_stdw.expand(batch, self.num_class, segments, -1)
		feat_stdw = feat_stdw.reshape(batch * self.num_class, segments, -1)  # batch x classes, frames, features
		# print(feat_target_sdtw.size()
		avg_feat = self.avg[:self.num_class].clone().detach()
		avg_feat = avg_feat.unsqueeze(0)
		avg_feat = avg_feat.expand(batch, self.num_class, segments, -1)
		avg_feat = avg_feat.reshape(batch * self.num_class, segments, -1)  # same above
		sdtw_out = self.sdtw(avg_feat, feat_stdw)
		sdtw_out = sdtw_out.view(batch, -1)
		# return -torch.log(sdtw_out)
		return -sdtw_out

	def classifier_source(self, feat):
		output = self.fc_classifier_source(feat)
		# output = self.fc_classifier_source2(feat)
		return output

	def tanh_basis(self, target_len, l = 300, ran = 3, p_a = 0.6, chan_par_range = 3, channel_num = 3):
		x = torch.linspace(-ran, ran, l)
		x = x.expand(channel_num, l)
		channels = torch.linspace(-chan_par_range, chan_par_range, channel_num)
		channels = channels.expand(l, channel_num).transpose(0, 1)

		out = np.tanh(p_a * (x - channels))
		mi = torch.min(out, 1)[0].view(channel_num, 1)
		ma = torch.max(out, 1)[0].view(channel_num, 1)
		out = (out - mi) / (ma - mi) * (target_len - 1)
		return out.cuda() # (channel_num, l)

	def poly_basis(self, target_len, l = 300, channel_num = 3, chan_par_range = 0.4):
		x = torch.linspace(0, 1, l)
		x = x.expand(channel_num, l)
		channels = torch.linspace(-chan_par_range, chan_par_range, channel_num)
		channels = torch.pow(10, channels)
		channels = channels.expand(l, channel_num).transpose(0, 1)
		out = torch.pow(x, channels)
		out = (target_len - 1) * out
		return out.cuda() # (channel_num, l)
	def gen_basis(self, l, gamma, f_l = 3):
		# a = np.zeros(gamma)
		# b = np.array([0.2, 0.6, 0.2])  # filter
		# out = []
		# for i in range(l):
		# 	index = int((i * (gamma - 1) / (l - 1))) if f_l % 2 == 1 else int((i * gamma / (l - 1)))  # control diagonal
		# 	row = np.copy(a)
		# 	row[0:f_l] = b
		# 	row = np.roll(row, index - int(f_l / 2))
		# 	if index < int(f_l / 2):  # tail to zero
		# 		for j in range((int(f_l / 2) - index)):
		# 			row[-(j + 1)] = 0
		# 	if index > gamma - (f_l / 2):  # head to zero
		# 		for j in range(math.ceil(index - gamma + (f_l / 2))):
		# 			row[j] = 0
		# 	out.append(row)
		# out = np.array(out)
		# out = torch.from_numpy(out).transpose(0, 1).float()
		out = np.zeros([gamma, l])
		for i in range(gamma):
			for j in range(l):
				out[i, j] = np.cos(j * (i * np.pi / 6) / l)
		return torch.from_numpy(out).float().cuda()

	def lstm_vid_classifier(self, feat):
		out, (h,c) = self.video_cls(feat)
		out = out.transpose(0, 1)[:, -1, :]
		out = self.video_cls_fc(out)
		out = F.relu(out)
		# out = F.dropout(out, p= 0.5)
		out = self.video_cls_fc2(out)
		return out.contiguous()

	def gen_H(self,feat):
		out, (h, c)  = self.H_gen(feat)
		out = torch.sum(out, 0)/feat.shape[0]
		# out = self.H_gen_fc(out)
		out = out.transpose(0,1)
		return out

	def gen_Q(self,feat):
		#fc
		# shape = feat.shape
		# feat = feat.view(-1, shape[-1])
		# out = self.Q_gen(feat)
		# out = out.view(shape[0],shape[1], -1)

		feat = feat.transpose(0, 1)
		out, (h, c) = self.Q_gen(feat)
		out = out.transpose(0, 1)

		return out

	def cdan(self, feat, pred, domain):
		'''
		feat: [batch * sequence_len, d] for frame
			  [batch , d] for video
		pred: [batch * sequence_len, num_class] for frame
			  [batch , num_class] for video
		'''
		features = feat.view(-1, self.output_size)
		pred = pred.view(-1, self.num_class)
		# return
		softmax_output = F.softmax(pred, dim=1)

		#
		# target_softmax_output = F.softmax(pred, dim=1)  # .detach()
		# features = torch.cat([source_features, target_features], dim=0)
		# softmax_output = torch.cat([source_softmax_output, target_softmax_output], dim=0)
		#
		op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
		features = op_out.view(-1, softmax_output.size(1) * features.size(1))

		entropy = Entropy(softmax_output)
		entropy = GradReverse.apply(entropy, 1)
		# entropy.register_hook(grl_hook(alpha))
		entropy = 1.0 + torch.exp(-entropy)

		# source_mask = torch.ones_like(entropy)
		# source_mask[features.size(0) // 2:] = 0
		# source_weight = entropy * source_mask
		# target_mask = torch.ones_like(entropy)
		# target_mask[0:features.size(0) // 2] = 0
		# target_weight = entropy * target_mask
		weight = entropy/ torch.sum(entropy).detach().item()
		# weight = source_weight / torch.sum(source_weight).detach().item() + \
		# 		 target_weight / torch.sum(target_weight).detach().item()
		if domain == 'source':
			dc_labels = torch.from_numpy(np.array([[1]] * features.size(0))).float().cuda()
		elif domain == 'target':
			dc_labels = torch.from_numpy(np.array([[0]] * features.size(0))).float().cuda()
		G_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(self.netD(features), dc_labels)) / torch.sum(weight).detach().item()
		return G_loss

	def forward(self, input_source, source_label, input_target, beta, mu, is_train, reverse, batchsize = 0, dummy = False):

		batch_source = input_source.size()[0]
		batch_target = input_target.size()[0]
		########################===================== attention =======
		if args.use_attention:
			# [b, l, 7,7,2048] -> [ b, l, 49, 2048]
			# [b, l, 2048, 7, 7] -> [ b, l,  2048, 49] this ->[ b, l, 49, 2048]
			input_source = input_source.view(batch_source, input_source.shape[1], 2048, 49).transpose(-1,-2)
			input_target = input_target.view(batch_target, input_target.shape[1], 2048, 49).transpose(-1,-2)

			# input_source = self.fc_tr(input_source)
			# input_target = self.fc_tr(input_target)
			input_source += self.enc_pos_encoding.cuda()
			input_target += self.enc_pos_encoding.cuda()


			# [ b, l, 49, d]-> [ b * l, 49, d] -> [49, b* l , d]
			input_source = input_source.view(-1, input_source.shape[2], input_source.shape[-1])
			input_source = self.transformer.encoder(input_source.transpose(0, 1))  # output [49, b* l , d]
			input_target = input_target.view(-1, input_target.shape[2], input_target.shape[-1])
			input_target = self.transformer.encoder(input_target.transpose(0, 1))  # output [49, b* l , d]


			input_source = input_source.transpose(0, 1).view(batch_source, -1, 49, input_source.shape[-1]).transpose(2, 3)
			input_source = input_source.view(input_source.shape[0], input_source.shape[1], input_source.shape[2], 7, 7)
			input_source = self.pool(input_source).squeeze(-1).squeeze(-1)
			input_target = input_target.transpose(0, 1).view(batch_source, -1, 49, input_target.shape[-1]).transpose(2, 3)
			input_target = input_target.view(input_target.shape[0], input_target.shape[1], input_target.shape[2], 7, 7)
			input_target = self.pool(input_target).squeeze(-1).squeeze(-1)

		else:
			input_source = input_source.view(input_source.shape[0],input_source.shape[1],self.output_size,7,7)
			input_target = input_target.view(input_target.shape[0],input_target.shape[1],self.output_size,7,7)

			input_source = self.pool(input_source).squeeze(-1).squeeze(-1)

			input_target = self.pool(input_target).squeeze(-1).squeeze(-1)
		###########=============== end ========
		flag = 0 # if 1, then rmdn
		source_means = torch.zeros(1,5).cuda()
		cdan_loss = torch.zeros(1).cuda()
		# batch_source = self.feature_gmm_source.size()[0]
		# batch_target = self.feature_gmm_target.size()[0]

		# num_segments = self.train_segments if is_train else self.val_segments
		num_segments = input_source.size()[1]
		# sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
		sample_len = self.new_length
		feat_all_source = []
		feat_all_target = []
		pred_domain_all_source = []
		pred_domain_all_target = []

		# input_data is a list of tensors --> need to do pre-processing

		feat_base_source0 = input_source.view(-1, input_source.size()[-1]) # e.g. 256 x 25 x 2048 --> 6400 x 2048
		feat_base_target0 = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
		if flag:
			feat_base_source = self.feature_gmm_source.view(-1, self.feature_gmm_source.size()[-1]) # e.g. 256 x 25 x 2048 --> 6400 x 2048
			feat_base_target = self.feature_gmm_target.view(-1, self.feature_gmm_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
		else:
			feat_base_source = input_source.view(-1, input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
			feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048


		#=== shared layers ===#
		# need to separate BN for source & target ==> otherwise easy to overfit to source data
		# ====mapping ====== #
		# raise ValueError(Back.RED + 'not enough fc layer')

		feat_fc_source = self.fc_feature_shared_source(feat_base_source)
		feat_fc_target = self.fc_feature_shared_target(feat_base_target) if args.share_mapping == False else self.fc_feature_shared_source(feat_base_target)

		# adaptive BN

		##== rmdn correction ===##
		loss_feature = torch.zeros(1).cuda()


		# print(loss_feature)

		# feat_fc = self.dropout_i(feat_fc)
		feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
		feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

			#############=============== CORE PART =======  ############

		# basis = self.gen_basis(l = self.train_segments, gamma= self.gamma) # train_segments when target.
		# print('basis:',basis.shape)
		norm_loss = torch.zeros(1).cuda()

		feat_fc_source = feat_fc_source.view(batch_source, self.train_segments, -1)



		if args.method == 'path_gen':
			#path gen for source
			feat_fc_source = feat_fc_source.view(batch_source, self.train_segments, -1)
			feat_fc_source2 = feat_fc_source.transpose(0, 1)
			W_s, (h, c) = self.path_gen(feat_fc_source2)
			W_s = W_s.transpose(0, 1)  # batch, original_size(target), target_size(source)
			feat_fc_source3 = feat_fc_source.transpose(1, 2)
			feat_fc_source = torch.matmul(feat_fc_source3, W_s).transpose(1, 2).contiguous()

			# path gen for target
			feat_fc_target = feat_fc_target.view(batch_target, self.val_segments, -1)
			feat_fc_target2 = feat_fc_target.transpose(0, 1)
			W_t, (h, c) = self.path_gen(feat_fc_target2)
			W_t = W_t.transpose(0, 1)  # batch, original_size(target), target_size(source)
			feat_fc_target3 = feat_fc_target.transpose(1, 2)
			feat_fc_target = torch.matmul(feat_fc_target3, W_t).transpose(1, 2).contiguous()
		elif args.method == 'QB':

			basis = self.basis.cuda()
			# gen H by source
			# basis = self.gen_H(feat_fc_source)
			# np.save('H.npy', basis.detach().cpu().numpy())

			# norm_loss  += torch.linalg.norm(Q_s)
			# u,s_s,v = torch.svd(Q_s)
			# norm_loss += torch.sum(s_s)
			# q b for source
			Q_s = self.gen_Q(feat_fc_source) # batch, original_size(target), target_size(source)
			W_s = torch.matmul(Q_s, basis)
			feat_fc_source3 = feat_fc_source.transpose(1, 2)
			feat_fc_source = torch.matmul(feat_fc_source3, W_s).transpose(1, 2).contiguous()

			# Q B for target
			feat_fc_target = feat_fc_target.view(batch_target, self.val_segments, -1)
			Q_t = self.gen_Q(feat_fc_target)# batch, original_size(target), target_size(source)
			W_t = torch.matmul(Q_t,basis)
			feat_fc_target3 = feat_fc_target.transpose(1, 2)
			feat_fc_target = torch.matmul(feat_fc_target3, W_t).transpose(1, 2).contiguous()


		# norm_loss += torch.linalg.norm(Q_t)
		# u, s_t, v = torch.svd(Q_t)
		# norm_loss += torch.sum(s_t)

		# temporal attention
		#[frame_len, batch, d]
		# feat_fc_source += self.enc_pos_encoding2.cuda()
		# feat_fc_source = self.transformer2.encoder(feat_fc_source.transpose(0,1)).transpose(0,1).contiguous()
		# feat_fc_target += self.enc_pos_encoding2.cuda()
		# feat_fc_target = self.transformer2.encoder(feat_fc_target.transpose(0,1)).transpose(0,1).contiguous()

		# class pred - frame
		feat_fc_source_temp = feat_fc_source.view(batch_source, -1)
		feat_fc_target_temp = feat_fc_target.view(batch_target, -1)
		pred_fc_source = self.classifier_source(feat_fc_source_temp)
		pred_fc_target = self.classifier_source(feat_fc_target_temp)

		# class pred -lstm
		pred_lstm_source = self.lstm_vid_classifier(feat_fc_source.view(batch_source, self.train_segments, -1).transpose(0, 1))
		pred_lstm_target = self.lstm_vid_classifier(feat_fc_target.view(batch_target, self.val_segments, -1).transpose(0, 1))

		# domain pred - video (by lstm)
		if args.use_cdan:
			s_cdan_loss_v = self.cdan(self.domain_classifier_video_pre(feat_fc_source), pred_lstm_source, 'source')
			t_cdan_loss_v = self.cdan(self.domain_classifier_video_pre(feat_fc_target), pred_lstm_target, 'target')
			# fill the blank
			pred_fc_domain_video_source = torch.randn(batch_source, 2).cuda()
			pred_fc_domain_video_target = torch.randn(batch_target, 2).cuda()
		else:
			pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_source)
			pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_target)

		# domain pred - frame -> frame for attention only
		if args.use_cdan:
			s_cdan_loss_f = self.cdan(feat_fc_source, pred_fc_source.expand(self.train_segments,batch_source,self.num_class).transpose(0,1).contiguous(), 'source')
			t_cdan_loss_f = self.cdan(feat_fc_target, pred_fc_target.expand(self.val_segments,batch_target,self.num_class).transpose(0,1).contiguous(), 'target')

			cdan_loss = s_cdan_loss_v + t_cdan_loss_v + s_cdan_loss_f + t_cdan_loss_f
			# fill the blank
			pred_fc_domain_frame_source = torch.randn(batch_source* self.train_segments, 2).cuda()
			pred_fc_domain_frame_target = torch.randn(batch_target* self.val_segments, 2).cuda()
		else:
			feat_fc_source = feat_fc_source.view(-1, input_source.size()[-1])
			feat_fc_target = feat_fc_target.view(-1, input_target.size()[-1])
			pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)
			pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)






		## ===== calculate the sdtw loss##

		# # print(feat_fc_source.size())
		avg_loss = []
		# loss_sdtw = 0
		# if not dummy:
		# 	feat_temp_source = feat_fc_source.view(batch_source, num_segments, -1)
		# 	feat_temp_target = feat_fc_source.view(batch_target, num_segments, -1)
		#
		# 	loss_sdtw = self.sdtw(feat_temp_source, feat_temp_target) / 2048
		# 	loss_sdtw = loss_sdtw.sum()
		#
		#
		# else:
		# 	pass
		# 	for i in range(len(source_label)):
		# 		if source_label[i]== self.num_class:
		# 			l = i
		# 			break
		# 		elif i == len(source_label)-1:
		# 			l = i + 1
		# 	print(len(source_label), i, l)
		# 	print(feat_fc_source.size())
		#
		# 	feat_temp_source = feat_fc_source[:l*num_segments]
		# 	feat_temp_source = feat_temp_source.view(l, num_segments, -1)
		# 	feat_temp_target = feat_fc_target[:l * num_segments].view(l, num_segments, -1)
		#
		# 	loss_sdtw = self.sdtw(feat_temp_source, feat_temp_target) / 2048
		# 	loss_sdtw = loss_sdtw.sum()
		#
		# # print('sdtw:',loss_sdtw)
		loss_sdtw = torch.zeros(1).cuda()








		########### ==================== adversarial branch (frame-level) ============================#

		pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))


		#=== source layers (frame-level) ===# # this part is actually skipped

		# average
		# feat_fc_source = feat_fc_source.view(batch_source, num_segments, -1)
		# feat_fc_target = feat_fc_target.view(batch_target, num_segments, -1)
		# feat_fc_source_temp = torch.sum(feat_fc_source, dim=1)/num_segments
		# feat_fc_target_temp = torch.sum(feat_fc_target, dim=1)/num_segments















		### aggregate the frame-based features to video-based features ###

		if self.if_trm:
			feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			# try to reduce the calculation
			# feat_fc_video_source = feat_fc_video_source[:,:self.train_segments,:]
			# feat_fc_video_target = feat_fc_video_target[:, :self.val_segments, :]
			#
			# feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
			# feat_fc_video_relation_target = self.TRN(feat_fc_video_target)
			# print('size:', feat_fc_video_relation_source.shape,)
			feat_fc_video_relation_source = torch.randn(feat_fc_video_source.shape[0],feat_fc_video_source.shape[1]-1,self.num_bottleneck).cuda()
			feat_fc_video_relation_target = torch.randn(feat_fc_video_target.shape[0],feat_fc_video_target.shape[1]-1,self.num_bottleneck).cuda()
			# adversarial branch
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

			# transferable attention

			attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
			attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error

			# sum up relation features (ignore 1-relation)
			# print('source:',feat_fc_source.shape)
			feat_fc_video_source = torch.sum(feat_fc_video_source, 1)
			feat_fc_video_target = torch.sum(feat_fc_video_target, 1)
		else:
			### testing code here
			feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			# feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
			# feat_fc_video_relation_target = self.TRN(feat_fc_video_target)
			feat_fc_video_source = feat_fc_video_source.view(feat_fc_video_source.shape[0],-1)  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_video_target.view(feat_fc_video_target.shape[0],-1)  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			feat_fc_video_source = self.fc_layer(feat_fc_video_source)
			feat_fc_video_target = self.fc_layer(feat_fc_video_target)
			attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
			attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)
			### tesing code finish


		# print('shape6:', feat_fc_video_source.shape) # 128 256

		if self.baseline_type == 'video':
			feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
			feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))

		#=== source layers (video-level) ===#
		feat_fc_video_source = self.dropout_v(feat_fc_video_source)
		feat_fc_video_target = self.dropout_v(feat_fc_video_target)

		if reverse:
			feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
			feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

		pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
		pred_fc_video_target = self.fc_classifier_video_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)
		# print('size:',pred_fc_video_source.shape)
		# print('shape7:', pred_fc_video_source.shape, 'zhi kan zhe ge pred')
		if self.baseline_type == 'video': # only store the prediction from classifier 1 (for now)
			feat_all_source.append(pred_fc_video_source.view((batch_source,) + pred_fc_video_source.size()[-1:]))
			feat_all_target.append(pred_fc_video_target.view((batch_target,) + pred_fc_video_target.size()[-1:]))

		#=== adversarial branch (video-level) ===#
		# pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
		# pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)
		# print('size:',pred_fc_domain_video_source.shape)
		pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

		# video relation-based discriminator
		if self.frame_aggregation == 'trn-m':
			num_relation = feat_fc_video_relation_source.size()[1]
			# pred_domain_all_source.append(torch.zeros((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]).cuda())
			# pred_domain_all_target.append(torch.zeros((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]).cuda())
			pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
			pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
		else:
			pred_domain_all_source.append(pred_fc_domain_video_source) # if not trn-m, add dummy tensors for relation features
			pred_domain_all_target.append(pred_fc_domain_video_target)
		# print('3rd shape:', torch.zeros((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]).size())

		#=== final output ===#
		# pred_fc_video_source , pred_fc_video_target
		# pred_lstm_source , pred_lstm_target
		output_source = self.final_output(pred_fc_source, pred_lstm_source, num_segments, batch_source) # select output from frame or video prediction
		output_target = self.final_output(pred_fc_target, pred_lstm_target, num_segments, batch_target)



		output_source_2 = output_source
		output_target_2 = output_target

		if self.ens_DA == 'MCD': # NO
			pred_fc_video_source_2 = self.fc_classifier_video_source_2(feat_fc_video_source)
			pred_fc_video_target_2 = self.fc_classifier_video_target_2(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source_2(feat_fc_video_target)
			output_source_2 = self.final_output(pred_fc_source, pred_fc_video_source_2, num_segments, batch_source)
			output_target_2 = self.final_output(pred_fc_target, pred_fc_video_target_2, num_segments, batch_target)

		return attn_relation_source, output_source, output_source_2, pred_domain_all_source[::-1], feat_all_source[::-1], attn_relation_target, output_target, output_target_2, pred_domain_all_target[::-1], feat_all_target[::-1], feat_base_source0,feat_base_target0, feat_base_source, feat_base_target, source_means, avg_loss, cdan_loss, norm_loss # lreverse the order of feature ist due to some multi-gpu issues
