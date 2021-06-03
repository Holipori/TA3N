
from torch.nn import Linear, LayerNorm, TransformerEncoder, MultiheadAttention
import torch.nn as nn

class myEncoderlayer(nn.TransformerEncoderLayer):
    __constants__ = ['norm']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(myEncoderlayer, self).__init__(d_model, nhead,dim_feedforward=2048, dropout=0.1, activation="relu")
        self.attn = 'None'

    def __setstate__(self, state):
        super(myEncoderlayer, self).__setstate__(state)
    def forward(self, src , src_mask = None, src_key_padding_mask = None) :
        src2 , self.attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class mytrans(nn.Transformer):
    def __init__(self, d_model = 512, nhead = 8, num_encoder_layers = 6,
                 num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0.1,
                 activation = "relu", custom_encoder = None, custom_decoder = None):
        super(mytrans, self).__init__()

        encoder_layer = myEncoderlayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)



        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
    def forward(self, src , tgt , src_mask = None, tgt_mask = None,
                memory_mask  = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):
        super(mytrans, self).forward(src , tgt, src_mask = None, tgt_mask = None,
                memory_mask  = None, src_key_padding_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None)
