import logging
import torch.nn as nn
from fastai.vision import *

from modules.model import _default_tfmer_cfg
from modules.model import Model
from modules.transformer import (PositionalEncoding, 
                                 TransformerDecoder,
                                 TransformerDecoderLayer)


class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_language_activation, _default_tfmer_cfg['activation'])
        num_layers = ifnone(config.model_language_num_layers, 4)
        self.d_model = d_model
        self.detach = ifnone(config.model_language_detach, True)
        self.use_self_attn = ifnone(config.model_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn, debug=self.debug)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.model_language_checkpoint is not None:
            logging.info(f'Read language model from {config.model_language_checkpoint}.')
            self.load(config.model_language_checkpoint)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        
        #print("tokens:",tokens.size())
        
        if self.detach: tokens = tokens.detach()        # tokens:[24, 26, 7935]  
                                                        # 24是样本个数，整体train的时候batch size为96；26是最大长度；7935是字符类数
        embed = self.proj(tokens)       # (N, T, E)     # embed:[24, 26, 512]    # embedding词嵌入编码
        
        embed = embed.permute(1, 0, 2)  # (T, N, E)     # embed:[26, 24, 512]
        embed = self.token_encoder(embed)  # (T, N, E)  # 加上位置编码 embed:[26, 24, 512]
        padding_mask = self._get_padding_mask(lengths, self.max_length)    #lengths:[24] padding_mask:[24, 26]

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)    #创建对角矩阵的mask[26, 26]，对角线上均为-inf
        
        output = self.model(qeury, embed,              # TransformerDecoder
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)   # output:[26, 24, 512]
        output = output.permute(1, 0, 2)               # (N, T, E)   # output:[24, 26, 512]
        logits = self.cls(output)       # (N, T, C)    # logits=[24, 26, 7935]
        pt_lengths = self._get_length(logits)          # pt_lengths=[24]

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
