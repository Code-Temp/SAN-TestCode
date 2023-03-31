import torch
import torch.nn as nn
from fastai.vision import *

from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_radical_language import BCNRadicalLanguage
from .model_alignment import BaseAlignment,BaseAlignmentRadical


class ABINetIterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        
        #!
        self.max_length_radical = config.dataset_max_length_radical + 1  # additional stop token
        self.language_radical = BCNRadicalLanguage(config)
        self.alignment_radical = BaseAlignmentRadical(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        #a_res = v_res
        all_l_res, all_a_res = [], []
        #!
        all_l_res_radical, all_a_res_radical = [], []
        
        for _ in range(self.iter_size):
            tokens = torch.softmax(v_res['logits'], dim=-1)
            lengths = v_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
            
        #!
        for _ in range(self.iter_size):
            tokens_radical = torch.softmax(v_res['logits_radical'], dim=-1)
            lengths_radical = v_res['pt_lengths_radical']
            lengths_radical.clamp_(2, self.max_length_radical)  # TODO:move to langauge model
            
            l_res_radical = self.language_radical(tokens_radical, lengths_radical)
            all_l_res_radical.append(l_res_radical)
            
            a_res_radical = self.alignment_radical(l_res_radical['feature'], v_res['feature_radical'])
            all_a_res_radical.append(a_res_radical)
            
        if self.training:
            return all_a_res, all_l_res, all_a_res_radical, all_l_res_radical, v_res
        else:
            return a_res, all_l_res[-1], v_res, a_res_radical  #!
