import torch
import torch.nn as nn
from fairseq import checkpoint_utils, tasks, utils
from avhubert.hubert_asr import AVHubertSeq2SeqConfig
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II, MISSING
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
import sys, logging
from avhubert.decoder import TransformerDecoder
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
import contextlib
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple, Any
import os

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

from src.model import HubertEncoderWrapper, encodercfg

@dataclass
class LMDecoderConfig(AVHubertSeq2SeqConfig):
    lmd_embedding: int = field(default=1002)
    lmd_embedding_dim: int = field(default=768)
    
    
@register_model("lmdecoder", dataclass=LMDecoderConfig)
class lmdecoder(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.embedding = nn.Embedding(num_embeddings=cfg.lmd_embedding, embedding_dim=cfg.lmd_embedding_dim)

    
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        transformer_enc_cfg = encodercfg
        transformer_enc_cfg.encoder_layers = 4
        transformer_enc_cfg.encoder_embed_dim = cfg.lmd_embedding_dim
        
        if transformer_enc_cfg.encoder_embed_dim == 1024:
            transformer_enc_cfg.encoder_ffn_embed_dim = 4096
            transformer_enc_cfg.encoder_attention_heads = 16
            
        encoder_ = TransformerEncoder(transformer_enc_cfg)
        encoder = HubertEncoderWrapper(encoder_)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)

        return cls(encoder, decoder, tgt_dict, cfg)

    def forward(self, **kwargs):
        B, T_token = kwargs['source']['units'].size()
        speech_unit = self.embedding(kwargs['source']['units'])
        output = self.encoder.cam_forward(speech_unit, kwargs['padding_mask'])
        decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
        
        return decoder_out

    def extract_encoder_output(self, net_input):
        B, T_token = net_input['source']['units'].size()
        speech_unit = self.embedding(net_input['source']['units'])
        output = self.encoder.cam_forward(speech_unit, net_input['padding_mask'])
        
        return output

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def load_state_dict(self, state, **kwargs):
        super().load_state_dict(state, strict=False)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
