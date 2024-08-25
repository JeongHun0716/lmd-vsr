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

from avhubert.hubert import AVHubertConfig

@dataclass
class encodercfg(AVHubertConfig):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=4, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )


class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward_(self, source, padding_mask, **kwargs):
        src ={}
        src['video'] = source
        src['audio'] = None
        w2v_args = {
            "source": src,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def forward(self, source, padding_mask, **kwargs):
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
            }

            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
            x = x.transpose(0,1)

            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }

    def cam_forward(self, source, padding_mask):

            x, _ = self.w2v_model(
                source,
                padding_mask=padding_mask,
                layer=None
            ) 
            x = x.transpose(0, 1)
            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }        

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out
    
@dataclass
class LMD_VSRConfig(AVHubertSeq2SeqConfig):
    lmd_embedding: int = field(default=1002)
    lmd_embedding_dim: int = field(default=768)
    lmd_cross_att_head: int = field(default=1)
    lmdecoder_pth: str = field(default='/path/to/lmdecoder')
    
    
@register_model("lmd_vsr", dataclass=LMD_VSRConfig)
class lmd_vsr(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.audio_encoder = audio_encoder
        self.embedding = nn.Embedding(num_embeddings=cfg.lmd_embedding, embedding_dim=cfg.lmd_embedding_dim)
        self.ctc_proj = nn.Linear(cfg.lmd_embedding_dim, 1000)
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=cfg.lmd_embedding_dim, num_heads=cfg.lmd_cross_att_head)
        self.layer_norm0 = nn.LayerNorm(cfg.lmd_embedding_dim)
         

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        src_pth = os.path.dirname(os.path.realpath(__file__))
        avhubert_pth = f'{src_pth}/pretrained_models/encoder/base_vox_iter5.pt'
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                avhubert_pth, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )


        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()
        
        transformer_enc_cfg = encodercfg
        transformer_enc_cfg.encoder_layers = 4
        audio_encoder_ = TransformerEncoder(transformer_enc_cfg)
        audio_encoder = HubertEncoderWrapper(audio_encoder_)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)

        return cls(encoder, decoder, tgt_dict, cfg, audio_encoder)

    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        T, B, C = output['encoder_out'].size()
        x = torch.tensor(list(range(1000))) + 1
        x = repeat(x, 't -> t b', b=B)
        speech_units = self.embedding(x.cuda()).detach() # 200 512 B
        x, _ = self.mha0(query=output['encoder_out'], key=speech_units, value=speech_units) # T B D
        x = self.layer_norm0(x)

        output = self.audio_encoder.cam_forward(x.transpose(0,1), output['padding_mask'])

        decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)

        return decoder_out, output

    def extract_encoder_output(self, net_input):
        output = self.encoder(net_input['source'],net_input['padding_mask'])
        T, B, C = output['encoder_out'].size()
        x = torch.tensor(list(range(1000))) + 1
        x = repeat(x, 't -> t b', b=B)
        speech_units = self.embedding(x.cuda())#.detach() # 200 512 B
        x, _ = self.mha0(query=output['encoder_out'], key=speech_units, value=speech_units) # T B D
        x = self.layer_norm0(x)
        
        output = self.audio_encoder.cam_forward(x.transpose(0,1), output['padding_mask'])

        return output

    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def load_state_dict(self, state, **kwargs):
        super().load_state_dict(state, strict=False)


@register_model("lmd_vsr_es", dataclass=LMD_VSRConfig)
class lmd_vsr_es(lmd_vsr):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder, tgt_dict, cfg, audio_encoder)
        self.layer_norm1 = nn.LayerNorm(768)
        
        ## LMDecoder ##
        src_pth = os.path.dirname(os.path.realpath(__file__))
        lmdecoder_pth = f'{src_pth}/pretrained_models/lmdecoder/es/best_ckpt.pt'
        
        lmdecoder_weights = torch.load(lmdecoder_pth)
        self.load_state_dict(lmdecoder_weights, strcit=False)
        
    def extract_encoder_output(self, net_input):
        output = self.encoder(net_input['source'],net_input['padding_mask'])
        T, B, C = output['encoder_out'].size()
        x = torch.tensor(list(range(1000))) + 1
        x = repeat(x, 't -> t b', b=B)
        speech_units = self.embedding(x.cuda())#.detach() # 200 512 B
        x, _ = self.mha0(query=output['encoder_out'], key=speech_units, value=speech_units) # T B D
        x = self.layer_norm0(x)
        
        output_ = self.audio_encoder.cam_forward(x.transpose(0,1), output['padding_mask'])
        output['encoder_out'] = self.layer_norm1(output_['encoder_out'] + output['encoder_out'])

        return output
    
@register_model("lmd_vsr_fr", dataclass=LMD_VSRConfig)
class lmd_vsr_es(lmd_vsr):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder, tgt_dict, cfg, audio_encoder)
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8)
        
        
@register_model("lmd_vsr_it", dataclass=LMD_VSRConfig)
class lmd_vsr_es(lmd_vsr):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder, tgt_dict, cfg, audio_encoder)
        
@register_model("lmd_vsr_pt", dataclass=LMD_VSRConfig)
class lmd_vsr_es(lmd_vsr):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder, tgt_dict, cfg, audio_encoder)


@register_model("lmd_vsr_en", dataclass=LMD_VSRConfig)
class lmd_vsr_en(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg, audio_encoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.audio_encoder = audio_encoder
        self.embedding = nn.Embedding(num_embeddings=1002, embedding_dim=1024)
        self.ctc_proj = nn.Linear(1024, 1000)
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=1024, num_heads=1)
        self.layer_norm0 = nn.LayerNorm(1024)
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        src_pth = os.path.dirname(os.path.realpath(__file__))
        avhubert_pth = f'{src_pth}/pretrained_models/encoder/large_vox_iter5.pt'
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                avhubert_pth, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        transformer_enc_cfg = encodercfg
        transformer_enc_cfg.encoder_layers = 4
        transformer_enc_cfg.encoder_embed_dim = 1024
        transformer_enc_cfg.encoder_ffn_embed_dim = 4096
        transformer_enc_cfg.encoder_attention_heads = 8
        audio_encoder_ = TransformerEncoder(transformer_enc_cfg)
        audio_encoder = HubertEncoderWrapper(audio_encoder_)
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)

        return cls(encoder, decoder, tgt_dict, cfg, audio_encoder)

    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        T, B, C = output['encoder_out'].size()
        x = torch.tensor(list(range(1000))) + 1
        x = repeat(x, 't -> t b', b=B)
        speech_units = self.embedding(x.cuda()).detach() # 200 512 B
        x, _ = self.mha0(query=output['encoder_out'], key=speech_units, value=speech_units) # T B D
        x = self.layer_norm0(x)

        output = self.audio_encoder.cam_forward(x.transpose(0,1), output['padding_mask'])
        decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)

        return decoder_out, output

    def extract_encoder_output(self, net_input):
        output = self.encoder(net_input['source'],net_input['padding_mask'])
        T, B, C = output['encoder_out'].size()
        x = torch.tensor(list(range(1000))) + 1
        x = repeat(x, 't -> t b', b=B)
        speech_units = self.embedding(x.cuda()).detach() # 200 512 B
        x, _ = self.mha0(query=output['encoder_out'], key=speech_units, value=speech_units) # T B D
        x = self.layer_norm0(x)
        output = self.audio_encoder.cam_forward(x.transpose(0,1), output['padding_mask'])
        

        return output

    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

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
