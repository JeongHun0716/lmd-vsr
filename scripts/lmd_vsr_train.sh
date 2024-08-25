target_lang=es # en, es, fr, it, pt
current_pth=$(pwd)
fairseq_pth=$current_pth/fairseq
src_pth=$current_pth/src

data_pth=$current_pth/labels/$target_lang
checkpoint_save_pth=$current_pth/exp/lmd_vsr/$target_lang

decoder_embed_dim=768
decoder_ffn_embed_dim=3072
decoder_attention_heads=4
lmd_embedding_dim=768
decoder_layers=6


if [ "$target_lang" == "en" ]; then
    decoder_embed_dim=1024
    decoder_ffn_embed_dim=4096
    decoder_attention_heads=8
    lmd_embedding_dim=1024
    decoder_layers=9
fi

PYTHONPATH=$fairseq_pth \
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
    --config-dir $src_pth/conf/lmd-vsr \
    --config-name $target_lang.yaml \
    task.data=$data_pth \
    task.label_dir=$data_pth \
    task.tokenizer_bpe_model=$current_pth/spm1000/$target_lang/spm_unigram1000.model \
    hydra.run.dir=$checkpoint_save_pth \
    common.user_dir=$src_pth \
    model.decoder_layers=$decoder_layers \
    model.decoder_embed_dim=$decoder_embed_dim \
    model.decoder_ffn_embed_dim=$decoder_ffn_embed_dim \
    model.decoder_attention_heads=$decoder_attention_heads \
    model._name=lmd_vsr_$target_lang