# Lip Reading for Low-resource Languages by Learning and Combining General Speech Knowledge and Language-specific Knowledge
This repository contains the PyTorch implementation of the following paper:
> **Lip Reading for Low-resource Languages by Learning and Combining General Speech Knowledge and Language-specific Knowledge**<be>
><br>
>**(ICCV 2023)**<br>
> \*Minsu Kim, \*Jeonghun Yeo, Jeongsoo Choi, Yong Man Ro<br>
> \[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Lip_Reading_for_Low-resource_Languages_by_Learning_and_Combining_General_ICCV_2023_paper.pdf)\]
<div align="center"><img width="100%" src="img/img.png?raw=true" /></div>


## Environment Setup
```bash
conda create -n lmd-vsr python=3.9 -y
conda activate lmd-vsr
git clone https://github.com/JeongHun0716/lmd-vsr
cd lmd-vsr
```
```bash
# PyTorch and related packages
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
(If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode
pip install omegaconf==2.0.6 hydra-core==1.0.7
pip install librosa
cd fairseq
pip install --editable ./
```


## Dataset preparation
For inference, Multilingual TEDx(mTEDx), and LRS2 Datasets are needed. 
  1. Download the mTEDx dataset from the [mTEDx link](https://www.openslr.org/100) of the official website.
  2. Download the LRS2 dataset from the [LRS2 link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) of the official website.


For training the LMDecoder, Multilingual LibriSpeech (MLS) and LRS3 Datasets are needed.
  1. Download the MLS dataset from the [MLS link](https://www.openslr.org/94/) of the official website.
  2. Download the LRS3 dataset from the [LRS3 link](https://mmai.io/datasets/lip_reading/) of the official website.

## Preprocessing 
After downloading the datasets, you should detect the facial landmarks of all videos and crop the mouth region using these facial landmarks. 
We recommend you preprocess the videos following [preparation](https://github.com/JeongHun0716/lmd-vsr/tree/main/avhubert/preparation).  


## Train
The train file list can be downloaded [here](https://www.dropbox.com/scl/fo/rom2ht2v6deptaup2q6ek/AIv1Og2C0d9TDGea87xH7Ro?rlkey=146wj786gfhi8hbtzhtskg1rh&st=c8t3p4fk&dl=0).

1. LMDecoder Pre-Training : To evaluate the performance of the LMDecoder, Please run the following command scripts/lmd_eval.sh.
In this project, we provide only the mTEDx files in the labels directory. To reproduce the same results with this paper, we recommend that you prepare the tsv, wrd, and unit files on the MLS dataset, and merge them with the provided files. 
```bash
bash scripts/lmd_pre_train.sh
```


2. LMD-VSR Training : If you wish to train only the LMD-VSR model, you can use the pre-trained LMDecoder model provided below, which has been trained using the MLS and mTEDx datasets.
```bash
bash scripts/lmd_vsr_train
```

## Inference
To measure the performance for each language, please run the following bash script in the lmd-vsr directory:
```bash
bash scripts/eval.sh
```
## Pretrained Models
Download the checkpoints from the below links and move them to the target directory. 
You can evaluate the performance of the finetuned model using the scripts available in the `scripts` directory.


| Model         | Training Datasets  | Used Language  |  WER(\%)  | Target Directory |
| AV-HuBERT Model         | Training Datasets  | Used Language   | Target Directory |
|--------------|:----------|:------------------:|:----------:|
| [base_vox_iter5.pt](https://facebookresearch.github.io/av_hubert/) |     LRS3 + VoxCeleb2      |        English         |  src/pretrained_models/encoder   |
| [large_vox_iter5.pt](https://facebookresearch.github.io/av_hubert/) |     LRS3 + VoxCeleb2      |        English         |   src/pretrained_models/encoder      |



| LMDecoder      | Training Datasets  | Used Language  |  WER(\%)  | Target Directory |
|--------------|:----------|:------------------:|:----------:|:----------:|
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/zxnycpjlffd18ok5bg7ob/AKwd8lxvbx_q_BECGnTI2Pc?rlkey=a8f5e8gjan15mmmcgmw1peo0p&st=2hmj185b&dl=0) |     LRS2 + LRS3      |        English           |     12.6  |  src/pretrained_models/lmdecoder/en   |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/33tx38c4r2nei64w0jtos/APuYqWYs3Zfl0_xZPdYFddg?rlkey=hsa5hshnl8091zm33w022cpaq&st=8xxgmp1z&dl=0) |     mTEDx + MLS      |        Spanish          |  25.1 |   src/pretrained_models/lmdecoder/es        |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/suovlgvi6l73wmdy1b3ik/AMq8VWaZcSW6wZKRmL0mbWk?rlkey=tj9fkw5dgp9s1key2r8j1ikox&st=idrouvm4&dl=0) |       mTEDx + MLS    |        French         |    28.3 | src/pretrained_models/lmdecoder/fr |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/zk3js3g02oibqg0a3umzz/AK9-DXffv852RVxmHc3KNv4?rlkey=aeab1jiymxc554v86cec2t7l3&st=9yc68za6&dl=0) |       mTEDx + MLS    |        Italian         |    29.4  | src/pretrained_models/lmdecoder/it |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/5mhqya2e5mdddsao4y3oc/AHanr8fswjGV2R5ZJnCNjz4?rlkey=4d45fzubbujq5cbavb5palfmx&st=p8ranvpf&dl=0) |       mTEDx + MLS     |        Portuguese         |  37.4  | src/pretrained_models/lmdecoder/pt |


| VSR Model     | Training Datasets  | Used Language  |  WER(\%)  | Target Directory |
|--------------|:----------|:------------------:|:----------:|:----------:|
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/60xihdj518w44ujnixp8p/AKhdf0TxhPL5MLjQLtX8zdc?rlkey=4ddbpecgqlg0rym4z9drkeg4d&st=023giear&dl=0) |     LRS2      |        English           |     23.8  |  src/pretrained_models/lmd_vsr/en   |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/fnxbjvrh8tythp2yvlbh5/AMYTkcFRC8RmUIh__25qKKc?rlkey=d9575wmdpfs398z5xhrfxbyof&st=atcbjq0p&dl=0) |     mTEDx      |        Spanish          |  70.2 |   src/pretrained_models/lmd_vsr/es        |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/1q1yezdiace2mn3sk4k3n/AIBLHO-5qbqjIa2zPp9_znI?rlkey=ihxcqp27qxy0ejcxnp042rek0&st=8lrb1w2i&dl=0) |       mTEDx    |        French         |    74.7 | src/pretrained_models/lmd_vsr/fr |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/jp1cnioc0g37yubtr13gs/AFx9q7OZ0oKGYFJcxqH33E0?rlkey=ufulmw6m04a1vpol22nx7nsy9&st=bk4jltpi&dl=0) |       mTEDx    |        Italian         |    68.0  | src/pretrained_models/lmd_vsr/it |
| [best_ckpt.pt](https://www.dropbox.com/scl/fo/6fakwmi26c8j9ujhqk4f1/AKrq4fM16AfqxM7zf2alHLM?rlkey=f84vxhdab1y8wnedkj797xjzo&st=bkmjyel0&dl=0) |       mTEDx     |        Portuguese         |    69.3  | src/pretrained_models/lmd_vsr/pt |



## Citation
If you find this work useful in your research, please cite the paper:
```bibtex
@inproceedings{kim2023lip,
  title={Lip reading for low-resource languages by learning and combining general speech knowledge and language-specific knowledge},
  author={Kim, Minsu and Yeo, Jeong Hun and Choi, Jeongsoo and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15359--15371},
  year={2023}
}
```
## Acknowledgement
This project is based on the [avhubert](https://github.com/facebookresearch/av_hubert) and [fairseq](https://github.com/facebookresearch/fairseq) code. We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.
