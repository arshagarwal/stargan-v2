
## StarGAN v2 - PyTorch Implementation with multi-gpu support


This repository adapts heavily from official stargan-v2 repository but simplifies data loading, removes style encoder, which greatly expedites training speed with marginal quality tradeoffs, provides multi-gpu support making it easier to experiment. Also, the repository makes use of Hinge-Loss instead of non-saturating loss which I conclude, through my experiments, works better in this particular setting.  

## Following are the results on CelebA-HQ Male-to-Female dataset.
![Results](https://github.com/arshagarwal/stargan-v2/blob/test3/assets/28500_256images.jpg)

## Software installation
Clone this repository:

```bash
git clone https://github.com/clovaai/stargan-v2.git
cd stargan-v2/
```

Install the dependencies:
```bash
conda create -n stargan-v2 python=3.6.7
conda activate stargan-v2
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
```

## Datasets and pre-trained networks
We provide a script to download datasets used in StarGAN v2 and the corresponding pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `data` and `expr/checkpoints` directories, respectively.

<b>CelebA-HQ.</b> To download the [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs) dataset and the pre-trained network, run the following commands:
```bash
bash download.sh celeba-hq-dataset
bash download.sh pretrained-network-celeba-hq
bash download.sh wing
```

<b>AFHQ.</b> To download the [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) dataset and the pre-trained network, run the following commands:
```bash
bash download.sh afhq-dataset
bash download.sh pretrained-network-afhq
```

## Training networks
To train StarGAN v2 from scratch, run the following commands. Generated images and network checkpoints will be stored in the `samples` and `expr/checkpoints` directories, respectively. Training takes about three days on a single Tesla V100 GPU. Please see [here](https://github.com/clovaai/stargan-v2/blob/master/main.py#L86-L179) for training arguments and a description of them. 

```bash
# celeba-hq
python main.py --mode train --num_domains 2 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val --gpus 0,1

# afhq
python main.py --mode train --num_domains 3 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val --gpus 0,1
```
## Multi-gpu training 
Use the `--gpus` argument to provide as input a string separated with **","** denoting device-ids of gpus to be used for training.
## Resuming training
Use the `--resume_iter` argument to restart training from a specifc checkpoint. 
