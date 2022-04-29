# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda

## requirements

- cuda
- pytorch 0.4.0
- python3
- ffmpeg (can install using anaconda)

### python packages

- tqdm
- pillow
- pretrainedmodels
- nltk

## Data


- MSR-VTT dataset download link:
https://www.mediafire.com/folder/h14iarbs62e7p/shared


## Options

all default options are defined in opt.py or corresponding code file, change them for your like.

## Acknowledgements

The original code is from: [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch) and [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) and this code is a slightly modifications of the original code.

Some code refers to [ImageCaptioning.pytorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## Usage

### (Optional) c3d features
you can use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

```bash
cd video-classification-3d-cnn-pytorch
```

To calculate video features for each 16 frames, use ```--mode feature```.
```bash
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode feature
```

### Steps

1. preprocess videos and labels

```bash
python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152 --n_frame_steps 40  --gpu 4,5

python prepro_vocab.py
```

2. Training a model

```bash

python train.py --gpu 0 --epochs 3001 --batch_size 300 --checkpoint_path data/save --feats_dir data/feats/resnet152 --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir data/feats/c3d_feats --dim_vid 4096
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_1000.pth --batch_size 100 --gpu 1
```

## TODO
- lstm
- beam search
- reinforcement learning
- dataparallel (broken in pytorch 0.4)


## Acknowledgements
Some code refers to [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
