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
Modify feature from output.json into format of (video0.npy, video1.npy, video1.npy ...)
```python
import json
c3d_feats=json.load(open("output.json", "r"))
c3d_feat_dir='c3d_feats/'
if not os.path.exists(c3d_feat_dir):
    os.makedirs(c3d_feat_dir)
for video_feat in c3d_feats:
    name=video_feat['video'].split('.')[0]+'.npy'
    feat=np.array([video['features'] for video in video_feat['clips']])
    np.save(os.path.join(c3d_feat_dir, name), feat)
```

### Steps

1. preprocess videos and labels

```bash
python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152 --n_frame_steps 40  --gpu 4,5

python prepro_vocab.py
```

2. Training a model

```bash

python train.py --gpu 0 --epochs 3001 --batch_size 300 --checkpoint_path data/save --feats_dir data/feats/resnet152 --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir video-classification-3d-cnn-pytorch/c3d_feats --dim_vid 2560
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_100.pth --batch_size 100 --gpu 1
```

## TODO
- lstm
- GPT-3 decoder
- Test other ViT model as encoder

## Acknowledgements
Some code refers to [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
