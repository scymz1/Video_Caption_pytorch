import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import clip
import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
from PIL import Image

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["video-classification-3d-cnn-pytorch/ffmpeg-5.0.1-amd64-static/ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn, flattern=False, preprocess=None, model_type='CNN'):
    global C, H, W
    model.eval()
    
    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    extracted_video = glob.glob('./data/feats/CLIP_ViT/*')
    extracted_video = [i.split("/")[-1] for i in extracted_video]
    # print(extracted_video)
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split("\\")[-1].split(".")[0]
        if video_id + ".npy" in extracted_video:
            continue
        dst = params['model'] + '_' + video_id
        extract_frames(video, dst)
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 2, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        if model_type == 'CNN':
            images = torch.zeros((len(image_list), C, H, W))
            for iImg in range(len(image_list)):
                img = load_image_fn(image_list[iImg])
                images[iImg] = img
            with torch.no_grad():
                fc_feats = model(images.cuda()).squeeze()
        elif model_type == 'ViT':
            images = []
            for iImg in image_list:
                image = Image.open(iImg).convert("RGB")
                images.append(preprocess(image))
            image_input = torch.tensor(np.stack(images)).cuda()
            with torch.no_grad():
                fc_feats = model.encode_image(image_input).float()
                fc_feats = torch.flatten(fc_feats, 1)
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')
    
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    
    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    
    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)
    elif params['model'] == 'CLIP_ViT_B-32':
        C, H, W = 3, 299, 299
        model, preprocess = clip.load("ViT-B/32")
        extract_feats(params, model, None, flattern=True, preprocess=preprocess, model_type='ViT')
        exit()
    else:
        print("doesn't support %s" % (params['model']))
    
    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    
    model = model.cuda()
    extract_feats(params, model, load_image_fn)
