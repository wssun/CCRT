import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import shuffle

import clip
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor

from PIL import Image
import matplotlib.pyplot as plt
import pathlib

import pdb
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
clip_path = "stable-diffusion-main/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(clip_path)
processor = CLIPProcessor.from_pretrained(clip_path)
model.to(device)

def get_files(path, prompt, epoch):
    file_names = sorted([file for file in path.glob(r'{}_*.png'.format(epoch))])
    clip_score = 0
    prompt = [prompt]
    for file in file_names:
        clip_score += get_clip_score(file,prompt)
    clip_score = clip_score /len(file_names)
    return clip_score


def get_clip_score(img_path,text):
    image = Image.open(img_path)

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 1,4
    return logits_per_image[0]

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog = 'eval', description = 'Evaluate clips')
    parser.add_argument('--reference_dir', help='dir of reference images', type=str, required=False,default="/home/david/hyr/erasing/autodl-tmp/stable-diffusion-main/eraseVanGptV1W0001/compvis-word_VanGoghstyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05_v_gpt_entityV1_w0001")
    parser.add_argument('--data_dir', help='dir of prompts', type=str, required=False,default="/home/david/hyr/erasing/autodl-tmp/stable-diffusion-main/data/art_prompts_entity.csv")
    parser.add_argument('--save_dir', help='path to save results', type=str, required=False, default="CLIPS_删除梵高gptw0001")


    args = parser.parse_args()


    reference_dir = args.reference_dir
    data_dir = args.data_dir
    df_prompt = pd.read_csv(data_dir)
    path_reference = pathlib.Path(reference_dir)

    image_names = os.listdir(reference_dir)
    epoch_ids = sorted(set(name.split("_")[0] for name in image_names if '.png' in name or '.jpg' in name))


    df = pd.DataFrame()
    clip_score_values = {}
    mean_value = 0
    for epoch_id in epoch_ids:
        prompt = df_prompt.iloc[int(epoch_id)]['prompt']

        clip_score_values[epoch_id] = round(float(get_files(path_reference, prompt, epoch_id)), 2)
        mean_value += clip_score_values[epoch_id]
        print(f"{epoch_id}: {prompt}，clipscore:{clip_score_values[epoch_id]}")
    mean_value = float(mean_value/len(epoch_ids))
    print(f" mean : {mean_value}")
    df[f"CLIPScore"] = pd.Series(clip_score_values)
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f"{args.save_dir}/Clips_TxtToImg.csv")

    
