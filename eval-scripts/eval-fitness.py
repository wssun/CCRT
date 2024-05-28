import sys
import os
dir_now=sys.path[0]
dir_up=os.path.dirname(dir_now)
dir_upup=os.path.dirname(dir_up)
print(dir_now)
print(dir_up)
print(dir_upup)
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import torch
import pickle
from omegaconf import OmegaConf
from pathlib import Path
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import sys
import os
from tqdm import tqdm
import random
import csv


def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def load_model_from_pt(config, pt, device="cpu", verbose=False):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    sd = torch.load(pt,map_location=device)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def get_models(config_path, ckpt_path, devices):
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model, sampler

def get_models_pt(config_path, ckpt_path, devices):
    model = load_model_from_pt(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model, sampler



@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def get_fitness_score(word, model_esd, model_org, sampler_esd, sampler_org, start_guidance, image_size, ddim_steps):
    ddim_eta = 0
    criteria = torch.nn.MSELoss()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t_esd = lambda x, s, code, t: sample_model(model_esd, sampler_esd,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    quick_sample_till_t_orig = lambda x, s, code, t: sample_model(model_org, sampler_org,
                                                                  x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                  start_code=code, till_T=t, verbose=False)

    emb_esd = model_esd.get_learned_conditioning([word])
    emb_org = model_org.get_learned_conditioning([word])

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        z_esd = quick_sample_till_t_esd(emb_esd.to(devices[0]), start_guidance, start_code,
                                        int(t_enc))  # emb_p seems to work better instead of emb_0
        z_ref = quick_sample_till_t_orig(emb_org.to(devices[0]), start_guidance, start_code, int(t_enc))

        e_ref = model_org.apply_model(z_ref.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_org.to(devices[1]))

        e_esd = model_esd.apply_model(z_esd.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_esd.to(devices[0]))
    e_ref.requires_grad = False
    loss = criteria(e_esd.to(devices[0]), e_ref.to(devices[0]))
    return loss.item()


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='eval', description='eval fitness')
    parser.add_argument('--class_dir', help='dir of ImageNet-1k word', type=str, required=False,
                        default="../wnid.txt")
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str,
                        required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False,
                        default='models/ldm/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False,
                        default=3)
    parser.add_argument('--k', help='k', type=float, required=False,
                        default=100)
    parser.add_argument('--crossover_num', help='crossover_num', type=float, required=False,
                        default=0.8)
    parser.add_argument('--pt_path', help='pt path for ccrt-et0', type=str, required=False,
                        default='models/compvis-word_Picassostyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/'
                                'compvis-word_Picassostyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt')


    args = parser.parse_args()
    class_dir =  args.class_dir
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    pt_path = args.pt_path
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    start_guidance = args.start_guidance
    k = args.k
    crossover_num = args.crossover_num

    model_org, sampler_org = get_models(config_path, ckpt_path, devices)
    model_esd, sampler_esd = get_models_pt(config_path,pt_path,devices)

    entities = []
    with open('data/entity-pklname.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            entities.append(row[1])
    total = 0
    for entity in entities:
        score = get_fitness_score(entity, model_esd, model_org, sampler_esd, sampler_org, start_guidance, image_size,
                      ddim_steps)
        total += score
        print(f"{entity} -- {score}")
    total = float(total / len(entities))
    print(f"mean :{total}")


























