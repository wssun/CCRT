import sys
import os
from tqdm import tqdm
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
import random

class individual:
    def __init__(self, id, concepts = None, score = 0.0):
        if concepts is None:
            self.concepts = []
        else:
            self.concepts = concepts
        self.id = id
        self.score = score

def exist_ancestor(individual1, individual2):
    id1_list = individual1.id
    id2_list = individual2.id
    common_ancestors = set()
    common_descendants = set()

    for id1 in id1_list:
        for id2 in id2_list:
            ancestors_of_id1 = ancestors_map.get(id1, set())
            ancestors_of_id2 = ancestors_map.get(id2, set())
            common_ancestors.update(ancestors_of_id1 & ancestors_of_id2)

    for id1 in id1_list:
        for id2 in id2_list:
            descendants_of_id1 = descendants_map.get(id1, set())
            descendants_of_id2 = descendants_map.get(id2, set())
            common_descendants.update(descendants_of_id1 & descendants_of_id2)

    if not common_ancestors and not common_descendants:
        return False, []

    combined_set = common_ancestors | common_descendants
    combined_list = list(combined_set)
    return True, combined_list

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

def get_fitness_score(individual_list, model_esd, model_org, sampler_esd, sampler_org, start_guidance, image_size, ddim_steps):
    ddim_eta = 0
    criteria = torch.nn.MSELoss()
    quick_sample_till_t_esd = lambda x, s, code, t: sample_model(model_esd, sampler_esd,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    quick_sample_till_t_orig = lambda x, s, code, t: sample_model(model_org, sampler_org,
                                                                  x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                  start_code=code, till_T=t, verbose=False)

    word = ','.join(individual_list[i].concepts)
    emb_esd = model_esd.get_learned_conditioning([word])
    emb_org = model_org.get_learned_conditioning([word])

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        z_esd = quick_sample_till_t_esd(emb_esd.to(devices[0]), start_guidance, start_code,
                                        int(t_enc))
        z_ref = quick_sample_till_t_orig(emb_org.to(devices[0]), start_guidance, start_code, int(t_enc))

        e_ref = model_org.apply_model(z_ref.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_org.to(devices[0]))

        e_esd = model_esd.apply_model(z_esd.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_esd.to(devices[0]))

    loss = criteria(e_esd.to(devices[0]), e_ref.to(devices[0]))
    return loss.item()

def remove_exceed_class(individual_list):
    remove_list = []
    for i in range(len(individual_list)):
        if len(individual_list[i].concepts) >= 4:
            remove_list.append(individual_list[i])
    return [ind for ind in individual_list if ind not in remove_list]

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog='eval', description='generate entity')
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
                        default='models/compvis-word_VanGoghstyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/'
                                'compvis-word_VanGoghstyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt')


    args = parser.parse_args()
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    pt_path = args.pt_path
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    start_guidance = args.start_guidance
    k = args.k
    crossover_num = args.crossover_num

    model_org, sampler_org = get_models(config_path, ckpt_path,devices)
    model_esd, sampler_esd = get_models_pt(config_path,pt_path,devices)

    individual_list = []
    ancestors_map = {}
    descendants_map = {}
    class_map = {}
    entity_map = {}
    gloss_map = {}

    with open('data/is_a.txt', 'r') as file:
        for line in file:
            ancestor, descendant = line.strip().split()
            ancestors_map.setdefault(ancestor, set()).add(descendant)
            descendants_map.setdefault(descendant, set()).add(ancestor)

    with open('data/words.txt') as file:
        for inner_line in file:
            parts = inner_line.strip().split('\t')
            id = parts[0]
            word = parts[1]
            class_map[id] = word
    file.close()

    with open('data/gloss.txt') as file:
        for inner_line in file:
            parts = inner_line.strip().split('\t')
            id = parts[0]
            word = parts[1]
            gloss_map[id] = word
    file.close()

    with open('data/wnid.txt') as f:
        for line in f:
            line_split = line.split()
            wnid = str(line_split[0])
            found_word = class_map[wnid]
            item = individual([wnid],[found_word])
            entity_map[wnid] = found_word
            individual_list.append(item)
    f.close()

    iteration = 0
    while len(individual_list) > k and iteration < 6:
        pbar = tqdm(range(len(individual_list)))
        iteration += 1
        for i in pbar:
            individual_list[i].score = get_fitness_score(individual_list, model_esd, model_org, sampler_esd, sampler_org, start_guidance, image_size, ddim_steps)

        top_k_individuals = sorted(individual_list, key=lambda x: x.score, reverse=True)[:k]

        selection_count = max(1, k * crossover_num)
        selection_count -= selection_count % 2

        random.shuffle(top_k_individuals)
        selected_individuals = top_k_individuals[:int(selection_count)]
        individual_list = top_k_individuals
        while selected_individuals:
            index1 = random.randrange(len(selected_individuals))
            index2 = random.randrange(len(selected_individuals) - 1) if len(selected_individuals) > 1 else 0
            if index1 == index2:
                index2 = (index2 + 1) % len(selected_individuals)
            ind1 = selected_individuals[index1]
            ind2 = selected_individuals[index2]

            flag, ancestors_descendants_list = exist_ancestor(ind1, ind2)
            if flag:
                crossover_id = ancestors_descendants_list
                crossover_concept = []
                for key in ancestors_descendants_list:
                    crossover_concept.append(class_map[key])
            else:
                crossover_concept = ind1.concepts + ind2.concepts
                crossover_id = ind1.id + ind2.id
            new_ind = individual(crossover_id, crossover_concept)
            individual_list.append(new_ind)
            if index1 > index2:
                del selected_individuals[index1]
                del selected_individuals[index2]
            else:
                del selected_individuals[index2]
                del selected_individuals[index1]

        num_elements_to_select = max(1, len(individual_list) // 4)
        selected_individuals_b = random.sample(individual_list, num_elements_to_select)
        individual_list = [ind for ind in individual_list if ind not in selected_individuals_b]
        for ind in selected_individuals_b:
            ids = ind.id
            concepts = ind.concepts
            id_to_index = {id_: index for index, id_ in enumerate(ids)}

            for id_ in ids:
                if random.random() < 0.05:
                    replacement_id, replacement_concept = random.choice(list(class_map.items()))
                    concepts[id_to_index[id_]] = replacement_concept
                    ids[id_to_index[id_]] = replacement_id

            individual_list.append(individual(ids, concepts))
        individual_list = remove_exceed_class(individual_list)
        print(f"iteration:{iteration} scale:{len(individual_list)}")

    for i in range(len(individual_list)):
        print(f"id:{individual_list[i].id}  concepts:{individual_list[i].concepts}")
    individual_depiction_list = []
    for i in range(len(individual_list)):
        ids = individual_list[i].id
        concepts = individual_list[i].concepts
        id_to_index = {id_: index for index, id_ in enumerate(ids)}

        for id_ in ids:
            depiction = gloss_map[id_]
            concepts[id_to_index[id_]] += " : " + depiction

        individual_depiction_list.append(individual(ids,concepts))


    filename_with_ext = os.path.basename(pt_path)
    filename, file_extension = os.path.splitext(filename_with_ext)
    file_path = f'../pkl/{filename}.pkl'
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        pickle.dump(individual_depiction_list, f)


























