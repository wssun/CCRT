## Continuous Concepts Removal in Text-to-image Diffusion Models

 This is the repo for our paper *Continuous Concepts Removal in Text-to-image Diffusion Models*. 
 
 Our paper is accepted by NeuralPS 2025 (poster)!

## Installation Guide

* Download the weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and move them to `stable-diffusion/models/ldm` (This will be `ckpt_path` variable in `train-scripts/train-ccrt.py`)

## **Genetic variation generates entities**

* [IMPORTANT] Edit `train-script/train-ccrt.py` and change the default argparser values according to your convenience (especially the config paths)
* To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
* `python train-scripts/train-ccrt.py --prompt 'Van Gogh style art work' --train_method 'xattn' --devices '0,1' --erase_mode '0' --erase_type '0'`
* The trained model is saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`.
* `python eval-scripts/generate-entity.py --k '100' --crossover_num '0.8' --pt_path 'models/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt'`.Save the entity to `pkl/{filename}.pkl`.
* Then run `python generate-LLM-entity.py --pkl_name 'compvis-word_VanGoghstyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-em_0-et_0'`.The result is `entity-<based on pkl name>.csv`

## Training Guide

After installation, follow these instructions to train a custom CCRT model:

* `cd stable-diffusion` to the main repository of stable-diffusion
* [IMPORTANT] Edit `train-script/train-ccrt.py`and change the default argparser values according to your convenience (especially the config paths)
* To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
* `python train-scripts/train-ccrt.py --prompt 'Van Gogh style art work' --train_method 'xattn' --devices '0,1' --entity_name 'compvis-word_VanGoghstyleartwork-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-em_0-et_0' --erase_mode '0' --erase_type '1'`    

Note that the default argparser values must be changed! If continuous erasure is needed, repeat **Genetic variation generates entities** and **Training Guide**, modifying `entity_name` , `erase_mode`and`erase_type`.


## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* The path to model can be customised in the script.
* It is to be noted that the current version requires the model to be in saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
* `python eval-scripts/generate-images.py --model_name='compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-em_0-et_1' --prompts_path 'data/art_prompts.csv' --save_path 'evaluation_folder' --num_samples 10`


## Cite our paper
```
@inproceedings{Han-2025-CCRT,
  title={Continuous concepts removal in text-to-image diffusion models},
  author={Han, Tingxu and Sun, Weisong and Hu, Yanrong and Fang, Chunrong and Zhang, Yonglong and Ma, Shiqing and Zheng, Tao and Chen, Zhenyu and Wang, Zhenting},
  booktitle={Advances in Neural Information Processing Systems: NeuralPS 2025},
  year={2025}
}
```
