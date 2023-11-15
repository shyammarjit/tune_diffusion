## Installation Steps

Conda env Create
```
conda create -y -n diffusers python=3.11
conda activate diffusers
```


Install diffuser from Our Space
```
pip install git+https://github.com/huggingface/diffusers
git clone git@github.com:shyammarjit/tune_diffusion.git
cd tune_diffusion 
pip install -e ".[torch]"
```

Install diffuser from HuggingFace
```
pip install git+https://github.com/huggingface/diffusers
git clone https://github.com/huggingface/diffusers.git
cd diffusers 
pip install -e ".[torch]"
```

Install requirements 
```
pip install -r requirements.txt 
pip install -r requirements_sdxl.txt
pip install bitsandbytes>=0.40.0
pip install xformers>=0.0.20
```

Install acclerator and wandb
```
pip install accelerator wandb
```

Install CLIP
```
# conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm scipy pandas
pip install git+https://github.com/openai/CLIP.git
```

Train dreambooth_sdxl using script file
```
bash run_lora_sdxl.sh
```

Generate images from the finetuned weights 
```
python generator.py
```

## What to run?
To run without text encoder config please hit this inside ```dreambooth``` folder
```
bash dreambooth/test.sh
```


To run with text encoder config please hit this inside ```dreambooth``` folder
```
bash test_text.sh
```


## How to edit the code?

Following are the files where we need to look into in order to change the things:
* All the files inside ```dreambooth``` folder.
* Inside ```src``` folder, in ```loaders.py``` script we need to revisit of all the functions.
* Inside ```src/tune_diffusion/models``` folder.
    * ```attention_processor.py``` ➡️ In ```LoRAAttnProcessor2_0``` class, add ```adapter_type``` and ```attn_update_unet```
    * ```lora.py``` ➡️ Need to visit all the functions.
    * ```unet_2d_condition.py``` ➡️ There was ```attn_processors``` as a @property of the prior class, here we have added ```ffn_processors``` as an another @property for ffn layers.
    Here, we also added ```set_ffn_processors``` another function within the parent class for ffn layers.
* Inside ```src/pipelines/stable_diffusion_xl``` folder.
    * In ```pipeline_stable_diffusion.py``` script, within ```load_lora_weights``` function need to add ```adapter_type```, ```attn_update_unet```, and ```attn_update_text``` as an extra arguments. This bypass call is getting only when we are using *SDXL* models. 