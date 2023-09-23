# Installation Steps

## Conda env Create
```
conda create -n diffusers
```

## Install acclerator
```
pip install accelerator
```

## Install diffuser
```
pip install git+https://github.com/huggingface/diffusers
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```

## Install requirements 
```
pip install -r requirements.txt 
pip install -r requirements_sdxl.txt
```

## Install CLIP
```
# conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm scipy pandas
pip install git+https://github.com/openai/CLIP.git
```

## Train dreambooth_sdxl using script file
```
bash run_lora_sdxl.sh
```

## Generate images from the finetuned weights 
```
python generator.py
```