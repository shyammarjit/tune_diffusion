# Installation Steps
## Conda env Create
```
conda create -n name_of_env
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
pip install -r requirements.txt  + clip install 
pip install -r requirements_sdxl.txt
```
## Train dreambooth_sdxl using script file
```
bash run_lora_sdxl.sh
```
## Generate images from the finetuned weights 
```
python generator.py
```


