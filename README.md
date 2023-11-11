## Create a conda environment

```
conda create -y -n <env_name>
```

## Install requirements  
```
pip install -r requirements.txt
```

# Install diffusers version 0.17.1 from [here](https://github.com/huggingface/diffusers/releases/tag/v0.17.1)
```
# Download at root
wget https://github.com/huggingface/diffusers/archive/refs/tags/v0.17.1.zip
unzip v0.17.1.zip 
cd diffusers-0.17.1
pip install -e.
```

## Install CLIP
```
pip install ftfy regex tqdm scipy pandas
pip install git+https://github.com/openai/CLIP.git
```

Note: Make two folders inside `training_scripts` folder. One is for `datas` and another is for `outputs`.




