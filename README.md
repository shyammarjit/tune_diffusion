## Create a conda environment

```
conda create -y -n <env_name>
```

## Install requirements  
```
pip install -r requirements.txt
```

Note: If `pip install diffusers` fails then try:
```
conda install -c conda-forge diffusers
```

Note: Make two folders inside `training_scripts` folder. One is for `datas` and another is for `outputs`.




