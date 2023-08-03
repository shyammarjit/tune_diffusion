# Installation Steps
### 1. Conda env Create   - conda create -n name_of_env
### 2. Install acclerator - pip install accelerator
### 3. Install diffuser  -
* a. For normal usage     : pip install git+https://github.com/huggingface/diffusers
* b. For clone : git clone https://github.com/huggingface/diffusers.git 
* c. For editable :  pip install -e ".[torch]" in diffusers folder

### 4. Install requirements - 
* pip install -r requirements.txt  + clip install 
* pip install -r requirements_sdxl.txt
### 5. Train dreambooth_sdxl using script file    : bash run_lora_sdxl.sh
### 6. Generate images from the finetuned weights : python generator.py


<br>

-----------------------------------------------

# TO DO : Changes to integrate krona and promts and metrics
### Make changes in LoRa and than first try to generate images 
### Add lora_or_Krona parser
### add promts and metrics in the parsers and generator


