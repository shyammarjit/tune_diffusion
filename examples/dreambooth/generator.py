
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
import torch, os, clip, glob, json
from PIL import Image
from prompts import get_promts
from tqdm import trange
from diffusers import loaders
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import wandb, shutil
from train_dreambooth_lora_sdxl import parse_args
import pandas as pd


class DatasetWrapper(Dataset):
    def __init__(self, images_path, input_size = 512):
        self.images_path = images_path; self.input_size = input_size
        # Build transform
        self.trans = T.Compose([T.PILToTensor(), T.Resize(size=(self.input_size, self.input_size))]) 
        # total_images = args.images
        # distribution = [i for i in range(total_images)]
        # num_selected_images = int(selection_p * total_images)
        # sampled_elements = random.sample(distribution, num_selected_images)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = preprocess(Image.open(self.images_path[idx]))
        return img
    
    

def CLIP_I(org_folder_path = None, gen_folder_path = None):
    org_image_files = sorted([os.path.join(org_folder_path, f) for f in os.listdir(org_folder_path)])
    gen_image_files = sorted([os.path.join(gen_folder_path, f) for f in os.listdir(gen_folder_path)])
    
    # create dataloader for both the folder
    if(len(org_image_files)>128): batch_size = 128
    else: batch_size = len(org_image_files)
    org_datloader = DataLoader(DatasetWrapper(org_image_files), batch_size=batch_size, shuffle=False)
    if(len(gen_image_files)>128): batch_size = 128
    else: batch_size = len(gen_image_files)
    gen_dataloader = DataLoader(DatasetWrapper(gen_image_files), batch_size=batch_size, shuffle=False)
    
    clipi = []
    for i_batch in org_datloader:
        i_batch = model.encode_image(i_batch.to(device)).to(device) # pass this to CLIP model
        for j_batch in gen_dataloader:
            j_batch = model.encode_image(j_batch.to(device)).to(device) # pass this to CLIP model
            clipi.append(torch.mean(F.cosine_similarity(i_batch.unsqueeze(1), j_batch.unsqueeze(0), dim=-1)).item())
    clipi = torch.mean(torch.FloatTensor(clipi))
    print(f"cosine cimilarity Image-2-Image {clipi}")
    return clipi.item()
    


def CLIP_T(gen_folder_path = None, prompts = None):
    total_similarity, num_images = 0.0, 0

    # Iterate over the images and prompts simultaneously
    for image_filename, prompt in zip(os.listdir(gen_folder_path), prompts):
        # Load and preprocess the image
        image_path = os.path.join(gen_folder_path, image_filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Tokenize and encode the text prompt
        text_input = clip.tokenize(prompt).to(device)
        text_embedding = model.encode_text(text_input).to(device)
        
        image_embedding = model.encode_image(image).to(device) # Encode the image
        similarity = torch.cosine_similarity(image_embedding, text_embedding)
        # Accumulate the similarity score
        total_similarity += similarity.item()
        num_images += 1

    # Calculate the average similarity score
    average_similarity = total_similarity / num_images
    print(f'Cosine similarity CLIP-T {average_similarity}')
    return average_similarity



def evaluator(args, prompts):
    image_dir = os.path.join(args.output_dir, 'images') 
    clipi = CLIP_I(org_folder_path = args.instance_data_dir, gen_folder_path = image_dir)
    clipt = CLIP_T(gen_folder_path = image_dir, prompts = prompts)
    return clipi, clipt


def generator(args, prompts):
    if(args.diffusion_model == "sdxl"):
        # load the SDXL model
        model_id = "stabilityai/stable-diffusion-xl-base-0.9"
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(args.output_dir, adapter_type=args.adapter_type)
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        refiner.to("cuda"); generator = torch.Generator("cuda").manual_seed(args.seed)
    
    elif(args.diffusion_model == "base"):
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(args.output_dir, adapter_type=args.adapter_type)
    else:
        raise AttributeError("only supported base and sdxl model")
    
    # Load phm_rule from saved pt file.
    if args.adapter_type == 'kadapt':
        phm_rule = torch.load(f'{args.output_dir}/phm_rule.pt')
    else: phm_rule = None

    # create the image folder
    image_dir = os.path.join(args.output_dir, 'images') 
    if(os.path.exists(image_dir)): pass
    else: os.mkdir(image_dir)
    
    for i in trange(len(prompts), desc = "generating images"):
        if(args.diffusion_model == "sdxl"):
            image = pipe(prompt=prompts[i], output_type="latent", generator=generator, cross_attention_kwargs=phm_rule).images[0]
            image = refiner(prompt=prompts[i], image=image[None, :], generator=generator).images[0]
        else: # without sdxl run
            image = pipe(prompts[i], num_inference_steps=50, guidance_scale=7, cross_attention_kwargs=phm_rule).images[0]
        
        image_save_path = os.path.join(image_dir, "image_{}.jpg".format(i+1))
        # wandb.log({'Final Images': wandb.Image(image)})
        # # wandb.log({'subject': wandb.weights(args.learning_rate_text)})
        # wandb.log({'rank(r)': wandb.weights(args.lora_rank)})
        # wandb.log({'learning rate': wandb.weights(args.learning_rate)})
        # wandb.log({'learning rate text': wandb.weights(args.learning_rate_text)})
        # wandb.log({'alpha text': wandb.weights(args.alpha_text)})
        # wandb.log({'alpha unet': wandb.weights(args.alpha_unet)})
        image.save(image_save_path)
    print(f"Image generation completed.")
    if args.diffusion_model == "sdxl":
        del pipe, refiner
    else:
        del pipe


def save_metrics(args, clipi, clipt):
    if not os.path.isdir(args.instance_data_dir):
        raise ValueError("The provided path is not a valid directory.")
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(args.instance_data_dir, ext)))
    number_of_input_img=len(images)    # Return the number of images found

    """ This function saves the config and evaluation results in the .csv format at the 
    given root output folder """
    # save all the info in the form of *.json file
    exp_info = {"Dataset_Name": os.path.basename(args.instance_data_dir), 
        "diffusion_ver": args.diffusion_model,
        "Adaptor": args.adapter_type,
        "lora_rank": args.lora_rank, 
        "clipi": clipi, "clipt": clipt, 
        "num_train_steps": args.max_train_steps, 
        "no_of_images": number_of_input_img, 
        "learning_rate": args.learning_rate, 
        "output_path": args.output_dir, 
        "with_prior_preservation": args.with_prior_preservation,
        }
    exp_name = f'{args.output_dir}/log_{args.diffusion_model}_{args.adapter_type}_{args.lora_rank}_{args.max_train_steps}_{number_of_input_img}_{args.learning_rate}_{args.with_prior_preservation}.json'
    print(exp_name)
    
    with open(exp_name, 'w') as f:
        f.write(json.dumps(exp_info, indent=4))
    
    filename = 'data.csv'
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=exp_info.keys())
        df.to_csv(filename, index=False)

    df = pd.read_csv(filename)
    df = pd.concat([df, pd.DataFrame(exp_info, index=[0])], ignore_index=True)
    df.to_csv('data.csv', index=False)

    # delete the save files
    remove_weights = False
    if(remove_weights):
        if(args.diffusion_model=="sdxl"):
            os.remove(os.path.join(args.output_dir, "pytorch_lora_weights.bin")) # delet lora weights
            if(os.path.isdir(os.path.join(args.output_dir, 'logs'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
            if(os.path.isdir(os.path.join(args.output_dir, 'checkpoint-500'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
            if(os.path.isdir(os.path.join(args.output_dir, 'checkpoint-1000'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
    
        elif(args.diffusion_model=="base"):
            os.remove(os.path.join(args.output_dir, "pytorch_lora_weights.bin")) # delet lora weights
            if(os.path.isdir(os.path.join(args.output_dir, 'logs'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
            if(os.path.isdir(os.path.join(args.output_dir, 'checkpoint-500'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
            if(os.path.isdir(os.path.join(args.output_dir, 'checkpoint-1000'))):
                shutil.rmtree(os.path.join(args.output_dir, 'logs'), ignore_errors=True)
        else:
            raise AttributeError("Wrong input for diffusion model.")




if __name__ == "__main__":
    # load the clip model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    # get the arguments
    args = parse_args()
    # generate the prompts
    prompts = get_promts(os.path.basename(args.instance_data_dir))
    # generate images base don given prompts
    generator(args, prompts)
    # compute the quantiative results (CLIP-I, CLIP-T)
    clipi, clipt = evaluator(args, prompts)
    save_metrics(args, clipi, clipt)

