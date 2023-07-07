from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch, os, clip
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from prompts import get_promts
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import wandb

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
# torch.manual_seed(0)


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



def evaluator(args):
    clipi = CLIP_I(org_folder_path = args.instance_data_dir, gen_folder_path = args.output_dir)
    clipt = CLIP_T(gen_folder_path = args.output_dir, prompts = get_promts(os.path.basename(args.instance_data_dir)))
    return clipi, clipt


def generator(args):
    if(args.lora_or_krona==1):
        from krona_diffusion import tune_lora_scale, patch_pipe
        wandb.init(project='krona-exp',name=f"krona_{args.lora_rank}_{args.learning_rate}_{args.learning_rate_text}_{args.alpha_text}_{args.alpha_unet}")

    elif(args.lora_or_krona==0):
        from lora_diffusion import tune_lora_scale, patch_pipe
        wandb.init(project='krona-exp',name=f"lora_{args.lora_rank}_{args.learning_rate}_{args.learning_rate_text}_{args.alpha_text}_{args.alpha_unet}")

    else:
        raise AttributeError(f"wrong ataptor type {args.adaptor}")
    
    
    prompts = get_promts(os.path.basename(args.instance_data_dir))
    lora_weight_path = os.path.join(args.output_dir, "lora_weight.safetensors")
    # Patch the pipe with the updated lora_weight_path
    patch_pipe(pipe, lora_weight_path, patch_text=True, patch_ti=True, patch_unet=True)
    
    # Tune lora_scale for unet and text_encoder
    tune_lora_scale(pipe.unet, args.alpha_unet)
    tune_lora_scale(pipe.text_encoder, args.alpha_text)
    
    args.output_dir = os.path.join(args.output_dir, 'images') # create the image folder
    if(os.path.exists(args.output_dir)): pass
    else: os.mkdir(args.output_dir)
    
    for i in trange(len(prompts), desc = "generating images"):
        image = pipe(prompts[i], num_inference_steps=50, guidance_scale=7).images[0]
        image_save_path = os.path.join(args.output_dir, "image_{}.jpg".format(i+1))
        wandb.log({'Final Images': wandb.Image(image)})
        # # wandb.log({'subject': wandb.weights(args.learning_rate_text)})
        # wandb.log({'rank(r)': wandb.weights(args.lora_rank)})
        # wandb.log({'learning rate': wandb.weights(args.learning_rate)})
        # wandb.log({'learning rate text': wandb.weights(args.learning_rate_text)})
        # wandb.log({'alpha text': wandb.weights(args.alpha_text)})
        # wandb.log({'alpha unet': wandb.weights(args.alpha_unet)})

        
        image.save(image_save_path)
    print(f"Image generation completed.")
