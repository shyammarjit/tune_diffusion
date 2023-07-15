# from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
# import torch

# model_id = "stabilityai/stable-diffusion-2-1-base"

# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
#     "cuda"
# )
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# # prompt = "Super Saiyan" #1
# # prompt = "Super Saiyan running on road" #2
# # prompt = "Super Saiyan with black hair and green clothes" #3x
# # prompt = "Super Saiyan electrically charged with blue theme" #4
# # prompt = "Super Saiyan angry face" #5


# prompt = "Super Saiyan angry face" #5
# prompt = "Super Saiyan with spiky golden hair, glowing blue eyes, and a fierce expression." #1
# prompt = "Super Saiyan charging up an energy blast, with their aura blazing around them." #2
# prompt = "Super Saiyan in mid-flight, with their hair flowing upward and their fists clenched." #3
# prompt = "Super Saiyan powering up, with electricity crackling around their body and their muscles bulging." #4
# prompt = "Super Saiyan angry face." #5
# prompt = "Super Saiyan performing a signature attack, such as the Kamehameha or Final Flash." #6
# prompt = "Two Super Saiyans engaged in an intense battle, with energy beams colliding and explosions in the background." #7
# prompt = "Super Saiyan in a calm and focused state, with a serene expression and a controlled aura." #8
# prompt = "Super Saiyan transforming from their base form, capturing the dynamic energy and hair change." #9
# prompt = "Super Saiyan with flowing silver hair and piercing green eyes, radiating an aura of pure energy." #10
# prompt = "Super Saiyan unleashing a devastating punch, shattering the ground beneath them." #11
# prompt = "Super Saiyan surrounded by a swirling vortex of energy, their body glowing with power." #12
# prompt = "Super Saiyan charging up a massive energy sphere, ready to unleash it upon their opponent." #13
# prompt = "Super Saiyan ascending into the sky, leaving streaks of golden light behind them." #14
# prompt = "Super Saiyan in a defensive stance, their aura forming a protective barrier around them." #15
# prompt = "Super Saiyan pushing their limits, with their muscles bulging and sweat pouring down their face." #16
# prompt = "Super Saiyan powering up, their hair standing on end and their energy crackling around them." #17
# prompt = "Super Saiyan charging headfirst into battle, their battle cry echoing through the air." #18
# prompt = "Super Saiyan meditating, their aura calm and controlled as they gather their inner strength." #19
# prompt = "Super Saiyan engaged in high-speed aerial combat, leaving afterimages in their wake." #20
# prompt = "Super Saiyan undergoing a transformation, their body surrounded by a cocoon of swirling energy." #21
# prompt = "Super Saiyan in a fierce battle, their fists clashing with an enemy's attack in a burst of energy." #22
# prompt = "Super Saiyan unleashing a powerful energy wave, obliterating everything in its path." #23
# prompt = "Super Saiyan charging up a beam attack, their hands crackling with electricity." #24
# prompt = "Super Saiyan surrounded by a group of allies, united in their fight against a common enemy." #25
# prompt = "Super Saiyan hovering above the ground, their hair and aura flowing with an ethereal glow." #26
# prompt = "Super Saiyan training in a secluded mountain area, their determination etched on their face." #27
# prompt = "Super Saiyan using their immense strength to lift a massive boulder effortlessly." #28
# prompt = "Super Saiyan engaged in a fierce sparring match, their fists colliding in a flurry of blows." #29
# prompt = "Super Saiyan caught in a fierce energy clash, their power equal to that of their opponent." #30
# prompt = "Super Saiyan soaring through the sky, leaving a trail of golden light behind them." #31
# prompt = "Super Saiyan surrounded by a swirling tornado of energy, their hair and clothes billowing in the intense winds." #32
# prompt = "Super Saiyan in a calm and focused meditation, harnessing their inner power." #33
# prompt = "Super Saiyan unleashing a flurry of rapid punches, their fists a blur of motion." #34
# prompt = "Super Saiyan engulfed in a fiery aura, their power radiating in waves." #35
# prompt = "Super Saiyan channeling their energy into a powerful beam attack, piercing through the sky." #36
# prompt = "Super Saiyan surrounded by a vibrant energy aura, their presence commanding and awe-inspiring." #37
# prompt = "Super Saiyan unleashing a devastating kick, causing shockwaves to ripple through the ground." #38
# prompt = "Super Saiyan training under a waterfall, their determination evident in their intense gaze." #39
# prompt = "Super Saiyan locked in an intense beam struggle, their energy clashing with an equally powerful opponent." #40
# prompt = "Super Saiyan with their arms crossed, exuding a sense of confidence and strength." #41
# prompt = "Super Saiyan transforming into a higher level of Super Saiyan, their power exponentially increasing." #42
# prompt = "Super Saiyan charging up an energy blast in each hand, ready to unleash a dual attack." #43
# prompt = "Super Saiyan surrounded by a cosmic backdrop, their energy blending with the stars." #44
# prompt = "Super Saiyan performing an acrobatic aerial maneuver, showcasing their agility and speed." #45
# prompt = "Super Saiyan raising their power level to its maximum, causing the ground to shake with their energy." #46
# prompt = "Super Saiyan engaged in a hand-to-hand combat, their movements fluid and precise." #47
# prompt = "Super Saiyan extending their arm, summoning an energy sphere that crackles with raw power." #48
# prompt = "Super Saiyan pushing themselves beyond their limits, their body surrounded by a blazing inferno." #49
# prompt = "Super Saiyan charging towards the viewer, their eyes ablaze with determination." #50





# torch.manual_seed(0)


# from lora_diffusion import tune_lora_scale, patch_pipe

# patch_pipe(
#     pipe,
#     # "./",
#     # "/home/smarjit/tune_diffusion/training_scripts/example_loras/lora_illust.safetensors",
    
#     "/home/smarjit/tune_diffusion/training_scripts/output_lora/lora_weight.safetensors", # lora.py -> LORA
    
#     # "/home/smarjit/tune_diffusion/training_scripts/output_krona_4/lora_weight.safetensors", # krona 
    
#     patch_text=True,
#     patch_ti=True,
#     patch_unet=True,
# )

# tune_lora_scale(pipe.unet, 1.00)
# tune_lora_scale(pipe.text_encoder, 1.00)

# torch.manual_seed(0)
# image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
# image.save("./b.jpg")






from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# prompts = [
#     "Super Saiyan with spiky golden hair, glowing blue eyes, and a fierce expression.",  #1
#     "Super Saiyan charging up an energy blast, with their aura blazing around them.",  #2
#     "Super Saiyan in mid-flight, with their hair flowing upward and their fists clenched.",  #3
#     "Super Saiyan powering up, with electricity crackling around their body and their muscles bulging.",  #4
#     "Super Saiyan angry face.",  #5
#     "Super Saiyan performing a signature attack, such as the Kamehameha or Final Flash.",  #6
#     "Two Super Saiyans engaged in an intense battle, with energy beams colliding and explosions in the background.",  #7
#     "Super Saiyan in a calm and focused state, with a serene expression and a controlled aura.",  #8
#     "Super Saiyan transforming from their base form, capturing the dynamic energy and hair change.",  #9
#     "Super Saiyan with flowing silver hair and piercing green eyes, radiating an aura of pure energy.",  #10
#     "Super Saiyan unleashing a devastating punch, shattering the ground beneath them.",  #11
#     "Super Saiyan surrounded by a swirling vortex of energy, their body glowing with power.",  #12
#     "Super Saiyan charging up a massive energy sphere, ready to unleash it upon their opponent.",  #13
#     "Super Saiyan ascending into the sky, leaving streaks of golden light behind them.",  #14
#     "Super Saiyan in a defensive stance, their aura forming a protective barrier around them.",  #15
#     "Super Saiyan pushing their limits, with their muscles bulging and sweat pouring down their face.",  #16
#     "Super Saiyan powering up, their hair standing on end and their energy crackling around them.",  #17
#     "Super Saiyan charging headfirst into battle, their battle cry echoing through the air.",  #18
#     "Super Saiyan meditating, their aura calm and controlled as they gather their inner strength.",  #19
#     "Super Saiyan engaged in high-speed aerial combat, leaving afterimages in their wake.",  #20
#     "Super Saiyan undergoing a transformation, their body surrounded by a cocoon of swirling energy.",  #21
#     "Super Saiyan in a fierce battle, their fists clashing with an enemy's attack in a burst of energy.",  #22
#     "Super Saiyan unleashing a powerful energy wave, obliterating everything in its path.",  #23
#     "Super Saiyan charging up a beam attack, their hands crackling with electricity.",  #24
#     "Super Saiyan surrounded by a group of allies, united in their fight against a common enemy.",  #25
#     "Super Saiyan hovering above the ground, their hair and aura flowing with an ethereal glow.",  #26
#     "Super Saiyan training in a secluded mountain area, their determination etched on their face.",  #27
#     "Super Saiyan using their immense strength to lift a massive boulder effortlessly.",  #28
#     "Super Saiyan engaged in a fierce sparring match, their fists colliding in a flurry of blows.",  #29
#     "Super Saiyan caught in a fierce energy clash, their power equal to that of their opponent.",  #30
#     "Super Saiyan soaring through the sky, leaving a trail of golden light behind them.",  #31
#     "Super Saiyan surrounded by a swirling tornado of energy, their hair and clothes billowing in the intense winds.",  #32
#     "Super Saiyan in a calm and focused meditation, harnessing their inner power.",  #33
#     "Super Saiyan unleashing a flurry of rapid punches, their fists a blur of motion.",  #34
#     "Super Saiyan engulfed in a fiery aura, their power radiating in waves.",  #35
#     "Super Saiyan channeling their energy into a powerful beam attack, piercing through the sky.",  #36
#     "Super Saiyan surrounded by a vibrant energy aura, their presence commanding and awe-inspiring.",  #37
#     "Super Saiyan unleashing a devastating kick, causing shockwaves to ripple through the ground.",  #38
#     "Super Saiyan training under a waterfall, their determination evident in their intense gaze.",  #39
#     "Super Saiyan locked in an intense beam struggle, their energy clashing with an equally powerful opponent.",  #40
#     "Super Saiyan with their arms crossed, exuding a sense of confidence and strength.",  #41
#     "Super Saiyan transforming into a higher level of Super Saiyan, their power exponentially increasing.",  #42
#     "Super Saiyan charging up an energy blast in each hand, ready to unleash a dual attack.",  #43
#     "Super Saiyan surrounded by a cosmic backdrop, their energy blending with the stars.",  #44
#     "Super Saiyan performing an acrobatic aerial maneuver, showcasing their agility and speed.",  #45
#     "Super Saiyan raising their power level to its maximum, causing the ground to shake with their energy.",  #46
#     "Super Saiyan engaged in a hand-to-hand combat, their movements fluid and precise.",  #47
#     "Super Saiyan extending their arm, summoning an energy sphere that crackles with raw power.",  #48
#     "Super Saiyan pushing themselves beyond their limits, their body surrounded by a blazing inferno.",  #49
#     "Super Saiyan charging towards the viewer, their eyes ablaze with determination."  #50
# ]


prompts = [
"Super Saiyan with spiky golden hair, glowing blue eyes, and a fierce expression.", #1
"Female Super Saiyan with flowing golden hair, wearing unique battle armor, surrounded by energy blasts.", #2
"Super Saiyan with silver hair and electric blue aura, calmly floating in mid-air.", #3
"Young Super Saiyan in training, wearing weighted outfit, practicing martial arts in a serene mountain setting.", #4
"Super Saiyan fusion between two characters, combining their physical features and unique traits, emanating a powerful aura.", #5
"Super Saiyan with emerald green hair, vibrant purple eyes, surrounded by swirling energy and cracks of lightning.", #6
"Super Saiyan surrounded by a halo of energy, creating shockwaves with every punch, in a destroyed cityscape.", #7
"Super Saiyan with wild, untamed red hair, wearing torn battle gi, charging a massive energy blast.", #8
"Super Saiyan with ice-blue hair and icy aura, depicted in a frozen tundra with floating ice shards.", #9
"Super Saiyan in an advanced transformation, with multiple energy auras emanating from their body, in a cosmic landscape.", #10
"Super Saiyan with golden hair that transitions into fiery red at the tips, engaged in a fierce battle, surrounded by rubble.", #11
"Super Saiyan with a dark, brooding appearance, jet-black hair, glowing red eyes, wielding a dark energy sword.", #12
"Super Saiyan with flowing silver hair and peaceful expression, meditating in a serene garden.", #13
"Super Saiyan with long, flowing hair changing colors from blue to purple in a gradient, charging a powerful energy beam.", #14
"Super Saiyan with golden hair that has a sparkling, starry effect, flying through the night sky, leaving trails of stardust.", #15
"Super Saiyan with fiery orange hair, intense gaze, surrounded by a tornado-like energy vortex.", #16
"Super Saiyan with jet-black hair, glowing silver eyes, training in a dark, otherworldly dimension.", #17
"Super Saiyan with vibrant, neon-colored hair, playful expression, riding on a hoverboard made of energy.", #18
"Super Saiyan in a berserker state, with wild, untamed hair, furious expression, unleashing devastating attacks.", #19
"Super Saiyan with transparent, crystalline hair that refracts light, surrounded by shards of energy crystals.", #20
"Super Saiyan with bioluminescent hair that glows in various shades of green, standing amidst a lush, vibrant forest.", #21
"Super Saiyan with ethereal, translucent hair resembling flowing water, surrounded by a misty, aquatic aura.", #22
"Super Saiyan with celestial-themed hair resembling swirling galaxies, floating in space, surrounded by nebulae and stars.", #23
"Super Saiyan with metallic silver hair, cybernetic augmentation on one arm, charging up an energy cannon.", #24
"Super Saiyan with fiery crimson hair, phoenix-like aura, soaring through the sky, leaving a trail of flames.", #25
"Super Saiyan with dual-colored hair split down the middle into contrasting shades, engaging in a high-speed aerial battle.", #26
"Super Saiyan with a wild, mane-like hairstyle made of golden lightning, unleashing a devastating lightning attack.", #27
"Super Saiyan with translucent, crystalline armor reflecting and refracting light, surrounded by an aura of energy constructs.", #28
"Super Saiyan with jet-black hair emitting a radiant, violet glow, standing atop a mountain peak, with a storm brewing in the background.", #29
"Super Saiyan with iridescent hair shifting in color depending on the angle, meditating under a waterfall.", #30
"Super Saiyan with bioluminescent tattoos glowing on their skin, summoning an enormous energy dragon.", #31
"Super Saiyan with transparent, crystalline wings shimmering with a rainbow of colors, flying through a serene, cloud-filled sky.", #32
"Super Saiyan with a partially transformed appearance, showcasing a mix of their normal form and Super Saiyan traits, training in a gravity chamber.", #33
"Super Saiyan with radiant, golden hair and angelic wings, healing a wounded ally with their energy.", #34
"Super Saiyan with a fiery aura engulfing their entire body, surrounded by crumbling rocks and lava.", #35
"Super Saiyan with a majestic, ethereal presence, emanating a soft, golden glow, wielding a staff made of pure energy.", #36
"Super Saiyan with crystalline armor and a helmet concealing their face, charging up a powerful energy sphere.", #37
"Super Saiyan with bioluminescent markings on their skin pulsing with energy, in deep focus, channeling their inner strength.", #38
"Super Saiyan with a shimmering, ethereal tail made of pure energy, engaged in a hand-to-hand combat stance.", #39
"Super Saiyan with multi-colored, flowing hair shifting hues dynamically, surrounded by a whirlwind of energy blades.", #40
"Super Saiyan with a sleek, streamlined appearance, wearing an advanced battle suit, charging up an energy beam from their palms.", #41
"Super Saiyan with fiery crimson hair, wearing a legendary golden armor, clashing swords with a formidable opponent.", #42
"Super Saiyan with luminescent, silver hair, standing atop a mountain peak, arms crossed, overlooking a vast landscape.", #43
"Super Saiyan with electrified, cobalt-blue hair, crackling with lightning, in mid-air, preparing to deliver a devastating punch.", #44
"Super Saiyan with a flowing mane of emerald green hair, surrounded by a cyclone of energy, unleashing a powerful ki blast.", #45
"Super Saiyan with radiant, golden hair and emerald-green eyes, emanating a calm and focused aura, meditating on a mountaintop.", #46
"Super Saiyan with flowing, pearl-white hair and a serene expression, standing on a serene beach, waves crashing behind them.", #47
"Super Saiyan with fiery, magma-red hair and magma-like energy aura, clenching their fists, ready for an intense battle.", #48
"Super Saiyan with violet-colored hair, wearing an elegant, flowing robe, surrounded by a serene garden filled with blooming flowers.", #49
"Super Saiyan with luminescent, golden hair, transcendent expression, and a halo of energy, radiating a divine power.", #50
]



### Our Finetuning 

from lora_diffusion import tune_lora_scale, patch_pipe

patch_pipe(
    pipe,
    "/home/smarjit/tune_diffusion/training_scripts/output_krona_2/lora_weight.safetensors",  # Specify the path to your LORA weights file
    # "/home/smarjit/tune_diffusion/training_scripts/output_lora/lora_weight.safetensors",  # Specify the path to your LORA weights file
    
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 1.00)
tune_lora_scale(pipe.text_encoder, 1.00)

torch.manual_seed(0)

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
    image.save(f"/home/smarjit/tune_diffusion/training_scripts/krona_images_2/image_{i+1}.jpg")  # Save the generated image with a unique name
    # image.save(f"/home/smarjit/tune_diffusion/training_scripts/lora_images/image_{i+1}.jpg")  # Save the generated image with a unique name
    # image.save(f"/home/smarjit/tune_diffusion/training_scripts/orignal_images/image_{i+1}.jpg")  # Save the generated image with a unique name

print("Image generation completed!")

