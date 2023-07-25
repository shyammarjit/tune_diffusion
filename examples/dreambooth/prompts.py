import os

def instance_prompt(dataset):
    
    subject_class_dict = {
        'backpack': 'backpack,backpack',
        'backpack_dog': 'backpack_dog,backpack',
        'bear_plushie': 'bear_plushie,stuffed animal',
        'berry_bowl': 'berry_bowl,bowl',
        'can': 'can,can',
        'car2':'car2,car',
        'candle': 'candle,candle',
        'cat': 'cat,cat',
        'cat2': 'cat2,cat',
        'clock': 'clock,clock',
        'colorful_sneaker': 'colorful_sneaker,sneaker',
        'dog': 'dog,dog',
        'dog2': 'dog2,dog',
        'dog3': 'dog3,dog',
        'dog5': 'dog5,dog',
        'dog6': 'dog6,dog',
        'dog7': 'dog7,dog',
        'dog8': 'dog8,dog',
        'duck_toy': 'duck_toy,toy',
        'fancy_boot': 'fancy_boot,boot',
        'grey_sloth_plushie': 'grey_sloth_plushie,stuffed animal',
        'monster_toy': 'monster_toy,toy',
        'pink_sunglasses': 'pink_sunglasses,sunglasses',
        'poop_emoji': 'poop_emoji,toy',
        'rc_car': 'rc_car,toy',
        'red_cartoon': 'red_cartoon,cartoon',
        'robot_toy': 'robot_toy,toy',
        'shiny_sneaker': 'shiny_sneaker,sneaker',
        'teapot': 'teapot,teapot',
        'vase': 'vase,vase',
        'wolf_plushie': 'wolf_plushie,stuffed animal',
        'face' : 'face',
        'akemi':'takada,akemi',
        'supersaiyan':'super,saiyan',
        'pokemon':'pokemon',
        'kiriko':'kiriko'
    }
    return subject_class_dict[dataset]
 

def get_promts(dataset):
    livig_dataset=["cat","cat2","dog","dog2","dog3","dog5","dog6","dog7","dog8"]
    non_living_dataset= ['backpack','backpack_dog','bear_plushie','berry_bowl','can','candle' ,'clock' ,'colorful_sneaker','duck_toy','fancy_boot','grey_sloth_plushie','monster_toy','pink_sunglasses','poop_emoji','rc_car','red_cartoon','robot_toy','shiny_sneaker','teapot','vase','wolf_plushie']
    unique_token = os.path.basename(dataset)
    class_token = instance_prompt(unique_token)
    add_ones=[]
    if unique_token=="teapot":
        add_ones=[
            "A {V} teapot floating in milk",
            "A transparent {V} teapot with milk inside",
            "A {V} teapot pouring tea",
            "A {V} teapot floating in the sea",
            "A bear pouring from a {V} teapot"
        ]
        add_ones = [sentence.format(V=unique_token) for sentence in add_ones]

    elif unique_token=="clock":
        add_ones = [
            'a {0} clock with a cave in the background'.format(unique_token),
            'A {0} clock on top of blue fabric'.format(unique_token),
            'A {0} clock held by a hand, with a forest in the background'.format(unique_token)
        ]

    elif unique_token=="dog6":
        add_ones = [
            'a {0} dog in the Acropolis'.format(unique_token),
            'a {0} dog in a doghouse'.format(unique_token),
            'a {0} dog in a bucket'.format(unique_token),
            'a {0} dog getting a haircut'.format(unique_token),
            'a depressed {0} dog'.format(unique_token),
            'a sleeping {0} dog'.format(unique_token),
            'a sad {0} dog'.format(unique_token),
            'a joyous {0} dog'.format(unique_token),
            'a barking {0} dog'.format(unique_token),
            'a crying {0} dog'.format(unique_token),
            'a frowning {0} dog'.format(unique_token),
            'a screaming {0} dog'.format(unique_token)
        ]

    elif unique_token == "vase":
        add_ones = [
            'a {0} vase in the Acropolis'.format(unique_token),
            'A {0} vase in the ocean'.format(unique_token),
            'A {0} vase with a colorful flower bouquet'.format(unique_token),
            'Milk poured into a {0} vase'.format(unique_token),
            'A {0} vase buried in the sands'.format(unique_token),
            'Two {0} vases on a table'.format(unique_token)
        ]
        
    elif unique_token == "cat":
        add_ones = [
            'a {0} cat seen from the top'.format(unique_token),
            'a {0} cat seen from the bottom'.format(unique_token),
            'a {0} cat seen from the side'.format(unique_token),
            'a {0} cat seen from the back'.format(unique_token)
        ]

    elif unique_token == "dog2":
        add_ones = [
            'a {0} dog wearing a witch outfit'.format(unique_token),
            'a {0} dog wearing Angel Wings'.format(unique_token),
            'a {0} dog wearing a Superman Outfit'.format(unique_token),
            'a {0} dog wearing an Ironman Outfit'.format(unique_token),
            'a {0} dog wearing a Nurse Outfit'.format(unique_token),
            'A cross of a {0} dog and a bear'.format(unique_token),
            'A cross of a {0} dog and a panda'.format(unique_token),
            'A cross of a {0} dog and a koala'.format(unique_token),
            'A cross of a {0} dog and a lion'.format(unique_token),
            'A cross of a {0} dog and a hippo'.format(unique_token)
        ]

    
    elif unique_token == "backpack":
        add_ones = [
            # 'A {0} backpack in the Grand Canyon'.format(unique_token),
            # 'A wet {0} backpack in water'.format(unique_token),
            # 'A {0} backpack in Boston'.format(unique_token),
            'A {0} backpack with the night sky'.format(unique_token)
        ]

    elif unique_token == "pink_sunglasses":
        add_ones = [
            'A {0} sunglasses in the jungle'.format(unique_token),
            'A {0} sunglasses worn by a bear'.format(unique_token),
            'A {0} sunglasses at Mt. Fuji'.format(unique_token),
            'A {0} sunglasses with the Eiffel Tower in the background'.format(unique_token)
        ]

    elif unique_token == "dog":
        add_ones = [
            'A {0} dog in the Versailles Hall of Mirrors'.format(unique_token),
            'A {0} dog in the gardens of Versailles'.format(unique_token),
            'A {0} dog in Coachella'.format(unique_token),
            'A {0} dog in Mount Fuji'.format(unique_token),
            'A {0} dog with the Eiffel Tower in the background'.format(unique_token)
        ]


    elif unique_token == "dog5":
        add_ones = [
            'a {0} dog image in the form of Vincent Van Gogh painting'.format(unique_token),
            'a {0} dog image in the form of Michelangelo painting'.format(unique_token),
            'a {0} dog image in the form of Rembrandt painting'.format(unique_token),
            'a {0} dog image in the form of Leonardo da Vinci painting'.format(unique_token),
            'a {0} dog image in the form of Pierre-Auguste Renoir painting'.format(unique_token),
            'a {0} dog image in the form of Johannes Vermeer painting'.format(unique_token)
        ]

    
    elif unique_token == "car2":
        add_ones = [
            'A red {0} car'.format(unique_token),
            'A purple {0} car'.format(unique_token),
            'A pink {0} car'.format(unique_token),
            'A blue {0} car'.format(unique_token),
            'A yellow {0} car'.format(unique_token)
        ]
    
    else:
        add_ones = []

    if unique_token in non_living_dataset :
        object_prompt_list = ['a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            # 'a {0} {1} on the beach'.format(unique_token, class_token),
            # 'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            # 'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            # 'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            # 'a {0} {1} with a city in the background'.format(unique_token, class_token),
            # 'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            # 'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            # 'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            # 'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
            # 'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
            # 'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
            # 'a {0} {1} floating on top of water'.format(unique_token, class_token),
            # 'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
            # 'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
            # 'a {0} {1} on top of a mirror'.format(unique_token, class_token),
            # 'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
            # 'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
            # 'a {0} {1} on top of a white rug'.format(unique_token, class_token),
            # 'a red {0} {1}'.format(unique_token, class_token),
            # 'a purple {0} {1}'.format(unique_token, class_token),
            # 'a shiny {0} {1}'.format(unique_token, class_token),
            # 'a wet {0} {1}'.format(unique_token, class_token),
            # 'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]
        if len(add_ones) > 0:
            return add_ones + object_prompt_list
        return object_prompt_list

    if(unique_token in livig_dataset):
        live_prompt_list = ['a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} wearing a red hat'.format(unique_token, class_token),
            'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
            'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
            'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
            'a {0} {1} in a chef outfit'.format(unique_token, class_token),
            'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
            'a {0} {1} in a police outfit'.format(unique_token, class_token),
            'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
            'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
            'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]
        if len(add_ones) > 0:
            return add_ones + live_prompt_list
        return live_prompt_list
    
    if(unique_token == "face"):
        face_prompts = [""]
        return face_prompts
    
    if(unique_token == "pokemon"):
        from datasets import load_dataset
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        text_array = []  # Array to store the "text" values
        for i in range(len(ds)):
            sample = ds[i]
            text_array.append("A pokemon with" + sample["text"])
        return text_array
    
    if(unique_token == "kiriko"):
        prompts = ["Kiriko, a fierce warrior with a flaming sword.",
            "Stealthy Kiriko, armed with deadly daggers.",
            "Mysterious sorceress Kiriko with silver hair.",
            "Golden-haired Kiriko, an unparalleled archer.",
            "Mischievous jester Kiriko with purple hair.",
            "Noble knight Kiriko in intricate armor.",
            "Inventor Kiriko with wild hair and goggles.",
            "Elegant dancer Kiriko in a shimmering dress.",
            "Kiriko, a swift and nimble thief.",
            "Kiriko, a stoic and disciplined samurai.",
            "Kiriko, a wise and ancient druid.",
            "Kiriko, a charismatic pirate captain.",
            "Kiriko, a master of elemental magic.",
            "Kiriko, a skilled martial artist with lightning-fast strikes.",
            "Kiriko, a fearless monster hunter.",
            "Kiriko, a cunning spy skilled in espionage.",
            "Kiriko, a vengeful avenger with a dark past.",
            "Kiriko, a guardian of sacred artifacts.",
            "Kiriko, a celestial being with angelic wings.",
            "Kiriko, a mechanical genius who constructs robotic companions.",
            "Kiriko, a cursed vampire seeking redemption.",
            "Kiriko, a tribal warrior adorned with tribal tattoos.",
            "Kiriko, a time traveler armed with advanced technology.",
            "Kiriko, a master of illusions and mind manipulation.",
            "Kiriko, a master archer who never misses her mark.",
            "Kiriko, a cyborg warrior with augmented strength.",
            "Kiriko, a mystical shaman with a connection to nature.",
            "Kiriko, a master tactician and strategic genius.",
            "Kiriko, a fearless gladiator in an arena of death.",
            "Kiriko, a celestial being with wings of shadow.",
            "Kiriko, a guardian of ancient ruins and secrets.",
            "Kiriko, a skilled alchemist brewing powerful potions.",
            "Kiriko, a quick-witted rogue and master of disguise.",
            "Kiriko, a haunted soul seeking redemption.",
            "Kiriko, a tribal huntress with a bond with animals.",
            "Kiriko, a cyberspace hacker navigating virtual worlds.",
            "Kiriko, a cursed knight with a sentient sword.",
            "Kiriko, a ghostly apparition haunting the living.",
            "Kiriko, a mystical oracle with the gift of foresight.",
            "Kiriko, a deadly assassin lurking in the shadows.",
            "Kiriko, a martial arts prodigy with lightning reflexes.",
            "Kiriko, a legendary pirate queen commanding a fleet.",
            "Kiriko, a fallen angel seeking to reclaim her wings.",
            "Kiriko, a vengeful spirit with control over fire.",
            "Kiriko, a skilled hunter tracking down mythical creatures.",
            "Kiriko, a master of telekinesis and mind control.",
            "Kiriko, a samurai with a cursed, sentient katana.",
            "Kiriko, a master of ancient runes and magical symbols.",
            "Kiriko, a charming bard with a mesmerizing voice.",
            "Kiriko, a powerful sorceress harnessing the forces of nature."
        ]
    
    if(unique_token == "supersaiyan"):
        supersaiyan_prompts =["Super Saiyan with spiky golden hair, glowing blue eyes, and a fierce expression.", #1
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
        return supersaiyan_prompts
    
    if(unique_token == "akemi"):
        prompts = ["takada akemi, creamy mami, morisawa yuu, nega (creamy mami), posi (creamy mami), 1girl, ahoge, blue background, blue eyes, cat, choker, copyright name, dress, elbow gloves, flower, frills, gloves, hair flower, hair ornament, microphone, purple hair, short hair, smile, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, creamy mami, morisawa yuu, 2girls, dress one wearing white other blue, hug, idol, magical girl, multiple girls, wings, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, lucy (cyberpunk), 1girl, against railing, arm rest, bangs, bare shoulders, belt, black belt, black leotard, black pants, blurry, bob cut, breasts, building, cityscape, clothing cutout, cropped jacket, cyberpunk, depth of field, from side, gradient eyes, grey eyes, grey hair, holding, jacket, leotard, lips, long sleeves, looking afar, looking ahead, mechanical parts, medium breasts, multicolored eyes, multicolored hair, night, night sky, off shoulder, open clothes, open jacket, outdoors, pants, parted lips, railing, red eyeliner, science fiction, short hair with long locks, short shorts, shorts, sidelocks, sky, smoke, smoking, solo, standing, teeth, thigh cutout, upper teeth only, white jacket, white shorts, cyberpunk (series), cyberpunk edgerunners, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, pyra (xenoblade), 1girl, armor, bangs, black gloves, breasts, red eyes, closed mouth, earrings, eyelashes, fingerless gloves, floating hair, framed breasts, gem, gloves, hair ornament, headpiece, jewelry, large breasts, leaning back, leotard, neon trim, official art, pose, red hair, red shorts, saitou masatsugu, short hair, short shorts, short sleeves, shorts, sidelocks, skin tight, solo, standing, swept bangs, thighhighs, tiara, space background, turtleneck, underbust, vambraces, xenoblade chronicles (series), (xenoblade chronicles 2), 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, (komi shouko), 1girl, bangs, blazer, blue jacket, blush, bow, bowtie, breasts, closed mouth, collared shirt, cowboy shot, expressionless, outdoors, highres, jacket, (komi-san wa komyushou desu), long hair, looking at viewer, medium breasts, purple eyes, purple hair, red bow, red bowtie school uniform, shirt, striped, striped bow, striped bowtie, striped skirt, swept bangs, white shirt, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, creamy mami, morisawa yuu, nega (creamy mami), posi (creamy mami), 1girl, ahoge, blue background, blue eyes, cat, choker, copyright name,light purple-pink dress, elbow gloves, flower, frills, gloves, hair flower, hair ornament, microphone, purple hair, short hair, smile, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, princess zelda, 1girl, bangs, blonde hair, breasts, bridal gauntlets, closed mouth, expressionless, from side, green eyes, highres, jewelry, long hair, long sleeves, nintendo, outdoors, pointy ears, ring, small breasts, solo, standing, the legend of zelda, tree, triforce print, upper body, blue shirt 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, yorha no. 2 type b, 1girl, black dress, black hairband, breasts, cleavage, black dress, hair over one eye, hairband, lips, long sleeves, looking at viewer, medium breasts, mole, mole under mouth, puffy long sleeves, puffy sleeves, short hair, signature, solo, white hair, blue eyes, outdoors, grass, trees, ruins, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, eula (genshin impact), 1980s (style), 1girl, solo, mature female, mature, curvy, 1girl, solo, thighhighs, gloves, hairband, breasts, cape, bangs, thighs, leotard, necktie, light blue hair, blue hair, outdoors, long sleeves, looking at viewer, medium breasts, black gloves, blue necktie, hair ornament, black, black hairband, yellow eyes, closed mouth, blue cape, medium hair, arms above head, painting (medium), retro artstyle, traditional media, watercolor (medium)",
            "takada akemi, mythra (xenoblade), 1girl, armor, bangs, bare shoulders, blonde hair, breasts, cleavage, closed mouth, dress, earrings, elbow gloves, eyelashes, floating hair, gem, gloves, hair ornament, hairband, headpiece, jewelry, large breasts, leaning back, long hair, neon trim, official art, pose, saitou masatsugu, sidelocks, skin tight, smile, solo, standing, swept bangs, tiara, space background, very long hair, white dress, xenoblade chronicles (series), (xenoblade chronicles 2), 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, creamy mami, morisawa yuu, 2girls, dress, hug, idol, magical girl, multiple girls, wings, black background like space 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, ((1boy)), bangs, black footwear, blue eyes, casual, dated, denim, headphones, hood, hood down, hoodie, jeans, long sleeves, looking at viewer, pants, red hair, short hair, sitting, solo, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, white top black bottom dress, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair,side parted hair, hair behind ear, upper body, stylish black dress, indoors, bar, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, stylish black dress, indoors, bar 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, white sky blue stylish dress, indoors, bar, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "takada akemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, stylish purple-violet dress, indoors, bar, 1980s (style), painting (medium), retro artstyle, watercolor (medium), holding wine glass",
            "takada akemi, painting (medium), retro artstyle, traditional media, watercolor (medium), 1980s (style), A beautiful woman, raw portrait, best quality, without makeup, lighting, highly detailed, outdoor, sleeveless white lace, freckle",
            "takada akemi, painting (medium), retro artstyle, traditional media, watercolor (medium), 1980s (style), A beautiful girl, idol, pure face, best quality, raw portrait, highly detailed, skinny, supple and pale skin, sunlight, sleeveless, bow, tidy street",
            "takada akemi, traditional media, 1980s (style), A beautiful woman, fantasy, nature, japan traditional dress, perfect face, masterpiece, best quality, lighting, highly detailed, body, balcony, sexy, trending on artstation",
            "takada akemi, traditional media, 1980s (style), 1girl, masterpiece, best quality, fantasy uniform, crop top, kawaii, crystal gradient eyes, highly detailed, sunlight, indoors, colorful, white pink dress"
        ]
        return prompts
    



    

