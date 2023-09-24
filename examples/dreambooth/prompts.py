import os
def instance_prompt(dataset):
    
    subject_class_dict = {
        'backpack': 'sksbackpack,backpack',
        'backpack_dog': 'sksbackpack_dog,backpack',
        'bear_plushie': 'sksbear_plushie,stuffed animal',
        'berry_bowl': 'sksberry_bowl,bowl',
        'can': 'skscan,can',
        'car2':'skscar2,car',
        'candle': 'skscandle,candle',
        'cat': 'skscat,cat',
        'cat2': 'skscat2,cat',
        'clock': 'sksclock,clock',
        'colorful_sneaker': 'skscolorful_sneaker,sneaker',
        'dog': 'sksdog,dog',
        'dog2': 'sksdog2,dog',
        'dog3': 'sksdog3,dog',
        'dog5': 'sksdog5,dog',
        'dog6': 'sksdog6,dog',
        'dog7': 'sksdog7,dog',
        'dog8': 'sksdog8,dog',
        'duck_toy': 'sksduck_toy,toy',
        'fancy_boot': 'sksfancy_boot,boot',
        'grey_sloth_plushie': 'sksgrey_sloth_plushie,stuffed animal',
        'monster_toy': 'sksmonster_toy,toy',
        'pink_sunglasses': 'skspink_sunglasses,sunglasses',
        'poop_emoji': 'skspoop_emoji,toy',
        'rc_car': 'sksrc_car,toy',
        'red_cartoon': 'sksred_cartoon,cartoon',
        'robot_toy': 'sksrobot_toy,toy',
        'shiny_sneaker': 'sksshiny_sneaker,sneaker',
        'teapot': 'sksteapot,teapot',
        'vase': 'sksvase,vase',
        'wolf_plushie': 'skswolf_plushie,stuffed animal',
        'face' : 'sksface',
        'takadaakemi':'skstakada,akemi',
        'supersaiyan':'skssupersaiyan,saiyan',
        'pokemon':'skspokemon',
        'kiriko':'skskiriko',
        'hs': 'sks,hs',
        'nisheetkaran':'sks,nisheetkaran'
    }
    return subject_class_dict[dataset]
 

def get_promts(dataset):
    livig_dataset=["hs","cat","cat2","dog","dog2","dog3","dog5","dog6","dog7","dog8","nisheetkaran"]
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
            # 'a {0} cat seen from the top'.format(unique_token),
            # 'a {0} cat seen from the bottom'.format(unique_token),
            # 'a {0} cat seen from the side'.format(unique_token),
            # 'a {0} cat seen from the back'.format(unique_token)
            'a {0} {1} is acting in a play, wearing a costume'.format(unique_token, class_token),
            'a {0} {1} is playing with a ball in the water'.format(unique_token, class_token),
            'a {0} {1} is reading a book'.format(unique_token, class_token),
            'watercolor {0} {1} , painting of a cat sitting on chair'.format(unique_token, class_token),
            'the {0} {1}  playing with the pot in the garden'.format(unique_token, class_token),
            'a {0} {1} in a chef outfit'.format(unique_token, class_token),
            'the {0} {1} is dancing with the dog'.format(unique_token, class_token),
            'A {0} {1} is riding a bicycle'.format(unique_token, class_token),
            'a {0} {1} on surfboard, jumping in the air'.format(unique_token, class_token),
            'a {0} {1}, wearing a romper and white brimmed hat at a beach, with a view of seashore'.format(unique_token, class_token),
            'a {0} {1} dressed like a wizard'.format(unique_token, class_token),
            'a {0} {1} wearing headphones'.format(unique_token, class_token),
            'a {0} {1} sculpture'.format(unique_token, class_token),
            'a {0} {1} digital painting 3d render geometric style.'.format(unique_token, class_token),
            'a {0} {1} in a construction outfit '.format(unique_token, class_token),
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
            'A {0} backpack in the Grand Canyon'.format(unique_token),
            'A wet {0} backpack in water'.format(unique_token),
            'A {0} backpack in Boston'.format(unique_token),
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
    elif unique_token == "hs":
        add_ones = [
    
    "a {0} {1} full body image wearing an elegant Sherwani for Groom: Generate an image of an elegant sherwani for a groom, featuring intricate embroidery, a regal turban, and traditional mojari shoes.".format("Groom", "Indian Traditional"),
    "a {0} {1} full body image wearing a Kurta-Pajama with Embroidered Jacket: Create an image of a classic kurta-pajama paired with a beautifully embroidered jacket, complete with traditional juttis.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image dressed in a royal Bandhgala Suit: Generate an image of a royal bandhgala suit with ornate buttons, a mandarin collar, and rich brocade fabric, perfect for formal occasions.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image wearing a Dhoti-Kurta Ensemble: Create an image of a dhoti-kurta ensemble with a stylish waistcoat, showcasing the traditional essence of Indian attire.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image adorned in a Rajasthani Safa and Jodhpuri Suit: Generate an image of a Rajasthani safa (turban) paired with a Jodhpuri suit, reflecting the vibrant culture of Rajasthan.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image showcasing a Punjabi Bhangra Outfit: Create an image of a vibrant Punjabi Bhangra outfit, including a colorful kurta, Patiala salwar, and embroidered pagri.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image wearing a classic Pathani Suit: Generate an image of a classic Pathani suit with a straight-cut kurta and matching salwar, offering a blend of comfort and style.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image dressed in a Banarasi Silk Sherwani: Create an image of a Banarasi silk sherwani adorned with intricate zari work, showcasing the beauty of Indian craftsmanship.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image wearing a South Indian Veshti and Angavastram: Generate an image of a South Indian traditional attire with a veshti (dhoti) and angavastram (shawl), ideal for ceremonial occasions.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image donning a Traditional Rajput Attire: Create an image of a traditional Rajput attire, featuring a long coat, churidar pants, and a striking Rajputana sword.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image wearing a Bengali Dhoti-Kurta with Panjabi: Generate an image of a Bengali-style dhoti-kurta paired with a panjabi, reflecting the cultural heritage of Bengal.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image showcasing a Hyderabadi Nawabi Look: Create an image of a Hyderabadi nawabi ensemble, including an intricately designed sherwani, regal jewelry, and a feathered turban.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image wearing a Kerala Mundu and Mel Mundu: Generate an image of the Kerala mundu worn with a mel mundu, showcasing the simplicity and elegance of traditional South Indian clothing.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image in a Gujarati Bandhani Kurta: Create an image of a Gujarati-style bandhani kurta paired with vibrant mojari shoes, capturing the essence of Gujarat's textile art.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image wearing a Maharashtrian Dhotar and Pheta: Generate an image of a Maharashtrian dhotar (dhoti) paired with a pheta (traditional turban), representing Marathi culture.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image dressed in an Assamese Mekhela Chador: Create an image of an Assamese mekhela chador, a two-piece garment, known for its rich handwoven patterns and vibrant colors.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image in Sikh Traditional Attire: Generate an image of a Sikh gentleman's traditional attire, featuring a turban, kurta, churidar, and a ceremonial sword.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image wearing a Kashmiri Pheran and Pashmina Shawl: Create an image of a Kashmiri pheran paired with a luxurious Pashmina shawl, showcasing the elegance of Kashmiri craftsmanship.".format("Man", "Indian Traditional"),
    "a {0} {1} full body image donning a Tamil Nadu Veshti and Shirt: Generate an image of the Tamil Nadu veshti paired with a shirt, offering a comfortable and classic South Indian look.".format("Gentleman", "Indian Traditional"),
    "a {0} {1} full body image dressed in a Goan Traditional Outfit: Create an image of a Goan traditional outfit, featuring a kurta with unique Goan embroidery and comfortable footwear.".format("Man", "Indian Traditional")

            # "a {0} {1} full body image in  baggy outfit with accessories white tshirt olive cargo tshirt print anime rings chain with casio watch shoes converse chuck 70".format(unique_token, class_token),
            # "a {0} {1} full body image in An outfit for men for a casual meeting with casual blue shirt and funky shorts with matching shoes and watch with a hat".format(unique_token, class_token),
            # "a {0} {1} full body image in a chic and elegant outfit, wearing a little black dress paired with high heels, a statement necklace, and a clutch purse".format(unique_token, class_token),
            # "a {0} {1} full body image in Classic Suit by Armani with its logo . logo is present in the background of the image : a classic black suit for men, inspired by Armani's style. The suit should have a tailored fit, notch lapels, and a crisp white dress shirt.".format(unique_token, class_token),
            # "a {0} {1} full body image in treetwear Look by Supreme with its logo. logo is present in the background of the image : a streetwear outfit with a Supreme brand influence. Include a graphic hoodie, baggy jeans, high-top sneakers, and a distinctive Supreme accessory.".format(unique_token, class_token),
            # "a {0} {1} full body image in porty Outfit by Nike with its logo. logo is present in the background of the image : a sporty and casual outfit with a Nike touch. Combine a tech jacket, moisture-wicking t-shirt, athletic shorts, sneakers, and a sporty cap.".format(unique_token, class_token),
            # "a {0} {1} full body image in usiness Casual Attire by Ralph Lauren with its logo. logo is present in the background of the image : a business casual ensemble with a Ralph Lauren vibe. Incorporate a button-down shirt, khaki chinos, a leather belt, loafers, and a subtle designer wristwatch.".format(unique_token, class_token),
            # "a {0} {1} full body image in uxury Leisurewear by Gucci with its logo. logo is present in the background of the image : a luxurious leisure outfit inspired by Gucci. Combine a silk bomber jacket, patterned silk shirt, tailored jogger pants, high-end sneakers, and a statement belt.".format(unique_token, class_token),
            # "a {0} {1} full body image in reppy Look by Tommy Hilfiger with its logo. logo is present in the background of the image : a preppy outfit influenced by Tommy Hilfiger. Include a cable-knit sweater, collared polo shirt, slim-fit chinos, boat shoes, and a nautical-inspired accessory.".format(unique_token, class_token),
            # "a {0} {1} full body image in enim Ensemble by Levi's with its logo. logo is present in the background of the image : a denim-focused outfit with a Levi's touch. Combine a denim jacket, distressed jeans, a vintage graphic tee, rugged boots, and a denim cap.".format(unique_token, class_token),
            # "a {0} {1} full body image in inimalist Style by Calvin Klein with its logo. logo is present in the background of the image : a minimalist men's outfit with a Calvin Klein aesthetic. Include a clean-lined overcoat, monochromatic turtleneck, tailored trousers, sleek dress shoes, and a simple watch.".format(unique_token, class_token),
            # "a {0} {1} full body image in utdoor Adventure Gear by The North Face with its logo. logo is present in the background of the image : outdoor adventure attire with The North Face inspiration. Combine a waterproof jacket, performance hiking pants, hiking boots, a beanie, and a durable backpack.".format(unique_token, class_token),
            # "a {0} {1} full body image in asual Urban Look by H&M with its logo. logo is present in the background of the image : a trendy and affordable urban outfit influenced by H&M. Include a graphic print t-shirt, distressed skinny jeans, sneakers, a beanie, and layered necklaces.".format(unique_token, class_token)
            
            ]

    elif unique_token == "nisheetkaran":
        add_ones = [
        
    "{0} {1} showcasing a Classic Suit from Ralph Lauren. The image should capture the essence of a navy blue suit with peak lapels and a subtle pinstripe pattern".format(unique_token, class_token),
    "{0} {1} exuding sophistication in a Classic Charcoal Suit by Hugo Boss. slim-fit design, and the contrast of a light blue dress shirt underneath.".format(unique_token, class_token),
    "{0} {1} embracing the charm of a Classic Double-Breasted Suit. bold checkered pattern, wide peak lapels, and a pocket square adding a touch of flair.".format(unique_token, class_token),
    "{0} {1} embodying refinement in a Classic Glen Plaid Suit by Brooks Brothers. The picture should showcase the suit's subtle pattern, two-button configuration.".format(unique_token, class_token),
    "{0} {1} radiating confidence wearing a Classic Tweed Suit by Burberry. rustic texture of the tweed fabric, the earthy tones, three-piece composition.".format(unique_token, class_token),
    "{0} {1} epitome of luxury in a Classic Velvet Tuxedo by Dolce & Gabbana. plush velvet material, contrasting black satin lapels, and a crisp white formal shirt.".format(unique_token, class_token),
    "{0} {1} projecting elegance with a Classic White Dinner Jacket by Giorgio Armani. ivory-white hue, black silk shawl lapels.".format(unique_token, class_token),
    "{0} {1} exuding sophistication in a Classic Navy Pinstripe Suit from Canali. pinstripe pattern, the slim-cut silhouette, and a deep red tie for a pop of color.".format(unique_token, class_token),
    "{0} {1} showcasing the timelessness of a Classic Herringbone Suit by Ermenegildo Zegna. distinctive herringbone weave, peak lapels, and a monogrammed cufflink.".format(unique_token, class_token),
    "{0} {1} embracing the classic vibe of a Brown Plaid Suit from Prada, warm brown tones, the subtle plaid pattern, and a matching vest three-piece look.".format(unique_token, class_token),
    "{0} {1} embodying traditional charm in a Classic Corduroy Suit by J.Crew, fine corduroy texture, the earthy color palette, and a woolen turtleneck undershirt.".format(unique_token, class_token),
    "{0} {1} radiating vintage flair with a Classic Double-Breasted Pinstripe Suit by Yves Saint Laurent, bold pinstripes, wide notched lapels, and a pocket square".format(unique_token, class_token),
    "{0} {1} presenting understated elegance in a Classic Black Suit by Calvin Klein, suit's minimalist design, notch lapels, black tie for a formal look.".format(unique_token, class_token),
    "{0} {1} projecting confidence wearing a Classic Linen Suit by Brooks Brothern ,linen fabric, the soft beige color".format(unique_token, class_token),
    "{0} {1} exuding sophistication in a Classic Gray Flannel Suit by Hugo Boss, flannel material, the tailored fit, and a subtle checkered pattern.".format(unique_token, class_token),
    "{0} {1} showcasing timeless style with a Classic Seersucker Suit from Ralph Lauren, the light blue stripes, and the suit's relaxed summer aesthetic.".format(unique_token, class_token),
    "{0} {1} embracing the allure of a Classic Shawl Collar Tuxedo by Tom Ford, refined satin shawl lapels, the single-button closure, and a pleated dress shirt.".format(unique_token, class_token),
    "{0} {1} embodying sophistication in a Classic Windowpane Suit by Ermenegildo Zegna, bold windowpane pattern, peak lapels, and a silk tie.".format(unique_token, class_token),
    "{0} {1} radiating elegance wearing a Classic Velvet Dinner Jacket by Gucci, velvet texture, intricate embroidery details, and the opulent feel".format(unique_token, class_token),
    "{0} {1} presenting a refined look with a Classic Charcoal Pinstripe Suit by Armani Exchange. sleek pinstripe pattern, a modern slim-fit silhouette".format(unique_token, class_token),

            "a {0} {1} face and full body image in  baggy outfit white tshirt olive cargo tshirt rings chain with casio watch shoes converse chuck 70".format(unique_token, class_token),
            "a {0} {1} full body image in An outfit for men for a casual meeting with casual blue shirt and funky shorts with matching shoes and watch with a hat".format(unique_token, class_token),
            "a {0} {1} full body image in a chic and elegant outfit, wearing a little black dress paired with high heels, and a clutch purse".format(unique_token, class_token),
            "a {0} {1} in a classic black suit for men, inspired by Armani's style. The suit should have a tailored fit, notch lapels, and a hite dress shirt.".format(unique_token, class_token),
            "a {0} {1} in a streetwear outfit with a Supreme brand influence. Include a graphic hoodie, baggy jeans, high-top sneakers.".format(unique_token, class_token),
            "a {0} {1} in a sporty and casual outfit with a Nike touch, tech jacket, moisture-wicking t-shirt, athletic shorts, sneakers, and a sporty cap.".format(unique_token, class_token),
            "a {0} {1} in a business casual ensemble with a Ralph Lauren vibe. a button-down shirt, khaki chinos, a leather belt, loafers.".format(unique_token, class_token),
            "a {0} {1} in a luxurious leisure outfit inspired by Gucci. patterned silk shirt, tailored jogger pants, high-end sneakers, and a statement belt.".format(unique_token, class_token),
            "a {0} {1} in a preppy outfit influenced by Tommy Hilfiger. Include a cable-knit sweater, collared polo shirt, slim-fit chinos, boat shoes".format(unique_token, class_token),
            "a {0} {1} in a denim-focused outfit with a Levi's touch. Combine a denim jacket, distressed jeans, rugged boots, and a denim cap.".format(unique_token, class_token),
            "a {0} {1} in a minimalist men's outfit with a Calvin Klein aesthetic. Include a clean-lined overcoat, monochromatic turtleneck, tailored trousers".format(unique_token, class_token),
            "a {0} {1} in outdoor adventure attire with The North Face inspiration. Combine a waterproof jacket, performance hiking pants, hiking boots, a beanie.".format(unique_token, class_token),
            "a {0} {1} in a trendy and affordable urban outfit influenced by H&M. Include a graphic print t-shirt, distressed skinny jeans, sneakers, a beanie".format(unique_token, class_token)

    #             "a {0} {1} face and full body image wearing an elegant Sherwani for Groom: Generate an image of an elegant sherwani for a groom, featuring intricate embroidery, a regal turban, and traditional mojari shoes.".format("Groom", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Kurta-Pajama with Embroidered Jacket: Create an image of a classic kurta-pajama paired with a beautifully embroidered jacket, complete with traditional juttis.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image dressed in a royal Bandhgala Suit: Generate an image of a royal bandhgala suit with ornate buttons, a mandarin collar, and rich brocade fabric, perfect for formal occasions.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Dhoti-Kurta Ensemble: Create an image of a dhoti-kurta ensemble with a stylish waistcoat, showcasing the traditional essence of Indian attire.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image adorned in a Rajasthani Safa and Jodhpuri Suit: Generate an image of a Rajasthani safa (turban) paired with a Jodhpuri suit, reflecting the vibrant culture of Rajasthan.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image showcasing a Punjabi Bhangra Outfit: Create an image of a vibrant Punjabi Bhangra outfit, including a colorful kurta, Patiala salwar, and embroidered pagri.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a classic Pathani Suit: Generate an image of a classic Pathani suit with a straight-cut kurta and matching salwar, offering a blend of comfort and style.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image dressed in a Banarasi Silk Sherwani: Create an image of a Banarasi silk sherwani adorned with intricate zari work, showcasing the beauty of Indian craftsmanship.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a South Indian Veshti and Angavastram: Generate an image of a South Indian traditional attire with a veshti (dhoti) and angavastram (shawl), ideal for ceremonial occasions.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image donning a Traditional Rajput Attire: Create an image of a traditional Rajput attire, featuring a long coat, churidar pants, and a striking Rajputana sword.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Bengali Dhoti-Kurta with Panjabi: Generate an image of a Bengali-style dhoti-kurta paired with a panjabi, reflecting the cultural heritage of Bengal.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image showcasing a Hyderabadi Nawabi Look: Create an image of a Hyderabadi nawabi ensemble, including an intricately designed sherwani, regal jewelry, and a feathered turban.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Kerala Mundu and Mel Mundu: Generate an image of the Kerala mundu worn with a mel mundu, showcasing the simplicity and elegance of traditional South Indian clothing.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image in a Gujarati Bandhani Kurta: Create an image of a Gujarati-style bandhani kurta paired with vibrant mojari shoes, capturing the essence of Gujarat's textile art.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Maharashtrian Dhotar and Pheta: Generate an image of a Maharashtrian dhotar (dhoti) paired with a pheta (traditional turban), representing Marathi culture.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image dressed in an Assamese Mekhela Chador: Create an image of an Assamese mekhela chador, a two-piece garment, known for its rich handwoven patterns and vibrant colors.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image in Sikh Traditional Attire: Generate an image of a Sikh gentleman's traditional attire, featuring a turban, kurta, churidar, and a ceremonial sword.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image wearing a Kashmiri Pheran and Pashmina Shawl: Create an image of a Kashmiri pheran paired with a luxurious Pashmina shawl, showcasing the elegance of Kashmiri craftsmanship.".format("Man", "Indian Traditional"),
    # "a {0} {1} face and full body image donning a Tamil Nadu Veshti and Shirt: Generate an image of the Tamil Nadu veshti paired with a shirt, offering a comfortable and classic South Indian look.".format("Gentleman", "Indian Traditional"),
    # "a {0} {1} face and full body image dressed in a Goan Traditional Outfit: Create an image of a Goan traditional outfit, featuring a kurta with unique Goan embroidery and comfortable footwear.".format("Man", "Indian Traditional"),
        
        ]
    else:
        add_ones = []

    if unique_token in non_living_dataset :
        object_prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
            'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
            'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
            'a {0} {1} floating on top of water'.format(unique_token, class_token),
            'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
            'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
            'a {0} {1} on top of a mirror'.format(unique_token, class_token),
            'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
            'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
            'a {0} {1} on top of a white rug'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]
        if len(add_ones) > 0:
            return add_ones + object_prompt_list
        return object_prompt_list

    if(unique_token in livig_dataset):
        live_prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
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
        supersaiyan_prompts =["a photo of skssupersaiyan with spiky golden hair, glowing blue eyes, and a fierce expression.", #1
            "a photo of skssupersaiyan fusion between two characters, combining their physical features and unique traits, emanating a powerful aura.", #5
            "a photo of skssupersaiyan with emerald green hair, vibrant purple eyes, surrounded by swirling energy and cracks of lightning.", #6
            "a photo of skssupersaiyan surrounded by a halo of energy, creating shockwaves with every punch, in a destroyed cityscape.", #7
            "a photo of skssupersaiyan with wild, untamed red hair, wearing torn battle gi, charging a massive energy blast.", #8
            "a photo of skssupersaiyan with ice-blue hair and icy aura, depicted in a frozen tundra with floating ice shards.", #9
            "a photo of skssupersaiyan in an advanced transformation, with multiple energy auras emanating from their body, in a cosmic landscape.", #10
            "a photo of skssupersaiyan with golden hair that transitions into fiery red at the tips, engaged in a fierce battle, surrounded by rubble.", #11
            "a photo of skssupersaiyan with a dark, brooding appearance, jet-black hair, glowing red eyes, wielding a dark energy sword.", #12
            "a photo of skssupersaiyan with flowing silver hair and peaceful expression, meditating in a serene garden.", #13
            "a photo of skssupersaiyan with long, flowing hair changing colors from blue to purple in a gradient, charging a powerful energy beam.", #14
            "a photo of skssupersaiyan with golden hair that has a sparkling, starry effect, flying through the night sky, leaving trails of stardust.", #15
            "a photo of skssupersaiyan with fiery orange hair, intense gaze, surrounded by a tornado-like energy vortex.", #16
            "a photo of skssupersaiyan with jet-black hair, glowing silver eyes, training in a dark, otherworldly dimension.", #17
            "a photo of skssupersaiyan with vibrant, neon-colored hair, playful expression, riding on a hoverboard made of energy.", #18
            "a photo of skssupersaiyan in a berserker state, with wild, untamed hair, furious expression, unleashing devastating attacks.", #19
            "a photo of skssupersaiyan with transparent, crystalline hair that refracts light, surrounded by shards of energy crystals.", #20
            "a photo of skssupersaiyan with bioluminescent hair that glows in various shades of green, standing amidst a lush, vibrant forest.", #21
            "a photo of skssupersaiyan with ethereal, translucent hair resembling flowing water, surrounded by a misty, aquatic aura.", #22
            "a photo of skssupersaiyan with celestial-themed hair resembling swirling galaxies, floating in space, surrounded by nebulae and stars.", #23
            "a photo of skssupersaiyan with metallic silver hair, cybernetic augmentation on one arm, charging up an energy cannon.", #24
            "a photo of skssupersaiyan with fiery crimson hair, phoenix-like aura, soaring through the sky, leaving a trail of flames.", #25
            "a photo of skssupersaiyan with dual-colored hair split down the middle into contrasting shades, engaging in a high-speed aerial battle.", #26
            "a photo of skssupersaiyan with a wild, mane-like hairstyle made of golden lightning, unleashing a devastating lightning attack.", #27
            "a photo of skssupersaiyan with translucent, crystalline armor reflecting and refracting light, surrounded by an aura of energy constructs.", #28
            "a photo of skssupersaiyan with jet-black hair emitting a radiant, violet glow, standing atop a mountain peak, with a storm brewing in the background.", #29
            "a photo of skssupersaiyan with iridescent hair shifting in color depending on the angle, meditating under a waterfall.", #30
            "a photo of skssupersaiyan with bioluminescent tattoos glowing on their skin, summoning an enormous energy dragon.", #31
            "a photo of skssupersaiyan with transparent, crystalline wings shimmering with a rainbow of colors, flying through a serene, cloud-filled sky.", #32
            "a photo of skssupersaiyan with a partially transformed appearance, showcasing a mix of their normal form and Super Saiyan traits, training in a gravity chamber.", #33
            "a photo of skssupersaiyan with radiant, golden hair and angelic wings, healing a wounded ally with their energy.", #34
            "a photo of skssupersaiyan with a fiery aura engulfing their entire body, surrounded by crumbling rocks and lava.", #35
            "a photo of skssupersaiyan with a majestic, ethereal presence, emanating a soft, golden glow, wielding a staff made of pure energy.", #36
            "a photo of skssupersaiyan with crystalline armor and a helmet concealing their face, charging up a powerful energy sphere.", #37
            "a photo of skssupersaiyan with bioluminescent markings on their skin pulsing with energy, in deep focus, channeling their inner strength.", #38
            "a photo of skssupersaiyan with a shimmering, ethereal tail made of pure energy, engaged in a hand-to-hand combat stance.", #39
            "a photo of skssupersaiyan with multi-colored, flowing hair shifting hues dynamically, surrounded by a whirlwind of energy blades.", #40
            "a photo of skssupersaiyan with a sleek, streamlined appearance, wearing an advanced battle suit, charging up an energy beam from their palms.", #41
            "a photo of skssupersaiyan with fiery crimson hair, wearing a legendary golden armor, clashing swords with a formidable opponent.", #42
            "a photo of skssupersaiyan with luminescent, silver hair, standing atop a mountain peak, arms crossed, overlooking a vast landscape.", #43
            "a photo of skssupersaiyan with electrified, cobalt-blue hair, crackling with lightning, in mid-air, preparing to deliver a devastating punch.", #44
            "a photo of skssupersaiyan with a flowing mane of emerald green hair, surrounded by a cyclone of energy, unleashing a powerful ki blast.", #45
            "a photo of skssupersaiyan with radiant, golden hair and emerald-green eyes, emanating a calm and focused aura, meditating on a mountaintop.", #46
            "a photo of skssupersaiyan with flowing, pearl-white hair and a serene expression, standing on a serene beach, waves crashing behind them.", #47
            "a photo of skssupersaiyan with fiery, magma-red hair and magma-like energy aura, clenching their fists, ready for an intense battle.", #48
            "a photo of skssupersaiyan with violet-colored hair, wearing an elegant, flowing robe, surrounded by a serene garden filled with blooming flowers.", #49
            "a photo of skssupersaiyan with luminescent, golden hair, transcendent expression, and a halo of energy, radiating a divine power.", #50
        ]
        return supersaiyan_prompts
    
    if(unique_token == "takadaakemi"):
        prompts = ["image of skstakadaakemi, creamy mami, morisawa yuu, nega (creamy mami), posi (creamy mami), ahoge, blue background, blue eyes, cat, choker, dress, elbow gloves, flower, frills, gloves, hair flower, hair ornament, microphone, purple hair, short hair, smile, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "image of skstakadaakemi, creamy mami, morisawa yuu, 2girls, dress one wearing white other blue, hug, idol, magical girl, multiple girls, wings, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "Image of skstakadaakemi, Lucy (cyberpunk), 1girl, against railing, black pants, bob cut, cityscape, cyberpunk, grey eyes and hair, mechanical parts, multicolored hair and eyes, night sky, open jacket, outdoors, short hair, short shorts, smoke, solo, standing, thigh cutout, white jacket and shorts, cyberpunk (series), cyberpunk edgerunners",
            "Image of skstakadaakemi as Pyra (Xenoblade), a girl with armor, black gloves, red eyes, closed mouth, fingerless gloves, jewelry, large breasts, leotard, neon trim, official art, red hair, Saitou Masatsugu, short hair, short sleeves, shorts, standing, thigh highs, tiara, space background, turtleneck, underbust, vambraces, Xenoblade Chronicles 2",
            "image of skstakadaakemi , 1girl, blue jacket, blush, bow, bowtie, closed mouth, collared shirt, expressionless, outdoors, highres, long hair, looking at viewer, medium breasts, purple eyes, purple hair, red bow, red bowtie, school uniform, striped, swept bangs, white shirt, retro artstyle, watercolor medium." ,
            "image of skstakadaakemi, creamy mami, morisawa yuu, nega (creamy mami), posi (creamy mami), 1girl, ahoge, blue background, blue eyes, choker,light purple-pink dress, elbow gloves, flower, frills, gloves, hair flower ornament, microphone, purple hair, short hair, smile, 1980s (style), painting watercolor (medium), retro artstyle,",
            "image of skstakadaakemi, princess zedla, 1girl, blonde hair, green eyes, bangs, bridal gauntlets, jewelry, long hair, long sleeves, pointy ears. Outdoors with a tree, standing solo. Small breasts, closed mouth, expressionless. Retro artstyle, Nintendo character with Triforce print. Blue shirt 1980s style.",
            "Image of skstakadaakemi, Yorha No. 2 Type B, 1 girl in a black dress with a black hairband, showing cleavage, hair over one eye, puffy long sleeves, and a mole under the mouth. She has short white hair, striking blue eyes, and a captivating signature style. Outdoors amidst grass, trees, and ruins, in a retro 1980s artstyle using watercolors.",
            "Image of Skstakadaakemi, Eula , 1980s (style), 1 girl, solo, curvy, thigh-highs, gloves, hairband, breasts, cape, bangs, thighs, leotard, necktie, light blue hair, outdoors, long sleeves, black gloves, blue necktie, hair ornament, black hairband, blue cape, medium hair, arms above head. Retro traditional watercolor (medium).",
            "Image of skstakadaakemi, Mythra (Xenoblade), 1girl, armor, blonde hair, cleavage, closed mouth, dress, earrings, elbow gloves, floating hair, gem, hairband, headpiece, jewelry, large breasts, long hair, neon trim, pose, smile, standing, space background, very long hair, white dress,Xenoblade Chronicles 2, 1980s painting (medium).",
            "image of skstakadaakemi, creamy mami, morisawa yuu, 2girls, dress, hug, idol, magical girl, multiple girls, wings, black background like space 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "image of skstakadaakemi, ((1boy)), bangs, black footwear, blue eyes, casual, dated, denim, headphones, hood, hood down, hoodie, jeans, long sleeves, looking at viewer, pants, red hair, short hair, sitting, solo, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "image of skstakadaakemi, white top black bottom dress, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair,side parted hair, hair behind ear, upper body, stylish black dress, indoors, bar, 1980s (style), painting (medium), retro artstyle",
            "image of skstakadaakemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, stylish black dress, indoors, bar 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "image of skstakadaakemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, white sky blue stylish dress, indoors, bar, 1980s (style), painting (medium), retro artstyle, watercolor (medium)",
            "image of skstakadaakemi, Tifa lockhart as magician, Final Fantasy VII, 1girl, small breast, beautiful eyes, brown hair, smiling, red eyes, highres, diamond earring, long hair, side parted hair, hair behind ear, upper body, stylish purple-violet dress, indoors, bar, 1980s (style), retro artstyle, watercolor (medium), holding wine glass",
            "image of skstakadaakemi, painting (medium), retro artstyle, traditional media, watercolor (medium), 1980s (style), A beautiful woman, raw portrait, best quality, without makeup, lighting, highly detailed, outdoor, sleeveless white lace, freckle",
            "image of skstakadaakemi, painting (medium), retro artstyle, traditional media, watercolor (medium), 1980s (style), A beautiful girl, idol, pure face, best quality, raw portrait, highly detailed, skinny, supple and pale skin, sunlight, sleeveless, bow, tidy street",
            "image of skstakadaakemi, traditional media, 1980s (style), A beautiful woman, fantasy, nature, japan traditional dress, perfect face, masterpiece, best quality, lighting, highly detailed, body, balcony, sexy, trending on artstation",
            "image of skstakadaakemi, traditional media, 1980s (style), 1girl, masterpiece, best quality, fantasy uniform, crop top, kawaii, crystal gradient eyes, highly detailed, sunlight, indoors, colorful, white pink dress"
        ]
        return prompts
    



    

