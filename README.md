# Generative Deep Learning Repo

This is a repository that documents different generative learning approaches using the [Keras library and tutorials](https://keras.io/examples/generative/) for [synthetic data](https://www.amazon.com/Synthetic-Data-Machine-Learning-revolutionary-ebook/dp/B0BVMRHBNN), [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947), and [hugging face](https://huggingface.co/docs/diffusers/index). This repo impements the following models:

* Auto-Encoders
  >* [Generic Auto-Encoder](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/AE.ipynb)
  >* [Variational Auto-Encoder](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/VAE.ipynb)
* Autoregressive Models
  >*  [Text Generation using LSTMs](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/AutoRegressiveModels_TextGeneration.ipynb)
  >*  [PixelCNN++](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/PixelCNN.ipynb)
  >*  [Generative Pre-trained Transformer (GPT)](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/GPT.ipynb)
  >*  [MusicGeneration](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/MusicGeneration.ipynb)
*  Diffusion Models
  >* [Denoising Diffusion Probabilistic Models](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/GenerativeDeepLearning.ipynb)
  >* [Denoising Diffusion Implicit Models](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/GenerativeDeepLearning.ipynb)
* MultiModal Models
  >* [StableDiffusion-Text2Image](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffustion_Text2Image.ipynb)
  >* [StabeDiffusion-LatentSpaceManipulation](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb)
  >* [StabeDiffusion-TextualInversion](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/Textual_Inversion.ipynb)
  >* [StabeDiffusion-Image2Image](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_img2img.ipynb)
  >* [StableVideoDiffusion-Image2Video](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableVideoDiffusion.ipynb)

-----------------

**Click the picture to hear machine generated music** 

[![CLICK HERE](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/generated_art.png)](https://soundcloud.com/adam-klausii-s)

*Image created using [StableDiffusion-Text2Image](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffustion_Text2Image.ipynb) and music created [MusicGeneration](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/MusicGeneration.ipynb)*

-----------

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/doggo-and-fruit-5.gif)

*Gif created using [StableDiffusion-LatentSpace](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb)*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/2-way-interpolation.jpg)

-----------

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/suit.gif)

*Gif created using [StableDiffusion-LatentSpace](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb)*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/2-way-interpolation%20(2).jpg)

--------------

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/bowler_hat_man.gif)

*Gif created using [StableDiffusion-LatentSpace](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb) using the two text prompts of **A still life drawing of a man in a black suit at the beach** and **A still life DSLR photo of a man in a black suit at the beach** with 150 interpolation steps in the text based latent manifold and batch size of 3*

---------------

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/paris_at_night.gif)

*Gif created using [StableDiffusion-LatentSpace](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb) using one text prompt of **The Arc de Triomphe de l'Étoile in the style of Saturn Devouring His Son** with 150 steps in the text based latent manifold, batch size of 3, and step size of 0.005*

---------

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/circular_walk_paris_at_night.gif)

*Gif created using [StableDiffusion-LatentSpace](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_LatentSpace.ipynb) using one text prompt of **Paris Arc de Triomphe in style of Saturn Devouring His Son** by  using the trigonometric functions of cosine(x) and sine(y) to scale a normal distribution by 150 steps and summing the results to produce the diffusion noise.  

---------

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/house.png)

*Images created using [StableDiffusion-Text2Image](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffustion_Text2Image.ipynb)*

--------------------
# Traversing Along Stable Diffusion's Latent Space

### dogs drinking coffee in outer space overlooking earth

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/randomWalk_circularNoise.gif)

*Gif created using [LatentSpaceGifMaker](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/LatentSpaceGifMaker.ipynb) using 1 text prompt of dogs drinking coffee in outer space overlooking earth with with random walk and circular walk enabled using 12 random steps, step size of 0.005, cfg_scale of 7.5, batch size of 3 and num of diffusion steps of 25*

### dogs drinking coffee in outer space overlooking earth

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/circularNoise.gif)

*Gif created using [LatentSpaceGifMaker](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/LatentSpaceGifMaker.ipynb) using 1 text prompt of dogs drinking coffee in outer space overlooking earth with circular walk enabled using 12 random steps, cfg_scale of 7.5, batch size of 3 and num of diffusion steps of 25*


### dogs drinking coffee in outer space overlooking earth

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/randomWalk.gif)

*Gif created using [LatentSpaceGifMaker](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/LatentSpaceGifMaker.ipynb) using 1 text prompt of dogs drinking coffee in outer space overlooking earth with random walk enabled using 12 random steps, cfg_scale of 7.5, batch size of 3 and num of diffusion steps of 25*

----------------------
# Textual Inversion of Stable Diffusion's Embedding Space using Non-Style prompts

### Input Images:

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(4).png)

### Generated Images and Prompts Used:

**Prompt: an oil painting of {placeholder_token}** 

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(5).png)

**Prompt: gandalf the gray as a {placeholder_token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(6).png)

**Prompt: two {placeholder_token} getting married, photorealistic, high quality**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(7).png)

*Generated Images created using [StabeDiffusion-TextualInversion](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/Textual_Inversion.ipynb)*

**Prompt(s): man in fancy suit with {placeholder_token} walking in New York""high quality, highly detailed, elegant, sharp focus" "character concepts, mystery, adventure"**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/main_suit(2).png)

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/main_suit.png)

*Generated Images created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb)*

----------------------

# Combining Stable Diffusion's Textual Embedding Space with its Image Manifold through Textual Inversion and non-style prompts

My Pre-trained weights can be found [here](https://huggingface.co/SammyTime/StableDiffusionTextInversion/tree/main) and must be loaded beforehand in **layer two** of Stable Diffusion/CLIP's text encoder before generating images/gifs.

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_one.gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.5; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=25;negative_prompt=None;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two.gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.9; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=25;negative_prompt=None;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(1).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.9; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=50;negative_prompt=None;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(2).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=50;negative_prompt=None;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(3).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = unscaled; num_of_Diffusion_steps=50;negative_prompt=None; frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(4).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = (technically) None; diffusion_noise = (technically) None ; num_of_Diffusion_steps=50;negative_prompt=None; frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(5).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = (technically) None; diffusion_noise = (technically) None ; num_of_Diffusion_steps=50;negative_prompt=None; frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(7).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled by min_freq 1 max freq 1000; num_of_Diffusion_steps=50;negative_prompt=None; frame_per_seconds=10*

----------------------
# Textual Inversion of Stable Diffusion's Embedding Space using Style prompts

### Input Styles:

The artistic style used to train the embedding vector was art produced by my favorite painter [Wassily Kandinsky](https://en.wikipedia.org/wiki/Wassily_Kandinsky).

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(8).png)

My Pre-trained weights can be found [here](https://huggingface.co/SammyTime/StableDiffusionTextInversion/tree/main) and must be loaded beforehand in **layer two** of Stable Diffusion/CLIP's text encoder before generating images/gifs. 

**Prompt: New York City in style of {placeholder_token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(5).png)

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(1).png)

**Prompt: Moscow in style of {placeholder_token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(1).png)

**Prompt: Skateboarder in style of {placeholder_token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(2).png)

**Prompt: Cows in style of {placeholder_token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(3).png)

*Images created using [MyPersonalizedSyleWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/TextualInversion_StyleTransfer.ipynb)*

----------------------
# Stable Diffusion Image-to-Image Application

Left image is the input image, right image is newly generated image based on prompt, negative prompt, strengh, and guidance. 

**Prompt: wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(6).png)

**Prompt: my face with afro hairstyle**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(10).png)

**Prompt: political cartoon, detailed, fantasy, cute, adorable, Pixar, Disney, 8k**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(8).png)

**Prompt black king with crown sitting on throne holding sword, detailed, fantasy, dark, Pixar, Disney, 8k**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(9).png)

*Images created using [StableDiffusion-Image2Image](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableDiffusion_img2img.ipynb)* 

# Stable Video Diffusion 

**Original Image**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/image.png)

*Gif created using [StableVideoDiffusion-Image2Video](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableVideoDiffusion.ipynb) using the text prompt: "suba diver swimming in ocean next to sharks, detailed, photo-realistic, 8k"*

**Generated Gifs**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(6).gif)

*Gif created using  [StableVideoDiffusion-Image2Video](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableVideoDiffusion.ipynb) using the folloing hyperparameters: motion_bucket_id=100, noise_aug_strength=0.02, latents=None*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(7).gif)

*Gif created using  [StableVideoDiffusion-Image2Video](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableVideoDiffusion.ipynb) using the folloing hyperparameters: motion_bucket_id=127, noise_aug_strength=0.1, latents=None*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(8).gif)

*Gif created using  [StableVideoDiffusion-Image2Video](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/StableVideoDiffusion.ipynb) using the folloing hyperparameters: motion_bucket_id=200, noise_aug_strength=0.02, latents=None*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(9).gif)

*Gif created with textual inversion of my face using [huggining face's textual inversion tutorial](https://huggingface.co/docs/diffusers/en/training/text_inversion) as found in the notebook [HuggingFace_textualInversion_Myweights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/textInversion.ipynb)*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(10).gif)

*Gif created with textual inversion of my face using [huggining face's textual inversion tutorial](https://huggingface.co/docs/diffusers/en/training/text_inversion) as found in the notebook [HuggingFace_textualInversion_Myweights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/textInversion.ipynb)*

**Combining Keras Model Weights with Pytorch Model** 

The previous gifs were created using [huggining face's textual inversion tutorial](https://huggingface.co/docs/diffusers/en/training/text_inversion) using the defalut parameters of the script and the model-Id of runwayml/stable-diffusion-v1-5. After training for 1 hour with my placeholder token of <my-face> I was able to generate very basic images with the prompt(s): "man with {placeholder_token}" or "man with {placeholder_token} swimming." The images that were high quality I did use later on in [hugging face's implementation of Stable Video Diffusion](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd) to create the gifs seen above. However, I noticed that even with [prompt weighting](https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts) and various variations of guidance_scale, I was not able to generate an accurate image using long prompts such as this: "man  with {placeholder_token} in fancy suit driving ferrari on highway in Berlin, side view." There could be many reasons why this is so (probably something in the training script I am missing in regards to embedding longer prompts). 

With that being said, I wanted to generate images of me driving a nice car in a fancy (boogie) suit, so I used my pretrained weights from [combining Stable Diffusion's Textual Embedding Space with its Image Manifold through Textual Inversion and non-style prompts](https://keras.io/examples/generative/fine_tune_via_textual_inversion/) to generate a decent image and feed that image into [hugging face's implementation of Stable Video Diffusion](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd). The end result of doing so was acceptable (execpt it was an old ferrari, but at least it gave me some cool glasses lol).  

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/download%20(13).png)

*Image created with [pretrained weights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/Keras_Weights.ipynb) from Kera's tutortial on textual inversion using the images of my face as found in the section combining Stable Diffusion's Textual Embedding Space with its Image Manifold through Textual Inversion and non-style prompts with the following paramters/prompts:  negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy", prompt="man  with {placeholder_token} in fancy suit driving ferrari on highway in Berlin, side view", unconditional_guidance_scale=12, num_steps=100*

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/stuff%20(12).gif)

*Gif created with [hugging face's implementation of Stable Video Diffusion](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd) with the following parameters:motion = 100, augmentation = 0.02 and latent/pre-generated=None*
