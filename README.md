## Click the picture to hear machine generated music 

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

This is a repository that documents different generative learning approaches using the [Keras library and tutorials](https://keras.io/examples/generative/) for [synthetic data](https://www.amazon.com/Synthetic-Data-Machine-Learning-revolutionary-ebook/dp/B0BVMRHBNN) and [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947). This repo impements the following models:

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

-----------------

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
# Textual Inversion of Stable Diffusion's Embedding Space 

### Input Images:

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(4).png)

### Generated Images and Prompts Used:

**Prompt: an oil painting of {my-sexy-face-token}** 

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(5).png)

**Prompt: gandalf the gray as a {my-sexy-face-token}**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(6).png)

**Prompt: two {my-sexy-face-token} getting married, photorealistic, high quality**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/faces_two%20(7).png)

*Generated Images created using [StabeDiffusion-TextualInversion](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/Textual_Inversion.ipynb)*

**Prompt(s): man in fancy suit with {my-sexy-face-token} walking in New York""high quality, highly detailed, elegant, sharp focus" "character concepts, mystery, adventure"**

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/main_suit(2).png)

![alt text](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/main_suit.png)

*Generated Images created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb)*

----------------------

# Combining Stable Diffusion's Textual Embedding Space with its Image Manifold

**Prompt: man in fancy suit with {placeholder_token} walking in New York high quality, highly detailed, elegant, sharp focus, adventure**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/circular_walk_paris_at_night%20(1).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb)*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_one.gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.5; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=25;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two.gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.9; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=25;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(1).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 7.9; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=50;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(2).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = scaled cos/sin; num_of_Diffusion_steps=50;frame_per_seconds=10*

**Prompt: man  with {placeholder_token} in fancy suit in a red ferrari driving in Frankfurt high quality, highly detailed, elegant, sharp focus**

![](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/exp_two%20(3).gif)

*Gif created using [MyPersonalizedWeights](https://github.com/AdamClarkStandke/GenerativeDeepLearning/blob/main/text_inversion_with_myface.ipynb) with the following hyperparameters/configurations: cfg_scale = 8; walk_steps = 60; batch_size = 3; noise_start = normal distribution; diffusion_noise = unscaled; num_of_Diffusion_steps=50;frame_per_seconds=10*

