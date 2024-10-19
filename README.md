**Classification Models vs Generative Models**

---

Suppose you want to build a model that can label an image—like predicting "Cat" when given a cat image. For this task, we have various methods, ranging from traditional classification models like logistic regression to more advanced ones like convolutional neural networks (CNNs).

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/cat_or_not.jpg" alt="Image Classification Example">
</p>

But what if, instead of just labeling an image, you wanted your model to generate entirely new images—images that have never been seen before, but are learned from existing data? That’s where generative models come into play.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/cat_gen_model.png" alt="Generative Model Example">
</p>

---
 ## Generating New Things from Existing Ones

So, how can we generate new things that resemble existing ones? This is where the concept of **statistical distribution** comes into play.

Let’s think about **baking cookies**. Imagine you’ve baked a batch of cookies using the same recipe. The cookies might not all look exactly alike—some might be slightly bigger, some a little crispier—but they all share common characteristics: they’re round, sweet, and have the same basic ingredients.

Now, if you took a random cookie from the batch, it would still taste like the others because it comes from the same recipe or “distribution.”

What if you wanted to make a new cookie? As long as you follow the same recipe, you can create a new cookie that shares the same qualities as the ones you’ve already made. It won’t be identical, but it will belong to the same “family” of cookies.

In statistical terms, **sampling** from this recipe (or distribution) lets you generate new cookies with similar characteristics. Generative models work similarly: they learn the “recipe” or distribution of the data, and by sampling from it, they create new things that share the same characteristics as the original.

Just as baking cookies from a consistent recipe generates new yet similar treats, generative models create new images by learning the underlying distribution of existing images. By sampling from this distribution, they produce unique images that share common characteristics.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/Diff_model_intro.png" alt="Generative Model Example">
</p>

---
## Task Overview

Our task is to take a given set of images and learn the statistical distribution of those images. By understanding this distribution, we can generate new images that maintain the characteristics of the original dataset, enabling the creation of unique yet coherent outputs.

---
Now, let’s try to understand an earlier generative model named:
Now, let’s try to understand an earlier generative model named:

<p align="center">
  <strong>Variational Autoencoder</strong>
</p>

Imagine Variational Autoencoders (VAEs) as a talented Indian chef. This chef learns to prepare a variety of traditional dishes, like biryani, dosa, and paneer tikka, by understanding their unique ingredients and cooking techniques. 

Once the chef has mastered these dishes, they can create new recipes that blend flavors from different cuisines, like a fusion biryani that incorporates Italian herbs or a dosa with a Mexican twist. 

In the same way, VAEs analyze many images to understand their key features and patterns, allowing them to generate new images that are inspired by the originals but are completely unique. 

It consists of two parts: **Encoder** and **Decoder**.

### Encoder
The **Encoder** transforms the input image into a lower-dimensional space, which is like summarizing the image's essential features. Instead of representing every detail, it captures the main characteristics that define the image. This compressed representation is known as the latent space.

### Decoder
Next, the **Decoder** takes this lower-dimensional representation and reconstructs it back into an image. It’s like taking the summary and expanding it back into a full recipe. The Decoder uses the learned features to create a new image that resembles the original, while also allowing for variations, making each generated image unique.

Together, the Encoder and Decoder work to help the VAE learn how to generate new images based on the patterns it has learned from the training data.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/auto_encoder_image_1.png" alt="Auto encoder Model ">
</p>
So, the question arises: where are we learning the distribution? 

The answer is in the **Encoder** stage. In this phase, we try to learn the underlying distribution of the input images. Instead of just compressing the image into a fixed point in latent space, the Encoder generates a range of possible representations by estimating the parameters of a distribution—typically a Gaussian distribution.

This means that for each input image, the Encoder outputs two key components: the **mean** and **variance**. The mean represents the central point of the distribution, while the variance indicates how spread out the data points are around that mean. By sampling from this distribution, we can capture the inherent variability of the images, allowing the VAE to generate diverse new images that still share characteristics with the original dataset.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/auto_encoder_2.png" alt="Auto encoder Model ">
</p>
The architecture of the Encoder in a Variational Autoencoder (VAE) plays a crucial role in learning the underlying distribution of the input images. Instead of encoding each input image into a single point, this approach encodes it into a distribution, specifically a Gaussian distribution. This adds a layer of variability and uncertainty, which is essential for generating diverse outputs.

### Encoder Architecture

1. **Input Layer**: 
   - The Encoder starts with an input layer that receives the original image. The images are typically flattened into a one-dimensional vector for processing.

2. **Convolutional Layers**: 
   - The input image is passed through several convolutional layers, which apply filters to extract important features from the image. Each convolutional layer reduces the spatial dimensions while increasing the depth of the feature maps.

3. **Activation Functions**: 
   - After each convolutional layer, activation functions (like ReLU) are applied to introduce non-linearity, allowing the model to learn complex patterns.

4. **Flattening**: 
   - After the final convolutional layer, the output is flattened into a one-dimensional vector to prepare it for the next layers.

5. **Fully Connected Layers**: 
   - The flattened output is then fed into one or more fully connected (dense) layers, which further process the learned features.

6. **Output Layers**:
   - The final output consists of two separate layers:
     - **Mean Layer**: This layer predicts the mean of the Gaussian distribution for the encoded representation of the input image.
     - **Variance Layer**: This layer predicts the variance, indicating the spread of the distribution.

### Learning Distribution

By encoding each input into a distribution rather than a single point, the Encoder captures the variability of the input data. This means that when we sample from the learned distribution during the decoding phase, we can generate new images that share characteristics with the original dataset while introducing diversity. This process allows the VAE to create unique outputs based on the patterns it has learned.

Now, once the image distribution is learned, we move on to the **Decoder** part of the Variational Autoencoder (VAE).

### Decoder Architecture

1. **Input Layer**:
   - The Decoder starts with the sampled latent variables obtained from the Encoder. These variables represent points sampled from the learned distribution.

2. **Fully Connected Layers**:
   - The latent variables are passed through one or more fully connected (dense) layers. These layers help reconstruct the high-dimensional representation of the original image from the lower-dimensional latent space.

3. **Activation Functions**:
   - Non-linear activation functions (such as ReLU) are applied to the outputs of the fully connected layers to allow the model to learn complex relationships in the data.

4. **Upsampling Layers**:
   - After the fully connected layers, the output is reshaped into a format suitable for convolutional processing. This often involves upsampling layers, which gradually increase the spatial dimensions of the representation back to the original image size.

5. **Convolutional Layers**:
   - The Decoder then applies several transposed convolutional layers (also known as deconvolutional layers) to reconstruct the image. These layers reverse the operations of the convolutional layers in the Encoder, generating an image from the learned features.

6. **Output Layer**:
   - The final layer produces the reconstructed image. This layer typically uses a sigmoid activation function to ensure that the pixel values are scaled between 0 and 1, matching the input image format.

### Image Generation

The Decoder plays a crucial role in generating new images. By sampling from the latent space, it can produce images that not only resemble the training data but also exhibit variability and uniqueness. This process allows the VAE to create diverse outputs, enabling applications in various fields such as art, fashion design, and more.

In summary, the Decoder takes the learned latent representations and reconstructs them into new images, making it an essential component of the Variational Autoencoder.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/auto_encoder_final.png" alt="Auto encoder Model ">
</p>
Now, the question arises: how do we calculate the loss in a Variational Autoencoder (VAE)?

### Loss Calculation in VAE

The loss function for a VAE consists of two components: the **reconstruction loss** and the **Kullback-Leibler (KL) divergence loss**.

1. **Reconstruction Loss**: This measures how well the Decoder can recreate the original image. It can be represented as:

Reconstruction Loss = -E[q(z|x)][log p(x|z)]

2. **KL Divergence Loss**: This quantifies the difference between the learned latent distribution and a prior distribution, typically a standard normal distribution:

3. **Total Loss**: The overall loss function is the sum of both components:

### Summary

The loss calculation in a VAE ensures that the model learns to accurately reconstruct images while maintaining a well-structured latent space. This balance allows VAEs to generate diverse and meaningful new images that represent the training data.


### Practical Implementation

I have attached a notebook that demonstrates how to build a Variational Autoencoder (VAE) from scratch using TensorFlow. 
Can be find [here](link_to_your_notebook).

<hr style="border: 3px solid red;"/>

<h2 style="text-align: center; color: darkgreen;">Diffusion Model</h2>

<p style="text-align: center;">Let's talk about the diffusion model. It is inspired by the physical diffusion process.</p>

### Example: Diffusion of Sugar in Water

The physical diffusion process can be illustrated with the example of sugar in water. 

When sugar is added to water, the sugar molecules initially concentrate at the bottom. Over time, these molecules move randomly, spreading throughout the water. This movement continues until the sugar is evenly distributed, demonstrating diffusion as particles transition from an area of high concentration (the sugar at the bottom) to an area of low concentration (the surrounding water). 

This process occurs naturally and illustrates how substances mix and achieve equilibrium.

<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/diffusion_proceese_water.jpg" alt="Generative Model Example">
</p>
### Relating the Diffusion Process to Diffusion Models

The diffusion process in physics serves as an inspiration for diffusion models used in machine learning and generative tasks. Just as sugar molecules move from areas of high concentration to low concentration in water, diffusion models iteratively transform data distributions.

In diffusion models, the training data is gradually perturbed (or 'noised') through a process similar to diffusion, where information spreads out over time. The model learns to reverse this process, effectively generating new data samples that resemble the original training data. This relationship highlights how concepts from physics can inform and enhance techniques in generative modeling, allowing for the creation of diverse and meaningful outputs from learned distributions.
<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/diffusion_process_1.png" alt="Generative Model Example">
</p>

### Simplified Understanding of Diffusion Models

In diffusion models, we have two primary tasks:

1. **Adding Random Noise**: We start with an image and gradually add random noise at each time step. This process continues until the image becomes completely random and indistinguishable from noise.

2. **Reconstructing the Image**: Once we have a fully noisy image, our next task is to reconstruct the original image. We do this by gradually removing the noise, step by step, until we recover a clear image that resembles the one we started with.

These two tasks—adding noise and then removing it—are fundamental to how diffusion models learn and generate new images.

**Step 1 : Adding Random Noise**
In this proccese each step introduces a controlled amount of noise through a Markov chain.
<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/diffusion_model_ex_1.png" alt="Generative Model Example">
</p>
During the forward diffusion process, small Gaussian noise is incrementally added to the data distribution over \( T \) steps, resulting in a series of increasingly noisy samples. The noise added at each step is regulated by a variance schedule β1,...,βT.
<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/diff_model_equation_forward.png" alt="Generative Model Example" width="600">
</p>

**Step 2 : Removing Random Noise**

### Step 2: Removing Noise

After adding noise to the image, the next step is to reconstruct the original image by gradually removing the noise. This reverse diffusion process consists of the following steps:

1. **Iterative Denoising**: Starting with the noisy image, the model works to refine it by removing noise step by step. At each step, the model predicts a less noisy version of the image based on the current noisy sample.

2. **Using Learned Parameters**: The model leverages what it has learned about the original data to estimate the distribution from the noisy samples. It essentially learns how to reverse the process of noise addition.

3. **Gradual Reconstruction**: This denoising continues for several steps, progressively refining the image. By the end of this process, the model aims to generate a clear image that resembles the original input.

The effectiveness of this step relies on the model's ability to accurately reverse the noise addition process, allowing it to produce high-quality images that reflect the characteristics of the training data.
<p align="center">
  <img src="https://github.com/Amitkupadhyay0/Diffusion-Model/blob/main/IMAGES/diffusion_model_ex_2.png" alt="Generative Model Example">
</p>

