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

