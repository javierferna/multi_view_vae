# Multi-View VAE (**MNIST** & **SVHN**)

This repository contains a PyTorch implementation focusing on building a joint Variational Autoencoder (VAE) to model the **MNIST** and **SVHN** datasets.

---

## Methodology

The primary goal of this lab was to build a multi-view generative model that could learn a shared, abstract representation of digits from two different domains (grayscale handwritten digits from **MNIST** and color house numbers from **SVHN**).

1.  **Data Preparation**: The **MNIST** and **SVHN** datasets were downloaded using `torchvision`. A custom `PairedDataset` class was implemented to filter and pair samples from both datasets that share the same digit label.

2.  **Model Architecture**: The model is a joint VAE with two separate CNN-based encoders and decoders. Both encoders map their inputs to the parameters of a 20-dimensional latent space.

3.  **Shared Latent Space (PoE)**: The projection into the shared latent space is achieved using a **Product of Experts (PoE)**. This method combines the mean $\mu$ and log-variance $$\log(\sigma^2)$$ from both encoders to compute the parameters of a joint posterior distribution.

4.  **Training**: The model is trained by optimizing the Evidence Lower Bound (ELBO), which consists of the sum of the Binary Cross-Entropy (BCE) loss (**MNIST**), Mean Squared Error (MSE) loss (**SVHN**), and the Kullback-Leibler (KL) divergence of the joint posterior from the standard normal prior.

---

## Visualizations and Results

The notebook includes some visualizations to evaluate the model's performance:

* **Joint Reconstruction**: The model successfully reconstructs both **MNIST** and **SVHN** images from the shared latent space.
* **Latent Space Generation**: By sampling random vectors ($z$) from the prior, the model generates new pairs of digits in both styles.
* **Cross-Domain Generation**: The model performs style transfer by encoding an image from one domain (e.g., **MNIST**) and decoding it into the other (e.g., **SVHN**), and vice-versa.
* **t-SNE Visualization**: A 2D t-SNE plot of the latent space shows that the 10-digit classes form distinct clusters, indicating a well-learned shared representation.

