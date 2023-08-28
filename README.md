# Variational Autoencoder for Behavioral Clustering of Agents

This project leverages the power of variational autoencoders (VAEs) to perform detailed behavioral clustering of individual agents. The project introduces two primary architectural frameworks: the Transformer structure and the LSTM structure. The flexibility of this project lies in the freedom to select either of these structures to construct the encoder and decoder for the VAE.

## Key Features

- Utilizes both Transformer and LSTM structures for VAE's encoder and decoder.
- Supports unsupervised training and self-supervision techniques for optimal learning.
- Offers self-supervision options through distinct Gaussian Mixture and k-means layers.
- Organized into three stages of training to progressively refine results.

## Project Overview

The primary objective of this project is to implement a Variational Autoencoder (VAE) capable of effectively clustering and understanding the behavioral patterns exhibited by individual agents. The project showcases two main architectural paradigms: the Transformer structure and the LSTM structure, either of which can be chosen for constructing the VAE's encoder and decoder components.

### Unsupervised and Self-Supervised Learning

The project offers the versatility of unsupervised training as well as advanced self-supervision techniques. In self-supervised mode, the architecture incorporates two specialized layers: the Gaussian Mixture Layer and the k-means Layer. This empowers the model to learn intricate agent behaviors by comparing predictions against k-means labels.

### Three Progressive Training Stages

1. **Stage One**: During this phase, the VAE structure is trained while the Gaussian mixture model remains fixed. This initial training phase sets the foundation for subsequent stages.
   
2. **Stage Two**: Building on the groundwork laid in Stage One, the VAE's layers are frozen, enabling the Gaussian mixture layer to become trainable. The model takes in latent dimensions and compares predictions with k-means labels, enhancing the understanding of agent behaviors.
   
3. **Stage Three**: The project reaches its apex as all layers are set to be trainable. This training stage fine-tunes the model's capabilities, resulting in a refined clustering of agent behaviors.


