# Arch-Ai-Tex

# house-floor-generator


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Key Features](#key-features)  
4. [Architecture & Methods](#architecture-methods)  
   - [Model Architecture](#model-architecture)  
   - [Loss Functions & Training Details](#loss-functions-training-details)  
   - [Dataset & Preprocessing](#dataset-preprocessing)  
   - [Resolution / Output Details](#resolution-output-details)  
5. [Project Structure](#project-structure)  
6. [Installation Instructions](#installation-instructions)  
7. [Usage / Running the Project](#usage-running-the-project)  
   - [Training](#training)  
   - [Generating Floor Plans](#generating-floor-plans)  
   - [Viewing Results](#viewing-results)  
8. [Results & Visualizations](#results-visualizations)  
9. [Evaluation & Metrics](#evaluation-metrics)  
10. [Limitations & Future Work](#limitations-future-work)  
11. [Contributing](#contributing)  
12. [License](#license)  
13. [Acknowledgements](#acknowledgements)  
14. [Contact / Author](#contact-author)  

---

## Project Overview  
This project implements a system to generate house floor plans using GANs. It takes as input architectural layout data and produces new, **synthetic floor plans** that realistic spatial ordering. The goal is to help architects, designers, normal people or hobbyists quickly prototype layout ideas.

## Motivation  
- Creating floor plans manually is time-consuming and requires domain knowledge in architecture.  
- With generative models, one can **automate** the creation of many candidate layouts, speeding up the design exploration process.  
- By analyzing the **layout space**, designers can draw inspiration from machine-generated designs and refine them.  
- This project also serves an academic interest in understanding how deep networks handle spatial/layout generation, generalization of designs, and evaluation of architecture-associated outputs.

## Key Features  
- Generate floor plans given random/noise input (or conditional input, e.g., number of rooms).    
- Visualize and compare generated designs versus dataset samples.  
- Web-app interface to input parameters (number of rooms, square footage, style) and output downloadable floor plan PNG.  

## Architecture & Methods  

### Model Architecture  
- **Generator**:
  generator is one of two neural networks in a GAN system that competes against a discriminator network to create new, realistic data. The generator takes random noise as input and tries to produce synthetic data (like images or music) that is so convincing that the discriminator cannot tell it apart from real data in the original training set. Through this adversarial process, the generator continuously improves its ability to generate authentic-looking outputse.g., “Takes a 100-dimensional latent vector z, passes through fully-connected + reshape, followed by several transposed convolution / deconvolution layers, BatchNorm, ReLU activations, producing an output image of size 256×256.”  
- **Discriminator**:
  The discriminator acts as a binary classifier helps in distinguishing between real and generated data. It learns to improve its classification ability through training, refining its parameters to detect fake samples more accurately. When dealing with image data, the discriminator uses convolutional layers or other relevant architectures which help to extract features and enhance the model’s ability.e.g., “Receives the generated or real floor plan image, passes through several convolutional layers with LeakyReLU activations, followed by a final sigmoid output indicating real vs fake.”  

### Loss Functions & Training Details  
- For GAN:  
  - Discriminator loss:
- 

       $$ L_D = - \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] 
      - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

  - Generator loss:
- 

       $$ L_G = - \mathbb{E}_{z \sim p_z}[\log D(G(z))] $$

- Optimizer: e.g., Adam with \( \beta_1=0.5, \beta_2=0.999 \) and initial learning rate 2e-4.  
- Batch size: e.g., 64.  
- Number of epochs: e.g., 200.  
- Learning rate schedule: e.g., halved every 50 epochs.  
- Any regularization: e.g., spectral normalization in discriminator, gradient penalty, instance normalization.  
- Data augmentation: flips, rotations, scaling to increase diversity.

### Dataset & Preprocessing  
- Dataset: e.g., “We used the [NAME] dataset of house floor plan images (or: collected from architectural websites) comprising X samples of plans with number of rooms ranging from min to max.”  
- Preprocessing steps:  
  - Convert to grayscale or RGB.  
  - Resize all images to e.g., 128×128 (or varying sizes).  
  - Normalize pixel values to \([-1,1]\).  
  - (Optional) Remove extraneous annotations, clean up layout, binarize walls vs empty space.  
  - Split dataset into training and validation sets (e.g., 80% training, 20% validation).

### Resolution / Output Details  
- The model supports multiple output resolutions: 64×64 (fast, low detail), 128×128 (moderate), 256×256 (higher detail).  
- Explain how resolution affects training time, memory usage, model capacity.  
- Discuss trade-offs: higher resolution yields more realistic floor plans but takes more GPU memory and longer training.
