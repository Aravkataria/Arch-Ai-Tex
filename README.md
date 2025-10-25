# Arch-Ai-Tex

# house-floor-generator


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Key Features](#key-features)  
4. [Architecture & Methods](#architecture--methods)  
   - [Model Architecture](#model-architecture)  
   - [Loss Functions & Training Details](#loss-functions--training-details)  
   - [Dataset & Preprocessing](#dataset--preprocessing)  
   - [Resolution / Output Details](#resolution--output-details)
5. [Usage / Running the Project](#usage--running-the-project)  
   - [Training](#training)  
   - [Generating Floor Plans](#generating-floor-plans)  
   - [Viewing Results](#viewing-results)  
6.  [Results & Visualizations](#results--visualizations)
7.  [Installation Instructions](#installation-instructions)
---

## Project Overview  
This project implements a **deep generative model** to create realistic **house floor plans** automatically.  
It combines **GAN-based architecture**, and **Random Forest regression** for estimating room area distributions.  
The final output is a high-resolution, architecturally coherent layout image.

The system can be extended for **conditional generation** (e.g., based on number of rooms or total area), serving as an **AI design assistant** for architects and planners.

The goal is to help architects, designers, normal people or hobbyists quickly prototype layout ideas.

## Motivation  
- Creating floor plans manually is time-consuming and requires domain knowledge in architecture.  
- With generative models, one can **automate** the creation of many candidate layouts, speeding up the design exploration process.  
- By analyzing the **layout space**, designers can draw inspiration from machine-generated designs and refine them.  
- This project also serves an academic interest in understanding how deep networks handle spatial/layout generation, generalization of designs, and evaluation of architecture-associated outputs.

## Key Features  
- Generate floor plans given random/noise input (or conditional input, e.g., number of rooms).
- Visualize and compare generated designs versus dataset samples. - Web-app interface to input parameters (number of rooms, square footage, style) and output downloadable floor plan PNG.

## Architecture & Methods  

### Model Architecture  
- **Generator**:
  generator is one of two neural networks in a GAN system that competes against a discriminator network to create new, realistic data. The generator takes random noise as input and tries to produce synthetic data (like images or music) that is so convincing that the discriminator cannot tell it apart from real data in the original training set. Through this adversarial process, the generator continuously improves its ability to generate authentic-looking outputse.g., “Takes a 100-dimensional latent vector z, passes through fully-connected + reshape, followed by several transposed convolution / deconvolution layers, BatchNorm, ReLU activations, producing an output image of size 256×256.”  
- **Discriminator**:
  The discriminator acts as a binary classifier helps in distinguishing between real and generated data. It learns to improve its classification ability through training, refining its parameters to detect fake samples more accurately. When dealing with image data, the discriminator uses convolutional layers or other relevant architectures which help to extract features and enhance the model’s ability.e.g., “Receives the generated or real floor plan image, passes through several convolutional layers with LeakyReLU activations, followed by a final sigmoid output indicating real vs fake.”  

### Training Details  

- Optimizer: Adam (lr=0.0002, β₁=0.5, β₂=0.999)
- Batch size: 8  
- Number of epochs: 100

### Dataset & Preprocessing  
- Preprocessing steps:  
  - greayscale images are generated.  
  - Resize all images to 256×256.  
  - Normalize pixel values to [-1,1].
  #### for room prediction
  - Split dataset into training and validation sets (e.g., 80% training, 20% validation).

### Resolution / Output Details  
- The model supports high output resolutions: 256×256.  
- The images is saved as .png.
- Optional denoising applied per user input.

## Usage / Running the Project
### Training

To train the GAN or Random Forest models from scratch, run the respective training scripts [floor_generater](floor_generater.py) and [room_predictor](room_predictor.py) respectively. 
the [floor_generater](floor_generater.py) will save the generator.pth, discriminator.pth model, and sample images per 10th epoch and will also save the checkpoint on every epoch.
the [room_predictor](room_predictor.py) will save the room_predictor.joblib model after training.

### Generating Floor Plans

To generate a new floor plan, run the script [main](main.py) or [main_webapp](main_webapp.py)

Follow the prompts to enter:
- Total area (sq.ft.)
- Number of bedrooms
- Whether to apply denoiser (y or n)

### Viewing Results
Generated floor plans are saved to the specified output path in the .png format.

## Results & Visualizations
Here are examples of floor plans generated by the model:

![Sample Floor Plan 1](sample/floorplan_2025-08-29_20-53-19.png)
![Sample Floor Plan 2](sample/floorplan_1_2025-10-17_21-20-51.png)
![Sample Floor Plan 3](sample/floorplan_3_2025-08-29_21-33-10.png)

## Installation Instructions
Follow these steps to set up the project locally:

1. Clone the Repository
   
          git clone https://github.com/AravKataria/Arch-Ai-Tex.git
          cd Arch-Ai-Tex
2. Set up the Environment
   
          python -m venv venv
          venv\Scripts\activate
3. Install requirements

            pip install -r requirements.txt
4. Run the Script
   
         main.py
         main_webapp.py
   
