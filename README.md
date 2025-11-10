# Arch-Ai-Tex

# house-floor-generator


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Key Features](#key-features)  
4. [Architecture & Methods](#architecture--methods)  
   - [Model Architecture](#model-architecture)  
   - [Loss Functions & Training Details](#training-details)  
   - [Dataset & Preprocessing](#dataset--preprocessing)  
   - [Resolution / Output Details](#resolution--output-details)
5. [Usage / Running the Project](#usage--running-the-project)  
   - [Training](#training)
   - [programing of electronic components](programing-of-electronic-components)
   - [Generating Floor Plans](#generating-floor-plans)  
   - [Viewing Results](#viewing-results)
6. [Features Explained](#Features-Explained)
   - [GAN Floorplan Generator](#GAN-Floorplan-Generator)
   - [Optimized Layout Generator](#Optimized-Layout-Generator)
   - [Real-Time Sensor Dashboard](#Real-Time-Sensor-Dashboard)
7.  [Results & Visualizations](#results--visualizations)
8.  [Installation Instructions](#installation-instructions)
9.  [Deployment](#Deployment)
---

## Project Overview  
This project implements a **deep generative model** to create realistic **house floor plans** automatically.  
It combines **GAN-based architecture**, and **Random Forest regression** for estimating room area distributions.  
The final output is a high-resolution, architecturally coherent layout image.

The system can be extended for **conditional generation** (e.g., based on number of rooms or length and breath), serving as an **AI design assistant** for architects and planners.

The goal is to help architects, designers, normal people or hobbyists quickly prototype layout ideas.

this system also suppoerts **Optimized Layout Generation** and **Real-Time Sensor Dashboard**
the **Optimized Layout Generation**: generates a rough layout from on dimenstions, number of rooms, Property Type, and Plot Shape.
the **Real-Time Sensor Dashboard**: capture house dimensions using ultrasonic and motion sensors and also generates a new layout by **GAN-based generator**


## Motivation  
- Creating floor plans manually is time-consuming and requires domain knowledge in architecture.  
- With generative models, one can **automate** the creation of many candidate layouts, speeding up the design exploration process.  
- By analyzing the **layout space**, designers can draw inspiration from machine-generated designs and refine them.  
- This project also serves an academic interest in understanding how deep networks handle spatial/layout generation, generalization of designs, and evaluation of architecture-associated outputs.

## Key Features  
- Generate floor plans given random/noise input (or conditional input, e.g., number of rooms).
- Visualize and compare generated designs versus dataset samples. - Web-app interface to input parameters (number of rooms, square footage, style) and output downloadable floor plan PNG.
- this also gives a option to add a denoiser for better images.
- Real-Time Sensor Dashboard - this system takes input from different sensors to automatically enter dimenstions and Motion & Obstacle Sensors to generated outputs.
- this also uses Optimized Layout Generation - this system gives room placment for the user by different input features.
- this system also shows a Segmented image of the generated image that help to locate the walls more easily 

## Architecture & Methods  

### Model Architecture  
- **Generator**:
  generator is one of two neural networks in a GAN system that competes against a discriminator network to create new, realistic data. The generator takes random noise as input and tries to produce synthetic data (like images or music) that is so convincing that the discriminator cannot tell it apart from real data in the original training set. Through this adversarial process, the generator continuously improves its ability to generate authentic-looking outputse.g., “Takes a 100-dimensional latent vector z, passes through fully-connected + reshape, followed by several transposed convolution / deconvolution layers, BatchNorm, ReLU activations, producing an output image of size 256×256.”  
- **Discriminator**:
  The discriminator acts as a binary classifier helps in distinguishing between real and generated data. It learns to improve its classification ability through training, refining its parameters to detect fake samples more accurately. When dealing with image data, the discriminator uses convolutional layers or other relevant architectures which help to extract features and enhance the model’s ability.e.g., “Receives the generated or real floor plan image, passes through several convolutional layers with LeakyReLU activations, followed by a final sigmoid output indicating real vs fake.”  
- **Circuit**:
  the Real-Time Sensor Dashboard uses different sensor, that help it to identidfy the distace and other parameters.
  we have used differnt electronic componenents for this model like arduino mega 2560, ESP32, HC-SR501 Passive Infrared sensor, and HC-SR04 Ultrasonic sensor, along with ESP32 for transimiting data via wifi.
  the data is processed theough arduino mega 2560 and then is transfered by wifi by ESP32.
  
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
- segmentation model for better understanding of structure
- auto room layout form Optimized Layout 

## Usage / Running the Project
### Training

To train the GAN or Random Forest models from scratch, run the respective training scripts [floor_generater](floor_generater.py) and [room_predictor](room_predictor.py) respectively. 
the [floor_generater](floor_generater.py) will save the generator.pth, discriminator.pth model, and sample images per 10th epoch and will also save the checkpoint on every epoch.
the [room_predictor](room_predictor.py) will save the room_predictor.joblib model after training.

### programing of electronic components

to program the electronic components for real time Sensor Dashboard we to program Arduino mega and ESP32 with [Arduino](mega/mega.ino) and [ESP32](esp32/esp32.ino) respectively which sends data in real time to a database with [server](server.py) with a delay of ~15 sec when connected to wifi. 

### Generating Floor Plans

To generate a new floor plan, run the script [main](main.py) or [main_webapp](app.py) or use the Deployed site at [[deployment](https://arch-ai-tex.streamlit.app/)] by streamlit.

Follow the prompts to enter:
  #### common prompts:
- Length (m)
- Width (m)
- Number of bedrooms
  #### for GANs generated images:
- Whether to apply denoiser (y or n)
  #### for Optimized Layout:
- Property Type
- Plot Shape

### Viewing Results
Generated floor plans are saved to the specified output path in the .png format.

### Features Explained
#### GAN Floorplan Generator:
The GAN Floorplan Generator is responsible for generating realistic floor plan layouts from either random noise or user-defined inputs such as number of rooms, dimensions, and style preferences.
It forms the creative core of the system, automatically producing architecturally consistent layouts that resemble real-world floor plans.

This component uses the trained [generator model](generator_epoch100.pth) to synthesize new layouts and can optionally apply a denoiser for enhanced clarity.
Each output is generated at 256×256 resolution and can be visualized or further segmented for wall detection and layout analysis.

##### Key Capabilities:
- Generates floor plans using the trained GAN model.
- Accepts conditional input (e.g., number of rooms, plot dimensions).
- Option to enable denoising for cleaner outputs.
- Produces results directly in the Streamlit app for real-time visualization.
- Works alongside the segmentation module to highlight walls and boundaries.

  Once trained, the generator can be called via the [web app](app.py) or [[deployment](https://arch-ai-tex.streamlit.app/)] under GAN Generator

#### Optimized Layout Generator:
The Optimized Layout Generator creates structured floor plan layouts based on user-specified design parameters such as plot dimensions, number of rooms, property type, and plot shape.
Unlike the GAN module, which focuses on visual realism, this system focuses on functional layout optimization — determining how rooms are arranged logically within the available space.

It leverages Random Forest Regression and heuristic placement algorithms to estimate the most practical arrangement of rooms while maintaining proportionality and connectivity.
This makes it ideal for scenarios where users need quick, constraint-based designs rather than creative random generations.

##### Key Capabilities:
- Generates a room-wise layout map using logical placement rules.
- Considers area efficiency, room adjacency, and aspect ratio constraints.
- Allows dynamic input for Property Type (e.g., apartment, villa, studio) and Plot Shape (rectangular, L-shaped, etc.).
- Can serve as a pre-processor for GAN input, allowing users to refine the generated image later.
- Works directly in the Streamlit app to visualize the optimized layout interactively.

  Once trained, the generator can be called via the [web app](app.py) or [[deployment](https://arch-ai-tex.streamlit.app/)] under Optimized Layout.

#### Real-Time Sensor Dashboard:
The Real-Time Sensor Dashboard bridges the gap between the physical environment and the digital floor plan generator.
It integrates multiple IoT sensors—such as HC-SR04 Ultrasonic Sensors for distance measurement and HC-SR501 PIR Sensors for motion detection—connected through Arduino Mega 2560 and ESP32 Wi-Fi modules.

This module continuously captures real-world dimensions and transmits the data to the system via Wi-Fi, enabling automatic entry of house dimensions without manual user input.
Once data is received, the dashboard instantly generates a new layout using the GAN-based floor plan generator, providing an intelligent, sensor-driven design experience.

##### Key Capabilities:
- Captures length and breadth measurements using ultrasonic sensors.
- Detects motion or occupancy via PIR sensors to enhance layout context.
- Uses Arduino Mega 2560 for signal processing and ESP32 for wireless data transmission.
- Data is updated on the web dashboard in real-time (~15s refresh rate).
- Offers a Reset and Generate control flow for user correction and immediate regeneration of layouts.
- Seamlessly integrates with the main web app to display generated plans directly from live sensor inputs.

  
## Results & Visualizations
Here are examples of floor plans generated by the GAN model:

![Sample Floor Plan 1](samples/floorplan_2025-08-29_20-53-19.png)
![Sample Floor Plan 2](samples/floorplan_1_2025-10-17_21-20-51.png)
![Sample Floor Plan 3](samples/floorplan_3_2025-08-29_21-33-10.png)


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
         app.py
   
## Deployment
deployment is done online on [[deployment](https://arch-ai-tex.streamlit.app/)]
