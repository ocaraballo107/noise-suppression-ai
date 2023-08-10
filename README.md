# Noise Suppressor AI using Convolutional Neural Networks

This project demonstrates the implementation of a Noise Suppressor using Convolutional Neural Networks (CNNs) with fully connected layers, ReLU activation, batch normalization, and a regression layer. The aim of this project is to clean noisy speech signals by training a CNN model.

## Project Overview

In this project, we recreate a Noise Suppressor using modern machine learning techniques, specifically Convolutional Neural Networks (CNNs). The goal is to suppress noise from noisy speech signals and enhance their quality.

## Getting Started

These instructions will guide you through setting up and running the Noise Suppressor project on your local machine.

### Prerequisites

- Python (>=3.6)
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ocaraballo107/noise-suppressor-ai.git

### Navigate to the project directory

cd noise-suppressor-ai

### Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate

### Install project dependencies

pip install -r requirements.txt

### Usage

Prepare your training and testing data (noisy speech and clean speech).
Replace X_data and y_data in the code with your actual preprocessed data.
Run the main script to train the model and evaluate its performance: python noise_suppressor.py
The script will print training progress, validation loss, test loss, and minimum loss achieved.

### Results

The model's performance is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The minimum loss achieved during training is also tracked and reported.

### Background

The traditional methods of noise suppression for speech signals involved complex signal processing techniques. In this project, we take a machine learning approach and utilize Convolutional Neural Networks to learn the mapping between noisy speech and clean speech.

#### Future Work

This project can be extended by:

Trying different architectures for the CNN model.
Exploring different optimization algorithms and learning rates.
Incorporating more advanced features such as spectrogram processing.

### Author

Oscar Caraballo

### License

This project is licensed under the MIT License
