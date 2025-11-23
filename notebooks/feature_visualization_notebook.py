#!/usr/bin/env python3
"""
Convert to Jupyter notebook for Feature Visualization
"""

notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Visualization - Interactive Notebook\\n",
    "\\n",
    "This notebook demonstrates the techniques from the Distill paper:\\n",
    "\\\"Feature Visualization\" by Olah, Mordvintsev, and Schubert (2017)\\n",
    "\\n",
    "https://distill.pub/2017/feature-visualization/\\n",
    "\\n",
    "## Key Techniques Demonstrated:\\n",
    "\\n",
    "1. **Basic Channel/Neuron Visualization** - Creating images that maximally activate specific features\\n",
    "2. **Diversity Visualization** - Multiple diverse visualizations of the same feature\\n",
    "3. **Interpolation** - Smooth transitions between different features\\n",
    "4. **Evolution Tracking** - Watching visualization develop during optimization\\n",
    "5. **Advanced Regularization** - Frequency and total variation regularization\\n",
    "6. **Method Comparison** - Comparing different optimization approaches"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages (uncomment if needed)\\n",
    "# !pip install torch torchvision numpy matplotlib pillow tqdm scikit-learn\\n",
    "\\n",
    "# Import libraries\\n",
    "import torch\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "from pytorch_lucid import modelzoo, optvis, misc\\n",
    "from pytorch_lucid.distill import FeatureVisualization\\n",
    "from pytorch_lucid.misc.io import show_image, save_image, create_image_grid\\n",
    "import os\\n",
    "\\n",
    "# Setup\\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\\n",
    "print(f\"Using device: {device}\")\\n",
    "\\n",
    "# Create output directory\\n",
    "output_dir = 'fv_notebook_outputs'\\n",
    "os.makedirs(output_dir, exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load pre-trained model\\n",
    "print(\"Loading VGG16 model...\")\\n",
    "model_wrapper = modelzoo.load_model('vgg16', device=device)\\n",
    "print(f\"Model loaded: {model_wrapper.model.__class__.__name__}\")\\n",
    "\\n",
    "# Get convolutional layers\\n",
    "conv_layers = model_wrapper.get_conv_layers()\\n",
    "print(f\"Found {len(conv_layers)} convolutional layers\")\\n",
    "\\n",
    "# Use middle layer for demonstrations\\n",
    "target_layer = conv_layers[len(conv_layers) // 2]\\n",
    "print(f\"Target layer: {target_layer}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Basic Channel Visualization\\n",
    "\\n",
    "The foundation of feature visualization - creating images that maximally activate specific channels in neural network layers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize feature visualization\\n",
    "fv = FeatureVisualization(model_wrapper.model, model_wrapper.device)\\n",
    "\\n",
    "# Basic channel visualization\\n",
    "print(\"Creating basic channel visualization...\")\\n",
    "basic_viz = fv.visualize_channel(\\n",
    "    target_layer, 0,\\n",
    "    num_steps=512,\\n",
    "    image_size=(224, 224),\\n",
    "    use_fft=True\\n",
    ")\\n",
    "\\n",
    "if basic_viz is not None:\\n",
    "    plt.figure(figsize=(8, 8))\\n",
    "    plt.imshow(basic_viz)\\n",
    "    plt.title(f'Channel 0 Visualization - {target_layer}')\\n",
    "    plt.axis('off')\\n",
    "    plt.show()\\n",
    "    \\n",
    "    # Save result\\n",
    "    save_image(torch.from_numpy(basic_viz).permute(2, 0, 1).unsqueeze(0), \\
",
    "               f'{output_dir}/basic_channel_viz.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Neuron Evolution During Optimization\\n",
    "\\n",
    "Track how a visualization develops over the course of optimization, showing the progressive refinement of the image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Track evolution during optimization\\n",
    "print(\"Tracking neuron evolution...\")\\n",
    "evolution_steps = [64, 128, 256, 512, 1024]\\n",
    "\\n",
    "evolution = fv.visualize_neuron_evolution(\\n",
    "    target_layer, 0,\\n",
    "    evolution_steps=evolution_steps,\\n",
    "    image_size=(128, 128)\\n",
    ")\\n",
    "\\n",
    "if evolution:\\n",
    "    fig, axes = plt.subplots(1, len(evolution), figsize=(20, 4))\\n",
    "    for i, (step, img) in enumerate(evolution.items()):\\n",
    "        axes[i].imshow(img)\\n",
    "        axes[i].set_title(f'Step {step}')\\n",
    "        axes[i].axis('off')\\n",
    "    \\n",
    "    plt.suptitle('Neuron Evolution During Optimization')\\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(f\"Evolution complete! Saved {len(evolution)} stages\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Diversity Visualization\\n",
    "\\n",
    "Create multiple diverse visualizations of the same feature by using different random seeds and optimization paths."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create diverse visualizations\\n",
    "print(\"Creating diverse visualizations...\")\\n",
    "diverse_images = fv.create_diversity_visualization(\\n",
    "    target_layer, 5,\\n",
    "    num_diverse=6,\\n",
    "    num_steps=256,\\n",
    "    image_size=(128, 128)\\n",
    ")\\n",
    "\\n",
    "if diverse_images:\\n",
    "    grid = create_image_grid(diverse_images, (2, 3))\\n",
    "    grid_array = grid.detach().cpu().numpy().transpose(1, 2, 0)\\n",
    "    \\n",
    "    plt.figure(figsize=(12, 8))\\n",
    "    plt.imshow(grid_array)\\n",
    "    plt.title('Diverse Visualizations of Same Feature')\\n",
    "    plt.axis('off')\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(f\"Created {len(diverse_images)} diverse visualizations\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Interpolation Between Features\\n",
    "\\n",
    "Create smooth transitions between different features to understand the feature space geometry."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create interpolation between features\\n",
    "layer_info = model_wrapper.get_layer_info(target_layer)\\n",
    "if 'out_channels' in layer_info and layer_info['out_channels'] > 1:\\n",
    "    second_channel = min(5, layer_info['out_channels'] - 1)\\n",
    "    \\n",
    "    print(f\"Creating interpolation: Channel 0 → Channel {second_channel}\")\\n",
    "    \\n",
    "    interpolations = fv.create_interpolation_visualization(\\n",
    "        target_layer, 0, second_channel,\\n",
    "        num_interps=5,\\n",
    "        num_steps=256,\\n",
    "        image_size=(128, 128)\\n",
    "    )\\n",
    "    \\n",
    "    if interpolations:\\n",
    "        fig, axes = plt.subplots(1, len(interpolations), figsize=(20, 4))\\n",
    "        for i, img in enumerate(interpolations):\\n",
    "            axes[i].imshow(img)\\n",
    "            axes[i].set_title(f'Interpolation {i}/{len(interpolations)-1}')\\n",
    "            axes[i].axis('off')\\n",
    "        \\n",
    "        plt.suptitle(f'Interpolation: Channel 0 → Channel {second_channel}')\\n",
    "        plt.tight_layout()\\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Advanced Regularization\\n",
    "\\n",
    "Use frequency regularization to create smoother, more natural-looking visualizations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Frequency regularization\\n",
    "print(\"Creating frequency-regularized visualization...\")\\n",
    "freq_viz = fv.create_frequency_regularized_visualization(\\n",
    "    target_layer, 0,\\n",
    "    frequency_penalty=1.0,\\n",
    "    num_steps=512,\\n",
    "    image_size=(224, 224)\\n",
    ")\\n",
    "\\n",
    "if freq_viz is not None:\\n",
    "    plt.figure(figsize=(8, 8))\\n",
    "    plt.imshow(freq_viz)\\n",
    "    plt.title('Frequency-Regularized Visualization')\\n",
    "    plt.axis('off')\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Method Comparison\\n",
    "\\n",
    "Compare different visualization approaches to understand their strengths and weaknesses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare different methods\\n",
    "print(\"Comparing visualization methods...\")\\n",
    "comparison = fv.create_visualization_comparison(\\n",
    "    target_layer, 0,\\n",
    "    methods=['basic', 'fft', 'regularized', 'preconditioned'],\\n",
    "    num_steps=256,\\n",
    "    image_size=(128, 128)\\n",
    ")\\n",
    "\\n",
    "if len(comparison) >= 2:\\n",
    "    fig, axes = plt.subplots(1, len(comparison), figsize=(5*len(comparison), 5))\\n",
    "    if len(comparison) == 1:\\n",
    "        axes = [axes]\\n",
    "    \\n",
    "    for i, (method, img) in enumerate(comparison.items()):\\n",
    "        axes[i].imshow(img)\\n",
    "        axes[i].set_title(f'{method.title()} Method')\\n",
    "        axes[i].axis('off')\\n",
    "    \\n",
    "    plt.suptitle('Visualization Method Comparison')\\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(f\"Compared {len(comparison)} different methods\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Feature Grid Visualization\\n",
    "\\n",
    "Create a grid of visualizations to understand the variety of features learned by a layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create feature grid\\n",
    "print(\"Creating feature visualization grid...\")\\n",
    "layer_info = model_wrapper.get_layer_info(target_layer)\\n",
    "if 'out_channels' in layer_info:\\n",
    "    num_channels = min(9, layer_info['out_channels'])\\n",
    "    channel_indices = list(range(num_channels))\\n",
    "else:\\n",
    "    channel_indices = list(range(9))\\n",
    "\\n",
    "feature_grid = fv.create_feature_visualization_grid(\\n",
    "    target_layer,\\n",
    "    channel_indices,\\n",
    "    grid_size=(3, 3),\\n",
    "    num_steps=256,\\n",
    "    image_size=(128, 128)\\n",
    ")\\n",
    "\\n",
    "plt.figure(figsize=(12, 12))\\n",
    "plt.imshow(feature_grid)\\n",
    "plt.title(f'Feature Visualization Grid - {target_layer}')\\n",
    "plt.axis('off')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary and Next Steps\\n",
    "\\n",
    "This notebook demonstrated the core techniques from the Feature Visualization paper:\\n",
    "\\n",
    "✅ **Basic Visualizations**: Channel and neuron visualizations with FFT parameterization\\n",
    "✅ **Evolution Tracking**: Monitoring optimization progress over time\\n",
    "✅ **Diversity Methods**: Creating multiple interpretations of the same feature\\n",
    "✅ **Interpolation**: Smooth transitions between features\\n",
    "✅ **Advanced Regularization**: Frequency-based regularization for natural results\\n",
    "✅ **Method Comparison**: Comparing different optimization approaches\\n",
    "✅ **Batch Processing**: Grid visualizations for comprehensive analysis\\n",
    "\\n",
    "### Next Steps:\\n",
    "\\n",
    "1. **Try Different Models**: Test with ResNet, AlexNet, or custom models\\n",
    "2. **Different Layers**: Explore early vs. late layer visualizations\\n",
    "3. **Custom Objectives**: Create your own optimization objectives\\n",
    "4. **Building Blocks**: Move on to the Building Blocks of Interpretability notebook\\n",
    "5. **Circuits Analysis**: Explore neural network circuits in the Circuits notebook"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

with open('/mnt/okcomputer/output/notebooks/feature_visualization.ipynb', 'w') as f:
    f.write(notebook_content)

print("Created feature_visualization.ipynb")