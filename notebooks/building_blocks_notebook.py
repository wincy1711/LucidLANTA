#!/usr/bin/env python3
"""
Convert to Jupyter notebook for Building Blocks of Interpretability
"""

notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Building Blocks of Interpretability - Interactive Notebook\\n",
    "\\n",
    "This notebook demonstrates the techniques from the Distill paper:\\n",
    "\\\"The Building Blocks of Interpretability\" by Olah et al. (2018)\\n",
    "\\n",
    "https://distill.pub/2018/building-blocks/\\n",
    "\\n",
    "## Key Techniques Demonstrated:\\n",
    "\\n",
    "1. **Semantic Dictionaries** - Visualizing the most important features in a layer\\n",
    "2. **Combined Interfaces** - Integrating multiple interpretability techniques\\n",
    "3. **Attribution Visualization** - Understanding how features contribute to outputs\\n",
    "4. **Activation Atlases** - Clustering and visualizing activation patterns\\n",
    "5. **Systematic Interface Design** - Exploring the space of interpretability interfaces"
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
    "from pytorch_lucid.distill import BuildingBlocks\\n",
    "from pytorch_lucid.misc.io import show_image, save_image, create_image_grid\\n",
    "import os\\n",
    "\\n",
    "# Setup\\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\\n",
    "print(f\"Using device: {device}\")\\n",
    "\\n",
    "# Create output directory\\n",
    "output_dir = 'bb_notebook_outputs'\\n",
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
    "## 1. Semantic Dictionary\\n",
    "\\n",
    "Create a semantic dictionary by visualizing the most important neurons/channels in a layer.\\n",
    "This helps understand what features the layer has learned to detect."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize building blocks\\n",
    "bb = BuildingBlocks(model_wrapper.model, model_wrapper.device)\\n",
    "\\n",
    "# Create semantic dictionary\\n",
    "print(\"Creating semantic dictionary...\")\\n",
    "semantic_dict = bb.create_semantic_dictionary(\\n",
    "    target_layer,\\n",
    "    top_k=16,\\n",
    "    num_steps=256,\\n",
    "    image_size=(128, 128)\\n",
    ")\\n",
    "\\n",
    "if semantic_dict['semantic_grid'] is not None:\\n",
    "    plt.figure(figsize=(12, 12))\\n",
    "    plt.imshow(semantic_dict['semantic_grid'])\\n",
    "    plt.title(f'Semantic Dictionary - {target_layer}')\\n",
    "    plt.axis('off')\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(f\"Created semantic dictionary with {semantic_dict['num_features']} features\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Attribution Visualization\\n",
    "\\n",
    "Combine feature visualization with attribution to understand how features contribute to the final output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create attribution visualization\\n",
    "print(\"Creating attribution visualization...\")\\n",
    "attribution_viz = bb.create_attribution_visualization(\\n",
    "    target_layer, 0,\\n",
    "    attribution_layer='output',\\n",
    "    num_steps=256,\\n",
    "    image_size=(224, 224)\\n",
    ")\\n",
    "\\n",
    "if attribution_viz:\\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 8))\\n",
    "    \\n",
    "    # Feature visualization\\\n",
    "    axes[0].imshow(attribution_viz['feature_visualization'])\\n",
    "    axes[0].set_title('Feature Visualization')\\n",
    "    axes[0].axis('off')\\n",
    "    \\n",
    "    # Attribution map\\\n",
    "    if attribution_viz['attribution_map'] is not None:\\n",
    "        axes[1].imshow(attribution_viz['attribution_map'], cmap='hot')\\n",
    "        axes[1].set_title('Attribution Map')\\n",
    "        axes[1].axis('off')\\n",
    "    \\n",
    "    plt.suptitle('Feature Visualization with Attribution')\\n",
    "    plt.tight_layout()\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Activation Atlas (Simplified)\\n",
    "\\n",
    "Create a simplified version of activation atlas by clustering activation patterns and visualizing cluster centers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create activation atlas\\n",
    "print(\"Creating activation atlas...\")\\n",
    "atlas = bb.create_activation_atlas_concept(\\n",
    "    target_layer,\\n",
    "    num_samples=50,  # Reduced for demo\\n",
    "    num_clusters=8\\n",
    ")\\n",
    "\\n",
    "if atlas['cluster_visualizations']:\\n",
    "    # Create atlas grid\\\n",
    "    images = list(atlas['cluster_visualizations'].values())\\n",
    "    grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2\\n",
    "    \\n",
    "    if images:\\n",
    "        atlas_grid = create_image_grid(images, grid_size)\\n",
    "        atlas_array = atlas_grid.detach().cpu().numpy().transpose(1, 2, 0)\\n",
    "        \\n",
    "        plt.figure(figsize=(10, 10))\\n",
    "        plt.imshow(atlas_array)\\n",
    "        plt.title('Activation Atlas - Cluster Centers')\\n",
    "        plt.axis('off')\\n",
    "        plt.show()\\n",
    "        \\n",
    "        print(f\"Created atlas with {len(images)} cluster centers\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Combined Interpretability Interface\\n",
    "\\n",
    "Create the comprehensive interface combining multiple interpretability techniques as described in the paper."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create test image\\n",
    "test_image = np.random.rand(224, 224, 3).astype(np.float32)\\n",
    "\\n",
    "# Create smooth gradients\\n",
    "x = np.linspace(0, 1, 224)\\n",
    "y = np.linspace(0, 1, 224)\\n",
    "X, Y = np.meshgrid(x, y)\\n",
    "\\n",
    "test_image[:, :, 0] = np.sin(X * 4 * np.pi) * np.cos(Y * 4 * np.pi)\\n",
    "test_image[:, :, 1] = np.sin(X * 2 * np.pi) * np.cos(Y * 6 * np.pi)\\n",
    "test_image[:, :, 2] = np.sin(X * 6 * np.pi) * np.cos(Y * 2 * np.pi)\\n",
    "\\n",
    "# Normalize\\n",
    "test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())\\n",
    "\\n",
    "# Create combined interface\\n",
    "print(\"Creating combined interpretability interface...\")\\n",
    "interface = bb.create_combined_interface(\\n",
    "    test_image,\\n",
    "    target_layer,\\n",
    "    top_k=6\\n",
    ")\\n",
    "\\n",
    "# Display interface components\\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\\n",
    "\\n",
    "# Input image\\\n",
    "axes[0, 0].imshow(interface['input_image'])\\n",
    "axes[0, 0].set_title('Input Image')\\n",
    "axes[0, 0].axis('off')\\n",
    "\\n",
    "# Attribution map\\\n",
    "if interface['attribution_map'] is not None:\\n",
    "    axes[0, 1].imshow(interface['attribution_map'], cmap='hot')\\n",
    "    axes[0, 1].set_title('Attribution Map')\\n",
    "    axes[0, 1].axis('off')\\n",
    "\\n",
    "# Feature visualizations grid\\\n",
    "if interface['feature_visualizations']:\\n",
    "    feature_images = list(interface['feature_visualizations'].values())[:4]\\n",
    "    if feature_images:\\n",
    "        feature_grid = create_image_grid(feature_images, (2, 2))\\n",
    "        feature_array = feature_grid.detach().cpu().numpy().transpose(1, 2, 0)\\n",
    "        axes[0, 2].imshow(feature_array)\\n",
    "        axes[0, 2].set_title('Top Feature Visualizations')\\n",
    "        axes[0, 2].axis('off')\\n",
    "\\n",
    "# Spatial activations\\\n",
    "if interface['spatial_activations']:\\n",
    "    spatial_keys = list(interface['spatial_activations'].keys())[:2]\\n",
    "    if len(spatial_keys) >= 2:\\n",
    "        axes[1, 0].imshow(interface['spatial_activations'][spatial_keys[0]], cmap='viridis')\\n",
    "        axes[1, 0].set_title(f'Spatial Activation - {spatial_keys[0]}')\\n",
    "        axes[1, 0].axis('off')\\n",
    "        \\n",
    "        axes[1, 1].imshow(interface['spatial_activations'][spatial_keys[1]], cmap='viridis')\\n",
    "        axes[1, 1].set_title(f'Spatial Activation - {spatial_keys[1]}')\\n",
    "        axes[1, 1].axis('off')\\n",
    "\\n",
    "# Info text\\\n",
    "info_text = f\"Predicted Class: {interface['predicted_class']}\\n\"\\n",
    "info_text += f\"Confidence: {interface['confidence']:.3f}\\n\"\\n",
    "info_text += f\"Top Channels: {interface['top_channels'][:3]}\"\\n",
    "\\n",
    "axes[1, 2].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,\\n",
    "               bbox=dict(boxstyle=\"round,pad=0.3\", facecolor=\"lightgray\"))\\n",
    "axes[1, 2].set_title('Analysis Info')\\n",
    "axes[1, 2].axis('off')\\n",
    "\\n",
    "plt.suptitle('Combined Interpretability Interface')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Interface Design Space Exploration\\n",
    "\\n",
    "Explore different ways to combine interpretability building blocks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Explore different interface combinations\\n",
    "print(\"Exploring interface design space...\")\\n",
    "\\n",
    "# Different ways to present the same information\\n",
    "interface_variants = {}\\n",
    "\\n",
    "# Variant 1: Feature-focused interface\\n",
    "print(\"Creating feature-focused interface...\")\\n",
    "interface_variants['feature_focused'] = bb.create_combined_interface(\\n",
    "    test_image, target_layer, top_k=12\\n",
    ")\\n",
    "\\n",
    "# Variant 2: Attribution-focused interface\\n",
    "print(\"Creating attribution-focused interface...\")\\n",
    "interface_variants['attribution_focused'] = bb.create_attribution_visualization(\\n",
    "    target_layer, 0, attribution_layer='output', top_k=8\\n",
    ")\\n",
    "\\n",
    "print(f\"Created {len(interface_variants)} interface variants\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary and Next Steps\\n",
    "\\n",
    "This notebook demonstrated the core techniques from the Building Blocks of Interpretability paper:\\n",
    "\\n",
    "✅ **Semantic Dictionaries** - Visualizing layer features in a structured way\\n",
    "✅ **Attribution Integration** - Combining feature visualization with attribution analysis\\n",
    "✅ **Activation Atlases** - Clustering and visualizing activation patterns\\n",
    "✅ **Combined Interfaces** - Integrating multiple interpretability techniques\\n",
    "✅ **Interface Design** - Systematically exploring interpretability interface options\\n",
    "\\n",
    "### Next Steps:\\n",
    "\\n",
    "1. **Try Different Models**: Test with different architectures and layer depths\\n",
    "2. **Real Images**: Use real images instead of synthetic test patterns\\n",
    "3. **Custom Objectives**: Create specific interface designs for your use case\\n",
    "4. **Circuits Analysis**: Move on to the Thread: Circuits notebook\\n",
    "5. **Feature Visualization**: Go back to explore basic visualization techniques"
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

with open('/mnt/okcomputer/output/notebooks/building_blocks.ipynb', 'w') as f:
    f.write(notebook_content)

print("Created building_blocks.ipynb")