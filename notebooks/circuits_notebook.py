#!/usr/bin/env python3
"""
Convert to Jupyter notebook for Thread: Circuits
"""

notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Thread: Circuits - Interactive Notebook\\n",
    "\\n",
    "This notebook demonstrates the techniques from the Distill paper:\\n",
    "\\\"Thread: Circuits\" by Cammarata et al. (2020)\\n",
    "\\n",
    "https://distill.pub/2020/circuits/\\n",
    "\\n",
    "## Key Techniques Demonstrated:\\n",
    "\\n",
    "1. **Curve Detector Analysis** - Identifying and analyzing curve-detecting neurons\\n",
    "2. **Weight Visualization** - Visualizing learned weight patterns\\n",
    "3. **Weight Banding Analysis** - Detecting organizational patterns in weights\\n",
    "4. **Branch Specialization** - Finding specialized neuron groupings\\n",
    "5. **High-Low Frequency Detectors** - Analyzing frequency transition detectors\\n",
    "6. **Circuit Reverse Engineering** - Understanding learned algorithms"
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
    "from sklearn.decomposition import PCA\\n",
    "from sklearn.cluster import KMeans\\n",
    "from pytorch_lucid import modelzoo, optvis, misc\\n",
    "from pytorch_lucid.distill import Circuits\\n",
    "from pytorch_lucid.misc.io import show_image, save_image, create_image_grid\\n",
    "import os\\n",
    "\\n",
    "# Setup\\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\\n",
    "print(f\"Using device: {device}\")\\n",
    "\\n",
    "# Create output directory\\n",
    "output_dir = 'circuits_notebook_outputs'\\n",
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
    "# Use early layer for circuit analysis\\n",
    "early_layers = [l for l in conv_layers if any(x in l for x in ['conv1', 'features.0', 'features.1', 'features.2', 'features.3'])]\\n",
    "target_layer = early_layers[0] if early_layers else conv_layers[0]\\n",
    "print(f\"Target layer: {target_layer}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Curve Detector Analysis\\n",
    "\\n",
    "Identify and analyze neurons that respond to curves, examining their preferred orientations and response patterns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize circuits analysis\\n",
    "circuits = Circuits(model_wrapper.model, model_wrapper.device)\\n",
    "\\n",
    "# Analyze curve detectors\\n",
    "print(\"Analyzing curve detectors...\")\\n",
    "curve_analysis = circuits.analyze_curve_detectors(\\n",
    "    target_layer,\\n",
    "    num_samples=50,  # Reduced for demo\\n",
    "    angle_range=(0, 180)\\n",
    ")\\n",
    "\\n",
    "print(f\"Found {curve_analysis['num_curve_detectors']} curve detectors\")\\n",
    "\\n",
    "if curve_analysis['curve_neurons']:\\n",
    "    print(\"\\nTop Curve Detectors:\")\\n",
    "    for i, curve_info in enumerate(curve_analysis['curve_neurons'][:5]):\\n",
    "        print(f\"  Neuron {curve_info['neuron_idx']}: Prefers {curve_info['preferred_angle']}°, \",\\n",
    "              f\"Max Response: {curve_info['max_response']:.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize curve detectors\\n",
    "if curve_analysis['curve_visualizations']:\\n",
    "    print(\"Visualizing curve detectors...\")\\n",
    "    \\n",
    "    # Show top curve detectors\\\n",
    "    images = list(curve_analysis['curve_visualizations'].values())\\n",
    "    if len(images) >= 2:\\n",
    "        fig, axes = plt.subplots(1, 2, figsize=(12, 6))\\n",
    "        \\n",
    "        # Show two curve detectors with different preferred orientations\\\n",
    "        img_keys = list(curve_analysis['curve_visualizations'].keys())[:2]\\n",
    "        for i, key in enumerate(img_keys):\\n",
    "            axes[i].imshow(curve_analysis['curve_visualizations'][key])\\n",
    "            if i < len(curve_analysis['curve_neurons']):\\n",
    "                angle = curve_analysis['curve_neurons'][i]['preferred_angle']\\n",
    "                axes[i].set_title(f'Curve Detector (prefers {angle}°)')\\n",
    "            else:\\n",
    "                axes[i].set_title(f'Curve Detector {i+1}')\\n",
    "            axes[i].axis('off')\\n",
    "        \\n",
    "        plt.suptitle('Curve Detectors in Network')\\n",
    "        plt.tight_layout()\\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Weight Visualization\\n",
    "\\n",
    "Visualize the learned weights to understand what patterns the network is looking for."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize layer weights\\n",
    "print(\"Visualizing layer weights...\")\\n",
    "weight_viz = circuits.visualize_weights(\\n",
    "    target_layer,\\n",
    "    max_channels=16\\n",
    ")\\n",
    "\\n",
    "if weight_viz['weight_grid'] is not None:\\n",
    "    plt.figure(figsize=(10, 10))\\n",
    "    plt.imshow(weight_viz['weight_grid'])\\n",
    "    plt.title(f'Weight Visualization - {target_layer}')\\n",
    "    plt.axis('off')\\n",
    "    plt.show()\\n",
    "    \\n",
    "    print(f\"Visualized {weight_viz['num_filters']} filters\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Weight Banding Analysis\\n",
    "\\n",
    "Analyze weight banding patterns - organizational structures in how weights are arranged."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze weight banding\\n",
    "print(\"Analyzing weight banding patterns...\")\\n",
    "banding = circuits.analyze_weight_banding(\\n",
    "    target_layer,\\n",
    "    num_bands=4\\n",
    ")\\n",
    "\\n",
    "if 'band_stats' in banding:\\n",
    "    print(f\"Found {len(banding['band_stats'])} weight bands\")\\n",
    "    print(\"\\nBand Statistics:\")\\n",
    "    for band_name, stats in banding['band_stats'].items():\\n",
    "        print(f\"  {band_name}: {stats['size']} neurons\")\\n",
    "    \\n",
    "    # Visualize PCA projection\\\n",
    "    if 'pca_components' in banding:\\n",
    "        plt.figure(figsize=(8, 6))\\n",
    "        scatter = plt.scatter(\\n",
    "            banding['pca_components'][:, 0],\\n",
    "            banding['pca_components'][:, 1],\\n",
    "            c=banding['bands'],\\n",
    "            cmap='tab10',\\n",
    "            alpha=0.6\\n",
    "        )\\n",
    "        plt.colorbar(scatter)\\n",
    "        plt.title(f'Weight Banding Analysis - {target_layer}')\\n",
    "        plt.xlabel('PCA Component 1')\\n",
    "        plt.ylabel('PCA Component 2')\\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Branch Specialization\\n",
    "\\n",
    "Detect when neurons self-organize into coherent groupings or specialized branches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detect branch specialization\\n",
    "print(\"Detecting branch specialization...\")\\n",
    "specialization = circuits.detect_branch_specialization(\\n",
    "    target_layer,\\n",
    "    num_samples=30\\n",
    ")\\n",
    "\\n",
    "if 'group_stats' in specialization:\\n",
    "    print(f\"Detected {specialization['num_groups']} specialized groups\")\\n",
    "    print(f\"Specialization score: {specialization['specialization_score']:.3f}\")\\n",
    "    \\n",
    "    # Visualize groupings\\\n",
    "    if 'pca_components' in specialization:\\n",
    "        plt.figure(figsize=(8, 6))\\n",
    "        scatter = plt.scatter(\\n",
    "            specialization['pca_components'][:, 0],\\n",
    "            specialization['pca_components'][:, 1],\\n",
    "            c=specialization['group_assignments'],\\n",
    "            cmap='tab10',\\n",
    "            alpha=0.6\\n",
    "        )\\n",
    "        plt.colorbar(scatter)\\n",
    "        plt.title(f'Branch Specialization - {target_layer}')\\n",
    "        plt.xlabel('PCA Component 1')\\n",
    "        plt.ylabel('PCA Component 2')\\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. High-Low Frequency Detectors\\n",
    "\\n",
    "Analyze neurons that detect transitions from high to low frequency content."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze high-low frequency detectors\\n",
    "print(\"Analyzing high-low frequency detectors...\")\\n",
    "hl_analysis = circuits.analyze_high_low_frequency_detectors(\\n",
    "    target_layer,\\n",
    "    num_samples=30\\n",
    ")\\n",
    "\\n",
    "print(f\"Found {hl_analysis['num_hl_detectors']} high-low frequency detectors\")\\n",
    "\\n",
    "if hl_analysis['hl_visualizations']:\\n",
    "    # Show HL detector visualizations\\\n",
    "    images = list(hl_analysis['hl_visualizations'].values())\\n",
    "    if images:\\n",
    "        grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2\\n",
    "        hl_grid = create_image_grid(images, grid_size)\\n",
    "        hl_array = hl_grid.detach().cpu().numpy().transpose(1, 2, 0)\\n",
    "        \\n",
    "        plt.figure(figsize=(8, 8))\\n",
    "        plt.imshow(hl_array)\\n",
    "        plt.title('High-Low Frequency Detectors')\\n",
    "        plt.axis('off')\\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Circuit Reverse Engineering\\n",
    "\\n",
    "Attempt to reverse engineer how specific neurons work by analyzing their weights and responses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reverse engineer curve algorithm (if we found curve detectors)\\n",
    "if curve_analysis['curve_neurons']:\\n",
    "    print(\"Reverse engineering curve detection algorithm...\")\\n",
    "    \\n",
    "    # Use the top curve detector\\\n",
    "    top_curve_neuron = curve_analysis['curve_neurons'][0]\\n",
    "    neuron_idx = top_curve_neuron['neuron_idx']\\n",
    "    \\n",
    "    algorithm_analysis = circuits.reverse_engineer_curve_algorithm(\\n",
    "        target_layer,\\n",
    "        neuron_idx\\n",
    "    )\\n",
    "    \\n",
    "    if algorithm_analysis:\\n",
    "        print(\"\\nAlgorithm Analysis:\")\\n",
    "        print(f\"Weight Analysis: {algorithm_analysis['weight_analysis']}\")\\n",
    "        print(f\"Curve Tests: {algorithm_analysis['curve_tests']}\")\\n",
    "        print(f\"Hypothesis: {algorithm_analysis['algorithm_hypothesis']}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Comprehensive Circuit Analysis\\n",
    "\\n",
    "Perform a comprehensive analysis combining multiple circuit detection techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Comprehensive analysis summary\\n",
    "print(\"\\n\" + \"=\"*60)\\n",
    "print(\"COMPREHENSIVE CIRCUIT ANALYSIS SUMMARY\")\\n",
    "print(\"=\"*60)\\n",
    "\\n",
    "print(f\"Layer Analyzed: {target_layer}\")\\n",
    "print(f\"Curve Detectors Found: {curve_analysis['num_curve_detectors']}\")\\n",
    "print(f\"High-Low Frequency Detectors: {hl_analysis['num_hl_detectors']}\")\\n",
    "\\n",
    "if 'band_stats' in banding:\\n",
    "    print(f\"Weight Bands Detected: {len(banding['band_stats'])}\")\\n",
    "\\n",
    "if 'group_stats' in specialization:\\n",
    "    print(f\"Specialized Groups: {specialization['num_groups']}\")\\n",
    "    print(f\"Specialization Score: {specialization['specialization_score']:.3f}\")\\n",
    "\\n",
    "print(\"\\nKey Findings:\")\\n",
    "if curve_analysis['curve_neurons']:\\n",
    "    print(\"✓ Curve detectors present - network can detect curved features\")\\n",
    "\\n",
    "if hl_analysis['num_hl_detectors'] > 0:\\n",
    "    print(\"✓ High-low frequency detectors - network responds to frequency transitions\")\\n",
    "\\n",
    "if 'band_stats' in banding:\\n",
    "    print(\"✓ Weight banding - organized weight patterns detected\")\\n",
    "\\n",
    "if 'group_stats' in specialization:\\n",
    "    print(\"✓ Branch specialization - neurons self-organize into groups\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary and Next Steps\\n",
    "\\n",
    "This notebook demonstrated the core techniques from the Thread: Circuits paper:\\n",
    "\\n",
    "✅ **Curve Detector Analysis** - Identifying neurons that respond to curved features\\n",
    "✅ **Weight Visualization** - Understanding learned patterns through weight analysis\\n",
    "✅ **Weight Banding** - Detecting organizational structures in neural networks\\n",
    "✅ **Branch Specialization** - Finding self-organized neuron groupings\\n",
    "✅ **Frequency Detectors** - Analyzing neurons that detect frequency transitions\\n",
    "✅ **Circuit Reverse Engineering** - Understanding learned algorithms\\n",
    "✅ **Comprehensive Analysis** - Combining multiple circuit analysis techniques\\n",
    "\\n",
    "### Next Steps:\\n",
    "\\n",
    "1. **Cross-Model Analysis** - Compare circuits across different architectures\\n",
    "2. **Layer Progression** - Study how circuits evolve through the network\\n",
    "3. **Task-Specific Circuits** - Analyze circuits for specific tasks or domains\\n",
    "4. **Building Blocks** - Go back to explore interpretability interfaces\\n",
    "5. **Feature Visualization** - Study basic visualization techniques"
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

with open('/mnt/okcomputer/output/notebooks/circuits.ipynb', 'w') as f:
    f.write(notebook_content)

print("Created circuits.ipynb")