#!/usr/bin/env python3
"""
Basic visualization example for PyTorch Lucid.

This example demonstrates how to:
1. Load a pre-trained model
2. Visualize individual neurons and channels
3. Create a grid visualization of multiple features
4. Save and display the results
"""

import torch
import numpy as np
from pytorch_lucid import modelzoo, optvis, recipes, misc
import matplotlib.pyplot as plt


def main():
    """Run the basic visualization example."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a pre-trained model
    print("\nLoading pre-trained VGG16 model...")
    model_wrapper = modelzoo.load_model('vgg16', device=device)
    
    # Print model summary
    print("\nModel Summary:")
    modelzoo.print_model_summary(model_wrapper)
    
    # Get some convolutional layers for visualization
    conv_layers = model_wrapper.get_conv_layers()
    print(f"\nFound {len(conv_layers)} convolutional layers")
    
    # Select a middle layer for visualization
    target_layer = conv_layers[len(conv_layers) // 2]  # Middle layer
    print(f"Target layer for visualization: {target_layer}")
    
    # Example 1: Visualize individual neurons
    print("\n" + "="*50)
    print("Example 1: Visualizing Individual Neurons")
    print("="*50)
    
    neuron_indices = [0, 10, 20, 30]
    neuron_visualizations = recipes.visualize_neurons(
        model_wrapper,
        target_layer,
        neuron_indices,
        num_steps=256,
        image_size=(128, 128),
        device=device,
        verbose=True
    )
    
    # Display neuron visualizations
    fig, axes = plt.subplots(1, len(neuron_indices), figsize=(15, 4))
    for i, (neuron_idx, img) in enumerate(neuron_visualizations.items()):
        axes[i].imshow(img)
        axes[i].set_title(f"Neuron {neuron_idx}")
        axes[i].axis('off')
    plt.suptitle(f"Neuron Visualizations - Layer {target_layer}")
    plt.tight_layout()
    plt.savefig('neuron_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 2: Visualize channels
    print("\n" + "="*50)
    print("Example 2: Visualizing Channels")
    print("="*50)
    
    channel_indices = [0, 5, 10, 15, 20, 25]
    channel_visualizations = recipes.visualize_channels(
        model_wrapper,
        target_layer,
        channel_indices,
        num_steps=256,
        image_size=(128, 128),
        device=device,
        verbose=True
    )
    
    # Display channel visualizations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, (channel_idx, img) in enumerate(channel_visualizations.items()):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(f"Channel {channel_idx}")
            axes[i].axis('off')
    plt.suptitle(f"Channel Visualizations - Layer {target_layer}")
    plt.tight_layout()
    plt.savefig('channel_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 3: Create feature visualization grid
    print("\n" + "="*50)
    print("Example 3: Feature Visualization Grid")
    print("="*50)
    
    feature_grid = recipes.create_feature_visualization_grid(
        model_wrapper,
        target_layer,
        num_channels=9,
        grid_size=(3, 3),
        num_steps=256,
        image_size=(128, 128),
        device=device
    )
    
    # Display feature grid
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_grid)
    plt.title(f"Feature Visualization Grid - Layer {target_layer}")
    plt.axis('off')
    plt.savefig('feature_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 4: DeepDream-style visualization
    print("\n" + "="*50)
    print("Example 4: DeepDream-style Visualization")
    print("="*50)
    
    # Use a deeper layer for DeepDream
    deep_layer = conv_layers[-3]  # Third-to-last layer
    print(f"DeepDream layer: {deep_layer}")
    
    deepdream_images = optvis.deepdream_visualization(
        model_wrapper.model,
        deep_layer,
        thresholds=(64, 128, 256, 512),
        device=device,
        verbose=True
    )
    
    # Display DeepDream evolution
    fig, axes = plt.subplots(1, len(deepdream_images), figsize=(16, 4))
    for i, img in enumerate(deepdream_images):
        axes[i].imshow(img)
        axes[i].set_title(f"Step {(i+1) * 128}")
        axes[i].axis('off')
    plt.suptitle(f"DeepDream Evolution - Layer {deep_layer}")
    plt.tight_layout()
    plt.savefig('deepdream_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 5: Layer evolution visualization
    print("\n" + "="*50)
    print("Example 5: Layer Evolution Visualization")
    print("="*50)
    
    evolution_steps = [32, 64, 128, 256, 512]
    evolution_visualizations = recipes.visualize_layer_evolution(
        model_wrapper,
        target_layer,
        channel_idx=0,
        save_intervals=evolution_steps,
        image_size=(128, 128),
        device=device
    )
    
    # Display evolution
    fig, axes = plt.subplots(1, len(evolution_visualizations), figsize=(16, 4))
    for i, (step, img) in enumerate(evolution_visualizations.items()):
        axes[i].imshow(img)
        axes[i].set_title(f"Step {step}")
        axes[i].axis('off')
    plt.suptitle(f"Layer Evolution - Channel 0, Layer {target_layer}")
    plt.tight_layout()
    plt.savefig('layer_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("Visualization Complete!")
    print("="*50)
    print("Generated files:")
    print("- neuron_visualizations.png")
    print("- channel_visualizations.png") 
    print("- feature_grid.png")
    print("- deepdream_evolution.png")
    print("- layer_evolution.png")


if __name__ == "__main__":
    main()