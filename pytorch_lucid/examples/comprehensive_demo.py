#!/usr/bin/env python3
"""
Comprehensive demonstration of PyTorch Lucid functionality.

This script demonstrates all major features of PyTorch Lucid:
1. Model loading and analysis
2. Basic feature visualizations
3. Advanced optimization techniques
4. Style transfer
5. DeepDream visualizations
6. Batch processing and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pytorch_lucid import modelzoo, optvis, recipes, misc


def create_demo_images():
    """Create example images for demonstrations."""
    
    # Create output directory
    os.makedirs('demo_outputs', exist_ok=True)
    
    # Content image: geometric shapes
    content = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # White circle
    center = (128, 128)
    radius = 60
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    content[mask] = [255, 255, 255]
    
    # Red rectangle
    content[50:100, 50:200] = [255, 0, 0]
    # Green rectangle  
    content[150:200, 50:200] = [0, 255, 0]
    
    Image.fromarray(content).save('demo_outputs/content_image.png')
    
    # Style image: colorful patterns
    style = np.zeros((256, 256, 3), dtype=np.uint8)
    x = np.linspace(0, 4*np.pi, 256)
    y = np.linspace(0, 4*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    
    style[:, :, 0] = (np.sin(X) * 127 + 128).astype(np.uint8)
    style[:, :, 1] = (np.cos(Y) * 127 + 128).astype(np.uint8)
    style[:, :, 2] = (np.sin(X + Y) * 127 + 128).astype(np.uint8)
    
    Image.fromarray(style).save('demo_outputs/style_image.png')
    
    return 'demo_outputs/content_image.png', 'demo_outputs/style_image.png'


def demo_model_analysis(model_wrapper):
    """Demonstrate model analysis capabilities."""
    
    print("\n" + "="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    
    # Print model summary
    print("\nModel Summary:")
    modelzoo.print_model_summary(model_wrapper)
    
    # Get feature layers
    feature_layers = modelzoo.get_feature_layers(model_wrapper)
    print(f"\nFound {len(feature_layers)} feature layers")
    print("First 10 feature layers:")
    for i, layer in enumerate(feature_layers[:10]):
        print(f"  {i+1:2d}: {layer}")
    
    return feature_layers


def demo_basic_visualizations(model_wrapper, feature_layers):
    """Demonstrate basic visualization techniques."""
    
    print("\n" + "="*60)
    print("BASIC VISUALIZATIONS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Select a middle layer for demonstration
    target_layer = feature_layers[len(feature_layers) // 2]
    print(f"\nUsing layer: {target_layer}")
    
    # Example 1: Single neuron visualization
    print("\n1. Single Neuron Visualization")
    print("-" * 40)
    
    neuron_images = optvis.visualize_neuron(
        model_wrapper.model,
        target_layer,
        neuron_idx=5,
        num_steps=256,
        image_size=(128, 128),
        device=device,
        verbose=True
    )
    
    if neuron_images:
        plt.figure(figsize=(6, 6))
        plt.imshow(neuron_images[-1])
        plt.title(f'Single Neuron - Layer {target_layer}')
        plt.axis('off')
        plt.savefig('demo_outputs/single_neuron.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Example 2: Channel visualization grid
    print("\n2. Channel Visualization Grid")
    print("-" * 40)
    
    feature_grid = recipes.create_feature_visualization_grid(
        model_wrapper,
        target_layer,
        num_channels=9,
        grid_size=(3, 3),
        num_steps=256,
        image_size=(128, 128),
        device=device
    )
    
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_grid)
    plt.title(f'Channel Grid - Layer {target_layer}')
    plt.axis('off')
    plt.savefig('demo_outputs/channel_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 3: Layer evolution
    print("\n3. Layer Evolution")
    print("-" * 40)
    
    evolution_steps = [64, 128, 256, 512]
    evolution_viz = recipes.visualize_layer_evolution(
        model_wrapper,
        target_layer,
        channel_idx=0,
        save_intervals=evolution_steps,
        image_size=(128, 128),
        device=device
    )
    
    fig, axes = plt.subplots(1, len(evolution_viz), figsize=(16, 4))
    for i, (step, img) in enumerate(evolution_viz.items()):
        axes[i].imshow(img)
        axes[i].set_title(f'Step {step}')
        axes[i].axis('off')
    plt.suptitle(f'Layer Evolution - Channel 0, {target_layer}')
    plt.tight_layout()
    plt.savefig('demo_outputs/layer_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_advanced_optimizations(model_wrapper, feature_layers):
    """Demonstrate advanced optimization techniques."""
    
    print("\n" + "="*60)
    print("ADVANCED OPTIMIZATIONS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_layer = feature_layers[len(feature_layers) // 2]
    
    # Example 1: Different parameterizations
    print("\n1. Parameterization Comparison")
    print("-" * 40)
    
    # Direct pixel parameterization
    direct_param = optvis.param.image((1, 3, 128, 128), decorrelate=True, sigmoid=True)
    direct_images = optvis.render_vis(
        model_wrapper.model,
        optvis.objectives.channel(target_layer, 0),
        param_f=direct_param,
        thresholds=(256,),
        device=device
    )
    
    # FFT parameterization
    fft_param = optvis.param.fft_image((1, 3, 128, 128))
    fft_images = optvis.render_vis(
        model_wrapper.model,
        optvis.objectives.channel(target_layer, 0),
        param_f=fft_param,
        thresholds=(256,),
        device=device
    )
    
    # Display comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if direct_images:
        axes[0].imshow(direct_images[-1])
        axes[0].set_title('Direct Pixel Parameterization')
    if fft_images:
        axes[1].imshow(fft_images[-1])
        axes[1].set_title('FFT Parameterization')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('Parameterization Comparison')
    plt.tight_layout()
    plt.savefig('demo_outputs/param_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 2: Different transform configurations
    print("\n2. Transform Configuration Comparison")
    print("-" * 40)
    
    # Minimal transforms
    minimal_transforms = optvis.transform.standard_transforms(jitter=4, scale=(0.95, 1.05))
    minimal_images = optvis.render_vis(
        model_wrapper.model,
        optvis.objectives.channel(target_layer, 1),
        transforms=minimal_transforms,
        thresholds=(256,),
        device=device
    )
    
    # Aggressive transforms
    aggressive_transforms = optvis.transform.standard_transforms(
        jitter=32, scale=(0.8, 1.2), rotate=(-15, 15), pad=32
    )
    aggressive_images = optvis.render_vis(
        model_wrapper.model,
        optvis.objectives.channel(target_layer, 1),
        transforms=aggressive_transforms,
        thresholds=(256,),
        device=device
    )
    
    # Display comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if minimal_images:
        axes[0].imshow(minimal_images[-1])
        axes[0].set_title('Minimal Transforms')
    if aggressive_images:
        axes[1].imshow(aggressive_images[-1])
        axes[1].set_title('Aggressive Transforms')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('Transform Configuration Comparison')
    plt.tight_layout()
    plt.savefig('demo_outputs/transform_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_style_transfer(model_wrapper):
    """Demonstrate style transfer functionality."""
    
    print("\n" + "="*60)
    print("STYLE TRANSFER")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create demo images
    content_path, style_path = create_demo_images()
    
    # Load images
    content_tensor = misc.io.load_image(content_path, size=(224, 224), device=device)
    style_tensor = misc.io.load_image(style_path, size=(224, 224), device=device)
    
    # Display input images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    content_display = content_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(content_display)
    axes[0].set_title('Content Image')
    
    style_display = style_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(style_display)
    axes[1].set_title('Style Image')
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('demo_outputs/style_transfer_inputs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Perform style transfer
    print("\nPerforming style transfer...")
    
    # Define style and content layers for VGG19
    style_layers = {
        'features.1': 1.0,   # conv1_1
        'features.6': 0.75,  # conv2_1
        'features.11': 0.2,  # conv3_1
        'features.20': 0.2,  # conv4_1
        'features.29': 0.2   # conv5_1
    }
    
    content_layers = {
        'features.22': 1.0  # conv4_2
    }
    
    # Create style transfer objective
    from pytorch_lucid.optvis.style import StyleTransferObjective
    
    style_objective = StyleTransferObjective(
        style_weight=1e6,
        content_weight=1.0,
        style_layers=style_layers,
        content_layers=content_layers
    )
    
    # Extract activations and perform style transfer
    result = recipes.style_transfer_visualization(
        model_wrapper,
        content_tensor,
        style_tensor,
        style_objective,
        num_steps=300,
        device=device
    )
    
    if result:
        plt.figure(figsize=(10, 10))
        plt.imshow(result)
        plt.title('Style Transfer Result')
        plt.axis('off')
        plt.savefig('demo_outputs/style_transfer_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save result
        misc.io.save_image(
            torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0),
            'demo_outputs/style_transfer_output.png'
        )


def demo_deepdream(model_wrapper, feature_layers):
    """Demonstrate DeepDream functionality."""
    
    print("\n" + "="*60)
    print("DEEPDREAM")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use deeper layers for DeepDream
    deep_layers = [f for f in feature_layers if 'layer4' in f or 'features.29' in f]
    if not deep_layers:
        deep_layers = feature_layers[-3:]
    
    target_layer = deep_layers[0] if deep_layers else feature_layers[-1]
    print(f"Using DeepDream layer: {target_layer}")
    
    # Basic DeepDream
    print("\n1. Basic DeepDream")
    print("-" * 40)
    
    deepdream_images = optvis.deepdream_visualization(
        model_wrapper.model,
        target_layer,
        thresholds=(128, 256, 512),
        device=device,
        verbose=True
    )
    
    fig, axes = plt.subplots(1, len(deepdream_images), figsize=(16, 4))
    for i, img in enumerate(deepdream_images):
        axes[i].imshow(img)
        axes[i].set_title(f'Step {(i+1) * 128}')
        axes[i].axis('off')
    plt.suptitle(f'DeepDream Evolution - Layer {target_layer}')
    plt.tight_layout()
    plt.savefig('demo_outputs/deepdream_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Multi-layer DeepDream
    print("\n2. Multi-layer DeepDream")
    print("-" * 40)
    
    if len(deep_layers) >= 3:
        multi_layer_result = recipes.deepdream_multilayer(
            model_wrapper,
            deep_layers[:3],
            weights=[1.0, 0.5, 0.25],
            num_steps=512,
            image_size=(224, 224),
            device=device
        )
        
        if multi_layer_result is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(multi_layer_result)
            plt.title('Multi-layer DeepDream')
            plt.axis('off')
            plt.savefig('demo_outputs/multilayer_deepdream.png', dpi=150, bbox_inches='tight')
            plt.show()


def demo_batch_processing(model_wrapper, feature_layers):
    """Demonstrate batch processing capabilities."""
    
    print("\n" + "="*60)
    print("BATCH PROCESSING")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Select a layer with many channels
    conv_layers = model_wrapper.get_conv_layers()
    target_layer = conv_layers[len(conv_layers) // 2]
    
    print(f"\nBatch visualizing channels in layer: {target_layer}")
    
    # Batch visualize first 8 channels
    channel_indices = list(range(8))
    
    batch_results = recipes.batch_visualize_features(
        model_wrapper,
        target_layer,
        channel_indices,
        output_dir='demo_outputs/batch_visualizations',
        num_steps=256,
        image_size=(128, 128),
        device=device
    )
    
    print(f"\nSuccessfully visualized {len(batch_results)} channels")
    
    # Create a summary grid
    if batch_results:
        from pytorch_lucid.misc.io import create_image_grid
        
        images = [batch_results[idx] for idx in channel_indices if idx in batch_results]
        if images:
            grid = create_image_grid(images, (2, 4))
            grid_np = grid.detach().cpu().numpy().transpose(1, 2, 0)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(grid_np)
            plt.title(f'Batch Visualization Results - {target_layer}')
            plt.axis('off')
            plt.savefig('demo_outputs/batch_summary.png', dpi=150, bbox_inches='tight')
            plt.show()


def demo_feature_analysis(model_wrapper, feature_layers):
    """Demonstrate feature analysis capabilities."""
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Select a layer for analysis
    conv_layers = model_wrapper.get_conv_layers()
    target_layer = conv_layers[len(conv_layers) // 2]
    
    print(f"\nAnalyzing feature diversity in layer: {target_layer}")
    
    # Analyze feature diversity
    diversity_results = recipes.analyze_feature_diversity(
        model_wrapper,
        target_layer,
        num_features=16,
        num_steps=256,
        image_size=(128, 128),
        device=device
    )
    
    if diversity_results['diversity_grid'] is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(diversity_results['diversity_grid'])
        plt.title(f'Feature Diversity Analysis - {target_layer}')
        plt.axis('off')
        plt.savefig('demo_outputs/feature_diversity.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"Analyzed {diversity_results['num_features']} features")
    
    # Compare different layers
    print("\nComparing feature representations across layers...")
    
    # Select a few representative layers
    layer_indices = [len(conv_layers) // 4, len(conv_layers) // 2, 3 * len(conv_layers) // 4]
    selected_layers = [conv_layers[i] for i in layer_indices if i < len(conv_layers)]
    
    layer_comparison = recipes.visualize_layer_progression(
        model_wrapper,
        selected_layers,
        feature_idx=0,
        feature_type='channel',
        num_steps=256,
        image_size=(128, 128),
        device=device
    )
    
    # Display comparison
    fig, axes = plt.subplots(1, len(layer_comparison), figsize=(15, 5))
    for i, (layer_name, img) in enumerate(layer_comparison.items()):
        axes[i].imshow(img)
        axes[i].set_title(f'Layer: {layer_name.split(".")[-1]}')
        axes[i].axis('off')
    plt.suptitle('Feature Evolution Across Layers')
    plt.tight_layout()
    plt.savefig('demo_outputs/layer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Run the comprehensive demonstration."""
    
    print("PyTorch Lucid - Comprehensive Demonstration")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading VGG16 model...")
    model_wrapper = modelzoo.load_model('vgg16', device=device)
    
    # Get feature layers
    feature_layers = demo_model_analysis(model_wrapper)
    
    # Run demonstrations
    demo_basic_visualizations(model_wrapper, feature_layers)
    demo_advanced_optimizations(model_wrapper, feature_layers)
    demo_style_transfer(model_wrapper)
    demo_deepdream(model_wrapper, feature_layers)
    demo_batch_processing(model_wrapper, feature_layers)
    demo_feature_analysis(model_wrapper, feature_layers)
    
    # Summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nGenerated files in demo_outputs/:")
    
    output_files = [
        'single_neuron.png',
        'channel_grid.png', 
        'layer_evolution.png',
        'param_comparison.png',
        'transform_comparison.png',
        'style_transfer_inputs.png',
        'style_transfer_result.png',
        'style_transfer_output.png',
        'deepdream_evolution.png',
        'multilayer_deepdream.png',
        'batch_summary.png',
        'feature_diversity.png',
        'layer_comparison.png',
        'content_image.png',
        'style_image.png'
    ]
    
    for file in output_files:
        if os.path.exists(f'demo_outputs/{file}'):
            print(f"  ✓ {file}")
    
    print(f"\nTotal demonstrations completed: 6")
    print(f"Model used: {model_wrapper.model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Feature layers analyzed: {len(feature_layers)}")


if __name__ == "__main__":
    main()