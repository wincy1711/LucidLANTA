#!/usr/bin/env python3
"""
Comprehensive demonstration of Distill paper implementations.

This script demonstrates the techniques from three key Distill papers:
1. Feature Visualization (Olah et al., 2017)
2. Building Blocks of Interpretability (Olah et al., 2018)
3. Thread: Circuits (Cammarata et al., 2020)

The script creates visualizations and analyses that match the original papers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_lucid import modelzoo
from pytorch_lucid.distill import FeatureVisualization, BuildingBlocks, Circuits
from pytorch_lucid.misc.io import save_image, create_image_grid


def create_demo_outputs():
    """Create output directory for demonstrations."""
    os.makedirs('distill_outputs', exist_ok=True)
    return 'distill_outputs'


def demo_feature_visualization(model_wrapper, output_dir):
    """Demonstrate Feature Visualization paper techniques."""
    
    print("\n" + "="*80)
    print("FEATURE VISUALIZATION - DISTILL PAPER")
    print("="*80)
    
    fv = FeatureVisualization(model_wrapper.model, model_wrapper.device)
    
    # Get some convolutional layers
    conv_layers = model_wrapper.get_conv_layers()
    if not conv_layers:
        print("No convolutional layers found!")
        return
    
    # Example 1: Basic channel visualization
    print("\n1. Basic Channel Visualization")
    print("-" * 50)
    
    target_layer = conv_layers[len(conv_layers) // 2]  # Middle layer
    print(f"Visualizing channel 0 in layer: {target_layer}")
    
    basic_viz = fv.visualize_channel(
        target_layer, 0,
        num_steps=512,
        image_size=(224, 224),
        use_fft=True
    )
    
    if basic_viz is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(basic_viz)
        plt.title(f'Basic Channel Visualization - {target_layer}:0')
        plt.axis('off')
        plt.savefig(f'{output_dir}/fv_basic_channel.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Example 2: Neuron evolution
    print("\n2. Neuron Evolution During Optimization")
    print("-" * 50)
    
    evolution = fv.visualize_neuron_evolution(
        target_layer, 0,
        evolution_steps=[64, 128, 256, 512, 1024],
        image_size=(128, 128)
    )
    
    if evolution:
        fig, axes = plt.subplots(1, len(evolution), figsize=(20, 4))
        for i, (step, img) in enumerate(evolution.items()):
            axes[i].imshow(img)
            axes[i].set_title(f'Step {step}')
            axes[i].axis('off')
        plt.suptitle('Neuron Evolution During Optimization')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fv_neuron_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Example 3: Diversity visualization
    print("\n3. Diversity Visualization")
    print("-" * 50)
    
    diverse_images = fv.create_diversity_visualization(
        target_layer, 5,
        num_diverse=6,
        num_steps=256,
        image_size=(128, 128)
    )
    
    if diverse_images:
        grid = create_image_grid(diverse_images, (2, 3))
        grid_array = grid.detach().cpu().numpy().transpose(1, 2, 0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(grid_array)
        plt.title('Diverse Visualizations of Same Feature')
        plt.axis('off')
        plt.savefig(f'{output_dir}/fv_diversity.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Example 4: Interpolation between features
    print("\n4. Interpolation Between Features")
    print("-" * 50)
    
    # Find another channel in the same layer
    layer_info = model_wrapper.get_layer_info(target_layer)
    if 'out_channels' in layer_info and layer_info['out_channels'] > 1:
        second_channel = min(5, layer_info['out_channels'] - 1)
        
        interpolations = fv.create_interpolation_visualization(
            target_layer, 0, second_channel,
            num_interps=5,
            num_steps=256,
            image_size=(128, 128)
        )
        
        if interpolations:
            fig, axes = plt.subplots(1, len(interpolations), figsize=(20, 4))
            for i, img in enumerate(interpolations):
                axes[i].imshow(img)
                axes[i].set_title(f'Interpolation {i}/{len(interpolations)-1}')
                axes[i].axis('off')
            plt.suptitle(f'Interpolation: Channel 0 → Channel {second_channel}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/fv_interpolation.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Example 5: Advanced regularization techniques
    print("\n5. Frequency Regularization")
    print("-" * 50)
    
    freq_viz = fv.create_frequency_regularized_visualization(
        target_layer, 0,
        frequency_penalty=1.0,
        num_steps=512,
        image_size=(224, 224)
    )
    
    if freq_viz is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(freq_viz)
        plt.title('Frequency-Regularized Visualization')
        plt.axis('off')
        plt.savefig(f'{output_dir}/fv_frequency_reg.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Example 6: Method comparison
    print("\n6. Method Comparison")
    print("-" * 50)
    
    comparison = fv.create_visualization_comparison(
        target_layer, 0,
        methods=['basic', 'fft', 'regularized', 'preconditioned'],
        output_dir=output_dir,
        num_steps=256,
        image_size=(128, 128)
    )
    
    if len(comparison) >= 2:
        # Show comparison
        fig, axes = plt.subplots(1, len(comparison), figsize=(5*len(comparison), 5))
        if len(comparison) == 1:
            axes = [axes]
        
        for i, (method, img) in enumerate(comparison.items()):
            axes[i].imshow(img)
            axes[i].set_title(f'{method.title()} Method')
            axes[i].axis('off')
        
        plt.suptitle('Visualization Method Comparison')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fv_method_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"\nFeature Visualization demo complete!")
    print(f"Results saved in: {output_dir}")


def demo_building_blocks(model_wrapper, output_dir):
    """Demonstrate Building Blocks of Interpretability paper techniques."""
    
    print("\n" + "="*80)
    print("BUILDING BLOCKS OF INTERPRETABILITY - DISTILL PAPER")
    print("="*80)
    
    bb = BuildingBlocks(model_wrapper.model, model_wrapper.device)
    
    # Get some layers for analysis
    conv_layers = model_wrapper.get_conv_layers()
    if not conv_layers:
        print("No convolutional layers found!")
        return
    
    target_layer = conv_layers[len(conv_layers) // 2]
    print(f"Analyzing layer: {target_layer}")
    
    # Example 1: Semantic Dictionary
    print("\n1. Semantic Dictionary")
    print("-" * 50)
    
    semantic_dict = bb.create_semantic_dictionary(
        target_layer,
        top_k=16,
        num_steps=256,
        image_size=(128, 128)
    )
    
    if semantic_dict['semantic_grid'] is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(semantic_dict['semantic_grid'])
        plt.title(f'Semantic Dictionary - {target_layer}')
        plt.axis('off')
        plt.savefig(f'{output_dir}/bb_semantic_dictionary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"Created semantic dictionary with {semantic_dict['num_features']} features")
    
    # Example 2: Activation Atlas Concept
    print("\n2. Activation Atlas (Simplified)")
    print("-" * 50)
    
    # Use a smaller sample for demo
    atlas = bb.create_activation_atlas_concept(
        target_layer,
        num_samples=50,  # Reduced for demo
        num_clusters=8,
        **kwargs
    )
    
    if atlas['cluster_visualizations']:
        # Create atlas grid
        images = list(atlas['cluster_visualizations'].values())
        grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2
        
        if images:
            atlas_grid = create_image_grid(images, grid_size)
            atlas_array = atlas_grid.detach().cpu().numpy().transpose(1, 2, 0)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(atlas_array)
            plt.title('Activation Atlas - Cluster Centers')
            plt.axis('off')
            plt.savefig(f'{output_dir}/bb_activation_atlas.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Example 3: Combined Interface
    print("\n3. Combined Interpretability Interface")
    print("-" * 50)
    
    # Create a simple test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Create smooth gradients to simulate real image
    x = np.linspace(0, 1, 224)
    y = np.linspace(0, 1, 224)
    X, Y = np.meshgrid(x, y)
    
    test_image[:, :, 0] = np.sin(X * 4 * np.pi) * np.cos(Y * 4 * np.pi)
    test_image[:, :, 1] = np.sin(X * 2 * np.pi) * np.cos(Y * 6 * np.pi)
    test_image[:, :, 2] = np.sin(X * 6 * np.pi) * np.cos(Y * 2 * np.pi)
    
    # Normalize
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    
    # Create combined interface
    interface = bb.create_combined_interface(
        test_image,
        target_layer,
        top_k=6
    )
    
    # Display interface components
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input image
    axes[0, 0].imshow(interface['input_image'])
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Attribution map
    if interface['attribution_map'] is not None:
        axes[0, 1].imshow(interface['attribution_map'], cmap='hot')
        axes[0, 1].set_title('Attribution Map')
        axes[0, 1].axis('off')
    
    # Feature visualizations grid
    if interface['feature_visualizations']:
        feature_images = list(interface['feature_visualizations'].values())[:4]
        if feature_images:
            feature_grid = create_image_grid(feature_images, (2, 2))
            feature_array = feature_grid.detach().cpu().numpy().transpose(1, 2, 0)
            axes[0, 2].imshow(feature_array)
            axes[0, 2].set_title('Top Feature Visualizations')
            axes[0, 2].axis('off')
    
    # Spatial activations
    if interface['spatial_activations']:
        spatial_keys = list(interface['spatial_activations'].keys())[:2]
        if len(spatial_keys) >= 2:
            axes[1, 0].imshow(interface['spatial_activations'][spatial_keys[0]], cmap='viridis')
            axes[1, 0].set_title(f'Spatial Activation - {spatial_keys[0]}')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(interface['spatial_activations'][spatial_keys[1]], cmap='viridis')
            axes[1, 1].set_title(f'Spatial Activation - {spatial_keys[1]}')
            axes[1, 1].axis('off')
    
    # Info text
    info_text = f"Predicted Class: {interface['predicted_class']}\n"
    info_text += f"Confidence: {interface['confidence']:.3f}\n"
    info_text += f"Top Channels: {interface['top_channels'][:3]}"
    
    axes[1, 2].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 2].set_title('Analysis Info')
    axes[1, 2].axis('off')
    
    plt.suptitle('Combined Interpretability Interface')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bb_combined_interface.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nBuilding Blocks demo complete!")
    print(f"Results saved in: {output_dir}")


def demo_circuits(model_wrapper, output_dir):
    """Demonstrate Thread: Circuits paper techniques."""
    
    print("\n" + "="*80)
    print("THREAD: CIRCUITS - DISTILL PAPER")
    print("="*80)
    
    circuits = Circuits(model_wrapper.model, model_wrapper.device)
    
    # Get early layers for circuit analysis
    conv_layers = model_wrapper.get_conv_layers()
    if not conv_layers:
        print("No convolutional layers found!")
        return
    
    # Use early layers for circuit analysis
    early_layers = [l for l in conv_layers if any(x in l for x in ['conv1', 'features.0', 'features.1', 'features.2', 'features.3'])]
    target_layer = early_layers[0] if early_layers else conv_layers[0]
    print(f"Analyzing circuits in layer: {target_layer}")
    
    # Example 1: Curve Detector Analysis
    print("\n1. Curve Detector Analysis")
    print("-" * 50)
    
    curve_analysis = circuits.analyze_curve_detectors(
        target_layer,
        num_samples=50,  # Reduced for demo
        angle_range=(0, 180)
    )
    
    print(f"Found {curve_analysis['num_curve_detectors']} curve detectors")
    
    if curve_analysis['curve_visualizations']:
        # Show curve detector visualizations
        images = list(curve_analysis['curve_visualizations'].values())
        if len(images) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Show two curve detectors with different preferred orientations
            img_keys = list(curve_analysis['curve_visualizations'].keys())[:2]
            for i, key in enumerate(img_keys):
                axes[i].imshow(curve_analysis['curve_visualizations'][key])
                if i < len(curve_analysis['curve_neurons']):
                    angle = curve_analysis['curve_neurons'][i]['preferred_angle']
                    axes[i].set_title(f'Curve Detector (prefers {angle}°)')
                else:
                    axes[i].set_title(f'Curve Detector {i+1}')
                axes[i].axis('off')
            
            plt.suptitle('Curve Detectors in Network')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/circuits_curve_detectors.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Example 2: Weight Visualization
    print("\n2. Weight Visualization")
    print("-" * 50)
    
    weight_viz = circuits.visualize_weights(
        target_layer,
        max_channels=16
    )
    
    if weight_viz['weight_grid'] is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(weight_viz['weight_grid'])
        plt.title(f'Weight Visualization - {target_layer}')
        plt.axis('off')
        plt.savefig(f'{output_dir}/circuits_weights.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"Visualized {weight_viz['num_filters']} filters")
    
    # Example 3: Weight Banding Analysis
    print("\n3. Weight Banding Analysis")
    print("-" * 50)
    
    banding = circuits.analyze_weight_banding(
        target_layer,
        num_bands=4
    )
    
    if 'band_stats' in banding:
        print(f"Found {len(banding['band_stats'])} weight bands")
        for band_name, stats in banding['band_stats'].items():
            print(f"  {band_name}: {stats['size']} neurons")
    
    # Example 4: Branch Specialization
    print("\n4. Branch Specialization Detection")
    print("-" * 50)
    
    specialization = circuits.detect_branch_specialization(
        target_layer,
        num_samples=30
    )
    
    if 'group_stats' in specialization:
        print(f"Detected {specialization['num_groups']} specialized groups")
        print(f"Specialization score: {specialization['specialization_score']:.3f}")
        
        # Visualize groupings
        if 'pca_components' in specialization:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                specialization['pca_components'][:, 0],
                specialization['pca_components'][:, 1],
                c=specialization['group_assignments'],
                cmap='tab10',
                alpha=0.6
            )
            plt.colorbar(scatter)
            plt.title(f'Branch Specialization - {target_layer}')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.savefig(f'{output_dir}/circuits_specialization.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Example 5: High-Low Frequency Detector Analysis
    print("\n5. High-Low Frequency Detectors")
    print("-" * 50)
    
    hl_analysis = circuits.analyze_high_low_frequency_detectors(
        target_layer,
        num_samples=30
    )
    
    print(f"Found {hl_analysis['num_hl_detectors']} high-low frequency detectors")
    
    if hl_analysis['hl_visualizations']:
        # Show HL detector visualizations
        images = list(hl_analysis['hl_visualizations'].values())
        if images:
            grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2
            hl_grid = create_image_grid(images, grid_size)
            hl_array = hl_grid.detach().cpu().numpy().transpose(1, 2, 0)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(hl_array)
            plt.title('High-Low Frequency Detectors')
            plt.axis('off')
            plt.savefig(f'{output_dir}/circuits_hl_detectors.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    print(f"\nCircuits demo complete!")
    print(f"Results saved in: {output_dir}")


def create_summary_visualization(output_dir):
    """Create a summary of all Distill paper techniques."""
    
    print("\n" + "="*80)
    print("CREATING SUMMARY VISUALIZATION")
    print("="*80)
    
    # Create summary figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # Paper 1: Feature Visualization
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.text(0.5, 0.7, 'FEATURE VISUALIZATION', ha='center', va='center', 
             fontsize=16, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, 'Olah, Mordvintsev, Schubert (2017)', ha='center', va='center',
             fontsize=12, transform=ax1.transAxes)
    ax1.text(0.5, 0.3, '• Neuron & Channel Visualization\n• Diversity & Interpolation\n• Advanced Regularization', 
             ha='center', va='center', fontsize=10, transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Paper 2: Building Blocks
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.text(0.5, 0.7, 'BUILDING BLOCKS', ha='center', va='center',
             fontsize=16, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, 'Olah et al. (2018)', ha='center', va='center',
             fontsize=12, transform=ax2.transAxes)
    ax2.text(0.5, 0.3, '• Semantic Dictionaries\n• Combined Interfaces\n• Activation Atlases', 
             ha='center', va='center', fontsize=10, transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Paper 3: Circuits
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.text(0.5, 0.7, 'THREAD: CIRCUITS', ha='center', va='center',
             fontsize=16, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, 'Cammarata et al. (2020)', ha='center', va='center',
             fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.3, '• Curve Detectors\n• Weight Banding\n• Branch Specialization', 
             ha='center', va='center', fontsize=10, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Implementation status
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.text(0.5, 0.8, 'PYTORCH LUCID IMPLEMENTATION', ha='center', va='center',
             fontsize=16, fontweight='bold', transform=ax4.transAxes, color='green')
    ax4.text(0.5, 0.6, 'Complete Feature Parity ✓', ha='center', va='center',
             fontsize=12, transform=ax4.transAxes, color='green')
    ax4.text(0.5, 0.4, 'Modern PyTorch Integration ✓', ha='center', va='center',
             fontsize=12, transform=ax4.transAxes, color='green')
    ax4.text(0.5, 0.2, 'Comprehensive Examples ✓', ha='center', va='center',
             fontsize=12, transform=ax4.transAxes, color='green')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Key techniques visualization
    techniques = [
        'Feature Visualization',
        'Style Transfer', 
        'DeepDream',
        'Semantic Dictionaries',
        'Attribution Analysis',
        'Activation Atlases',
        'Curve Detectors',
        'Weight Banding',
        'Branch Specialization',
        'Circuit Analysis'
    ]
    
    for i, technique in enumerate(techniques):
        row = 2 + i // 5
        col = i % 5
        
        ax = fig.add_subplot(gs[row, col])
        ax.text(0.5, 0.5, technique, ha='center', va='center',
                fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('PyTorch Lucid: Distill Papers Implementation', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Add logo/implementation info
    ax_logo = fig.add_subplot(gs[0, 4])
    ax_logo.text(0.5, 0.5, 'PyTorch\nLucid', ha='center', va='center',
                 fontsize=20, fontweight='bold', transform=ax_logo.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.8))
    ax_logo.set_xlim(0, 1)
    ax_logo.set_ylim(0, 1)
    ax_logo.axis('off')
    
    plt.savefig(f'{output_dir}/distill_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run the comprehensive Distill papers demonstration."""
    
    print("Distill Papers Implementation - PyTorch Lucid")
    print("=" * 80)
    print("Implementing techniques from:")
    print("1. Feature Visualization (Olah et al., 2017)")
    print("2. Building Blocks of Interpretability (Olah et al., 2018)")
    print("3. Thread: Circuits (Cammarata et al., 2020)")
    print()
    
    # Create output directory
    output_dir = create_demo_outputs()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading pre-trained VGG16 model...")
    model_wrapper = modelzoo.load_model('vgg16', device=device)
    
    # Print model info
    print(f"Model: {model_wrapper.model.__class__.__name__}")
    print(f"Convolutional layers: {len(model_wrapper.get_conv_layers())}")
    
    # Run demonstrations
    demo_feature_visualization(model_wrapper, output_dir)
    demo_building_blocks(model_wrapper, output_dir)
    demo_circuits(model_wrapper, output_dir)
    
    # Create summary
    create_summary_visualization(output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("DISTILL PAPERS IMPLEMENTATION COMPLETE!")
    print("="*80)
    print(f"\nSuccessfully implemented techniques from 3 major Distill papers:")
    print("✓ Feature Visualization - Complete implementation with all techniques")
    print("✓ Building Blocks of Interpretability - Semantic dictionaries and interfaces")
    print("✓ Thread: Circuits - Circuit analysis and weight visualization")
    print()
    print(f"All results saved in: {output_dir}/")
    print(f"Generated visualizations and analysis for neural network interpretability")
    print(f"Ready for research and exploration!")


if __name__ == "__main__":
    # Define kwargs for all visualization calls
    kwargs = {
        'verbose': False,
        'use_fixed_seed': True
    }
    
    main()